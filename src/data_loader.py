import numpy as np
import torch
import torch.nn as nn
import jax, jax.numpy as jnp, jax.nn as jnn
import flax
from flax import nnx

import minari

dataset = minari.load_dataset("mujoco/ant/simple-v0", download=True)

def _coerce_observations(obs_container):
    """
    Return obs as float32 ndarray of shape [T+1, obs_dim], handling:
      - dicts with keys like 'qpos'/'qvel' (concatenate),
      - dicts with 'observation'/'obs'/'state',
      - plain lists/ndarrays already flattened.
    """
    # Case 1: dict of arrays
    if isinstance(obs_container, dict):
        # Prefer explicit qpos/qvel if available
        if "qpos" in obs_container and "qvel" in obs_container:
            qpos = np.asarray(obs_container["qpos"], dtype=np.float32)
            qvel = np.asarray(obs_container["qvel"], dtype=np.float32)
            return np.concatenate([qpos, qvel], axis=-1)

        # Otherwise, try common flat keys
        for k in ("observation", "obs", "state"):
            if k in obs_container:
                return np.asarray(obs_container[k], dtype=np.float32)

        # Fall back: pick the first key deterministically and warn
        first_key = sorted(obs_container.keys())[0]
        print(f"[data_loader] Warning: using obs['{first_key}'] (available keys: {list(obs_container.keys())})")
        return np.asarray(obs_container[first_key], dtype=np.float32)

    # Case 2: already an array/list
    return np.asarray(obs_container, dtype=np.float32)


def build_episodes_from_minari(minari_dataset):
    """
    Extract episodes as plain numpy arrays (observations, actions).

    Ensures:
      obs: [T+1, obs_dim]
      acts: [T,   act_dim]
    If lengths mismatch, truncate conservatively to align (Minari should usually be aligned).
    """
    episodes = []
    for idx, ep in enumerate(minari_dataset.iterate_episodes()):
        obs = _coerce_observations(ep.observations)         # [T+1, obs_dim] or [T, obs_dim]
        acts = np.asarray(ep.actions, dtype=np.float32)     # [T, act_dim]

        # Align lengths safely
        # Expect obs_len = acts_len + 1; otherwise, truncate to the shortest consistent pair.
        obs_len = obs.shape[0]
        act_len = acts.shape[0]
        if obs_len != act_len + 1:
            # Try to coerce to aligned lengths
            T = min(act_len, max(0, obs_len - 1))
            if T == 0:
                print(f"[data_loader] Warning: episode {idx} too short after alignment; skipping.")
                continue
            obs = obs[:T + 1]
            acts = acts[:T]

        episodes.append({"observations": obs, "actions": acts})

    if not episodes:
        raise RuntimeError("[data_loader] No usable episodes found after parsing/alignment.")
    # Optional: brief summary
    o_dim = episodes[0]["observations"].shape[-1]
    a_dim = episodes[0]["actions"].shape[-1]
    print(f"[data_loader] Parsed {len(episodes)} episodes | obs_dim={o_dim} act_dim={a_dim}")
    return episodes


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=32,
        sequence_length=10,
        shuffle=True,
        use_jax=True,
        # episode-split + normalization controls:
        episodes=None,
        episode_indices=None,
        norm_stats=None,           # (obs_mean, obs_std, action_mean, action_std)
        fit_normalization=True,    # compute stats from selected episodes
    ):
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.shuffle = bool(shuffle)
        self.use_jax = bool(use_jax)

        # 1) materialize episodes once
        if episodes is None:
            episodes = build_episodes_from_minari(dataset)

        # 2) restrict to selected split
        if episode_indices is not None:
            episodes = [episodes[i] for i in episode_indices]
        self.episodes = episodes

        # 3) normalization (train computes; val/test reuse)
        if fit_normalization and norm_stats is None:
            all_obs = np.concatenate([ep["observations"][:-1] for ep in self.episodes], axis=0)  # [sum(T), D]
            all_actions = np.concatenate([ep["actions"] for ep in self.episodes], axis=0)        # [sum(T), A]
            self.obs_mean = all_obs.mean(axis=0, keepdims=True)
            self.obs_std  = all_obs.std(axis=0, keepdims=True) + 1e-8
            self.action_mean = all_actions.mean(axis=0, keepdims=True)
            self.action_std  = all_actions.std(axis=0, keepdims=True) + 1e-8
        else:
            if norm_stats is None:
                raise ValueError("norm_stats must be provided when fit_normalization=False")
            self.obs_mean, self.obs_std, self.action_mean, self.action_std = norm_stats

        # 4) apply normalization to a copy of episode arrays
        norm_eps = []
        for ep in self.episodes:
            obs = (ep["observations"] - self.obs_mean) / self.obs_std
            act = (ep["actions"] - self.action_mean) / self.action_std
            norm_eps.append({"observations": obs.astype(np.float32),
                             "actions": act.astype(np.float32)})
        self.episodes = norm_eps

        # 5) slice each episode into overlapping sequences (time-first)
        #    each seq: states [S+1, D], actions [S, A]
        S = self.sequence_length
        self.sequences = []
        for ep in self.episodes:
            obs = ep["observations"]  # [Te+1, D]
            act = ep["actions"]       # [Te,   A]
            Te = act.shape[0]
            if Te < S:
                continue
            for start in range(Te - S + 1):
                end = start + S
                self.sequences.append({
                    "states":  obs[start:end+1, :],  # [S+1, D]
                    "actions": act[start:end,   :],  # [S,   A]
                })

        # 6) index list for batching
        self.indices = np.arange(len(self.sequences), dtype=np.int64)
        self._reset_epoch_permutation()

    def _reset_epoch_permutation(self):
        self._perm = self.indices.copy()
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self._perm)
        self._cursor = 0

    def __len__(self):
        # number of full batches (drop last)
        if self.batch_size <= 0:
            return 0
        return len(self.sequences) // self.batch_size

    def __iter__(self):
        self._reset_epoch_permutation()
        B = self.batch_size
        while self._cursor + B <= len(self._perm):
            batch_idx = self._perm[self._cursor:self._cursor+B]
            self._cursor += B

            # gather sequences
            seqs = [self.sequences[i] for i in batch_idx]
            # stack with TIME FIRST, then BATCH
            states  = np.stack([s["states"]  for s in seqs], axis=1)  # [S+1, B, D]
            actions = np.stack([s["actions"] for s in seqs], axis=1)  # [S,   B, A]

            # ensure exact dtypes/shapes; NO leading singleton axes
            states  = states.astype(np.float32, copy=False)
            actions = actions.astype(np.float32, copy=False)

            yield {"states": states, "actions": actions}

    # convenience for quick probes
    def get_random_batch(self, batch_size=None):
        B = batch_size or self.batch_size
        rng = np.random.default_rng()
        if len(self.sequences) < B:
            raise RuntimeError(f"Not enough sequences ({len(self.sequences)}) for batch_size={B}")
        idx = rng.choice(len(self.sequences), size=B, replace=False)
        seqs = [self.sequences[i] for i in idx]
        states  = np.stack([s["states"]  for s in seqs], axis=1)  # [S+1, B, D]
        actions = np.stack([s["actions"] for s in seqs], axis=1)  # [S,   B, A]
        return {"states": states.astype(np.float32), "actions": actions.astype(np.float32)}

    def get_norm_stats(self):
        return (self.obs_mean, self.obs_std, self.action_mean, self.action_std)



if __name__ == "__main__":
    # Example usage
    dataloader = DataLoader(
        dataset, 
        batch_size=128, 
        sequence_length=10,
        shuffle=True, 
        use_jax=True
    )
    
    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  States shape: {batch['states'].shape}")
        print(f"  Actions shape: {batch['actions'].shape}")
        if batch_idx >= 2:  # Just show first few batches
            break
    
    # Get random batch
    random_batch = dataloader.get_random_batch(64)
    print(f"\nRandom batch:")
    print(f"  States shape: {random_batch['states'].shape}")
    print(f"  Actions shape: {random_batch['actions'].shape}")