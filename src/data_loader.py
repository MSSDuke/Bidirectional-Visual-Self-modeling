from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import jax, jax.numpy as jnp, jax.nn as jnn
import flax
from flax import nnx

from pathlib import Path

def _resize_hw(x: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    H, W = x.shape[0], x.shape[1]
    ih = (np.arange(new_h) * (H / new_h)).astype(np.int32)
    iw = (np.arange(new_w) * (W / new_w)).astype(np.int32)
    return x[ih][:, iw]

class NPZSequenceDatasetJAX:
    """
    Loads .npz with:
      states:       (N, D)
      actions:      (N, A)
      spectrograms: (N, H, W, 3)

    Builds sequence starts per 512-step episode, so each sample returns:
      states_seq:  [S+1, D]
      actions_seq: [S,   A]
      specs_seq:   [S,   3, H, W]   (CHW)
    """
    def __init__(
        self,
        path_like: str | Path,
        steps_per_episode: int = 512,
        sequence_length: int = 10,
        normalize_sa: bool = True,
        specs_scale_01: bool = True,
        specs_downsample_hw: tuple[int, int] | None = None,
    ):
        self.path = Path(path_like)
        self.E = int(steps_per_episode)
        self.S = int(sequence_length)
        self.normalize_sa = bool(normalize_sa)
        self.specs_scale_01 = bool(specs_scale_01)
        self.specs_downsample_hw = specs_downsample_hw

        npz_path = self.path if self.path.is_file() else (self.path / "rollout.npz")
        with np.load(npz_path, allow_pickle=False) as f:
            states = f["states"].astype(np.float32, copy=False)
            actions = f["actions"].astype(np.float32, copy=False)
            if "spectrograms" not in f.files:
                raise ValueError("spectrograms missing in npz.")
            specs = f["spectrograms"]

        N = min(states.shape[0], actions.shape[0], specs.shape[0])
        n_eps = N // self.E
        if n_eps <= 0:
            raise ValueError("Not enough steps for a full episode.")
        N_use = n_eps * self.E
        states, actions, specs = states[:N_use], actions[:N_use], specs[:N_use]

        if self.specs_downsample_hw is not None:
            new_h, new_w = self.specs_downsample_hw
            out = np.empty((specs.shape[0], new_h, new_w, specs.shape[3]), dtype=specs.dtype)
            for i in range(specs.shape[0]):
                out[i] = _resize_hw(specs[i], new_h, new_w)
            specs = out

        self.states, self.actions, self.specs = states, actions, specs
        self.D, self.A = states.shape[1], actions.shape[1]
        self.H, self.W = specs.shape[1], specs.shape[2]

        if self.normalize_sa:
            self.obs_mean = states.mean(axis=0, keepdims=True)
            self.obs_std = states.std(axis=0, keepdims=True) + 1e-8
            self.act_mean = actions.mean(axis=0, keepdims=True)
            self.act_std = actions.std(axis=0, keepdims=True) + 1e-8
        else:
            self.obs_mean = np.zeros((1, self.D), np.float32)
            self.obs_std = np.ones((1, self.D), np.float32)
            self.act_mean = np.zeros((1, self.A), np.float32)
            self.act_std = np.ones((1, self.A), np.float32)

        starts = []
        for e in range(n_eps):
            base = e * self.E
            max_start = self.E - (self.S + 1)
            for s in range(max_start + 1):
                starts.append(base + s)
        self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self): return self.starts.shape[0]

    def get_norm_stats(self):
        return (self.obs_mean.copy(), self.obs_std.copy(),
                self.act_mean.copy(), self.act_std.copy())

    def __getitem__(self, i: int) -> dict[str, jnp.ndarray]:
        s0 = int(self.starts[i])

        st = self.states[s0:s0+self.S+1]
        st = (st - self.obs_mean) / self.obs_std
        st = st.astype(np.float32, copy=False)

        ac = self.actions[s0:s0+self.S]
        ac = (ac - self.act_mean) / self.act_std
        ac = ac.astype(np.float32, copy=False)

        sp = self.specs[s0:s0+self.S].astype(np.float32, copy=False)
        if self.specs_scale_01: sp = sp / 255.0
        sp = np.transpose(sp, (0, 3, 1, 2))

        return {
            "states": jnp.asarray(st),
            "actions": jnp.asarray(ac),
            "spectrograms": jnp.asarray(sp),
        }

class JAXEpochLoader:
    """
    Finite, epoch-style iterator with attributes for logging.
    Yields time-major, batched JAX arrays:
      states:  [S+1, B, D]
      actions: [S,   B, A]
      specs:   [S,   B, 3, H, W]
    """
    def __init__(self, dataset: NPZSequenceDatasetJAX, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.sequence_length = int(dataset.S)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(dataset), dtype=np.int64)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        B = self.batch_size
        for start in range(0, len(self), 1):
            idx = self.indices[start*B:(start+1)*B]
            samples = [self.dataset[int(i)] for i in idx]
            st = jnp.stack([s["states"] for s in samples], axis=1)
            ac = jnp.stack([s["actions"] for s in samples], axis=1)
            sp = jnp.stack([s["spectrograms"] for s in samples], axis=1)
            yield {"states": st, "actions": ac, "spectrograms": sp}
