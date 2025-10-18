import numpy as np
import torch
import torch.nn as nn
import jax, jax.numpy as jnp, jax.nn as jnn
import flax
from flax import nnx

import minari

dataset = minari.load_dataset("mujoco/ant/simple-v0", download=True)

"""
Temporary data loader, will be replaced when real-world data is available
"""

class DataLoader():
    def __init__(self, dataset, batch_size=32, shuffle=True, use_jax=True):
        """
        DataLoader for MuJoCo Ant dataset from Minari.
        
        Args:
            dataset: Minari dataset object
            batch_size: Number of transitions per batch
            shuffle: Whether to shuffle the data
            use_jax: If True, return JAX arrays; if False, return NumPy arrays
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_jax = use_jax
        
        # Extract all data from episodes
        self._load_data()
        
    def _load_data(self):
        """Extract qpos, qvel observations and actions from all episodes."""
        qpos_list = []
        qvel_list = []
        actions_list = []
        
        # Iterate through all episodes
        for episode in self.dataset.iterate_episodes():
            observations = episode.observations
            actions = episode.actions
            
            # CRITICAL: Episodes have N observations but N-1 actions
            # We need to trim observations to match actions
            episode_length = len(actions)  # Use action length as reference
            
            # Extract qpos and qvel, trimming to match actions
            qpos = observations[:episode_length, 0:15]
            qvel = observations[:episode_length, 15:29]
            
            qpos_list.append(qpos)
            qvel_list.append(qvel)
            actions_list.append(actions)
        
        # Concatenate all episodes
        self.qpos = np.concatenate(qpos_list, axis=0)
        self.qvel = np.concatenate(qvel_list, axis=0)
        self.actions = np.concatenate(actions_list, axis=0)
        
        # Create combined observation array (qpos + qvel)
        self.observations = np.concatenate([self.qpos, self.qvel], axis=-1)
        
        self.obs_mean = self.observations.mean(axis=0, keepdims=True)
        self.obs_std = self.observations.std(axis=0, keepdims=True) + 1e-8
        self.observations = (self.observations - self.obs_mean) / self.obs_std
        
        self.action_mean = self.actions.mean(axis=0, keepdims=True)
        self.action_std = self.actions.std(axis=0, keepdims=True) + 1e-8
        self.actions = (self.actions - self.action_mean) / self.action_std
        
        self.num_samples = len(self.observations)
        
        # Sanity check
        assert self.observations.shape[0] == self.actions.shape[0], \
            f"Mismatch: {self.observations.shape[0]} observations vs {self.actions.shape[0]} actions"
        
        self.indices = np.arange(self.num_samples)
        
        print(f"\nLoaded {self.num_samples} transitions")
        print(f"qpos shape: {self.qpos.shape}")
        print(f"qvel shape: {self.qvel.shape}")
        print(f"Combined observation shape: {self.observations.shape}")
        print(f"Actions shape: {self.actions.shape}")
        print(f"Observation stats - mean: {self.obs_mean[0, :5]}, std: {self.obs_std[0, :5]}")
        print(f"Normalized obs range: [{self.observations.min():.2f}, {self.observations.max():.2f}]")
        print(f"Normalized action range: [{self.actions.min():.2f}, {self.actions.max():.2f}]")
        
    def __len__(self):
        """Return number of batches."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            batch_obs = self.observations[batch_indices]
            batch_actions = self.actions[batch_indices]
            
            if self.use_jax:
                batch_obs = jnp.array(batch_obs)
                batch_actions = jnp.array(batch_actions)
            
            yield batch_obs, batch_actions
    
    def get_batch(self, indices):
        """Get a specific batch by indices."""
        batch_obs = self.observations[indices]
        batch_actions = self.actions[indices]
        
        if self.use_jax:
            batch_obs = jnp.array(batch_obs)
            batch_actions = jnp.array(batch_actions)
        
        return batch_obs, batch_actions


if __name__ == "__main__":
    # Example usage
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, use_jax=True)
    
    # Iterate through batches
    for batch_idx, (observations, actions) in enumerate(dataloader):
        print(f"Batch {batch_idx}: obs shape {observations.shape}, actions shape {actions.shape}")
        if batch_idx >= 2:  # Just show first few batches
            break
    
    # Or get random batch
    random_indices = np.random.choice(len(dataloader.observations), size=64, replace=False)
    obs_batch, action_batch = dataloader.get_batch(random_indices)
    print(f"\nRandom batch: obs shape {obs_batch.shape}, actions shape {action_batch.shape}")