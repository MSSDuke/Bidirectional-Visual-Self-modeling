import numpy as np
import torch
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax
import flax.nnx as nnx

import minari

dataset = minari.load_dataset("mujoco/ant/simple-v0", download=True)
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)


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
            # Observations are a flat array of shape (episode_length, 105)
            # According to Ant-v4 docs:
            # - qpos: indices 0:15 (15 values)
            # - qvel: indices 15:29 (14 values)
            # The remaining indices contain other sensor data we don't need
            observations = episode.observations
            
            # Extract qpos (first 15 values) and qvel (next 14 values)
            qpos = observations[:, 0:15]
            qvel = observations[:, 15:29]
            
            # Actions have shape (episode_length, 8)
            actions = episode.actions
            
            qpos_list.append(qpos)
            qvel_list.append(qvel)
            actions_list.append(actions)
        
        # Concatenate all episodes
        self.qpos = np.concatenate(qpos_list, axis=0)
        self.qvel = np.concatenate(qvel_list, axis=0)
        self.actions = np.concatenate(actions_list, axis=0)
        
        # Create combined observation array (qpos + qvel)
        self.observations = np.concatenate([self.qpos, self.qvel], axis=-1)
        
        self.num_samples = len(self.observations)
        self.indices = np.arange(self.num_samples)
        
        print(f"\nLoaded {self.num_samples} transitions")
        print(f"qpos shape: {self.qpos.shape}")
        print(f"qvel shape: {self.qvel.shape}")
        print(f"Combined observation shape: {self.observations.shape}")
        print(f"Actions shape: {self.actions.shape}")
        
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