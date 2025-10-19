import numpy as np
import torch
import torch.nn as nn
import jax, jax.numpy as jnp, jax.nn as jnn
import flax
from flax import nnx

import minari

dataset = minari.load_dataset("mujoco/ant/simple-v0", download=True)

class DataLoader():
    def __init__(self, dataset, batch_size=32, sequence_length=10, shuffle=True, use_jax=True):
        """
        DataLoader for MuJoCo Ant dataset from Minari.
        
        Args:
            dataset: Minari dataset object
            batch_size: Number of sequences per batch
            sequence_length: Length of each sequence
            shuffle: Whether to shuffle the data
            use_jax: If True, return JAX arrays; if False, return NumPy arrays
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.use_jax = use_jax
        
        # Extract all data from episodes
        self._load_data()
        
    def _load_data(self):
        """Extract qpos, qvel observations and actions from all episodes, maintaining episode boundaries."""
        self.episodes = []
        
        # Process each episode separately
        for episode in self.dataset.iterate_episodes():
            observations = episode.observations
            actions = episode.actions
            
            # Episodes have N observations but N-1 actions
            episode_length = len(actions)
            
            # Extract qpos and qvel
            qpos = observations[:episode_length + 1, 0:15]  # Keep N observations for N-1 actions
            qvel = observations[:episode_length + 1, 15:29]
            
            # Combine observations
            obs = np.concatenate([qpos, qvel], axis=-1)
            
            # Store episode data
            self.episodes.append({
                'observations': obs,  # Shape: [episode_length + 1, obs_dim]
                'actions': actions     # Shape: [episode_length, action_dim]
            })
        
        # Compute normalization statistics across all episodes
        all_obs = np.concatenate([ep['observations'][:-1] for ep in self.episodes], axis=0)
        all_actions = np.concatenate([ep['actions'] for ep in self.episodes], axis=0)
        
        self.obs_mean = all_obs.mean(axis=0, keepdims=True)
        self.obs_std = all_obs.std(axis=0, keepdims=True) + 1e-8
        
        self.action_mean = all_actions.mean(axis=0, keepdims=True)
        self.action_std = all_actions.std(axis=0, keepdims=True) + 1e-8
        
        # Normalize episode data
        for ep in self.episodes:
            ep['observations'] = (ep['observations'] - self.obs_mean) / self.obs_std
            ep['actions'] = (ep['actions'] - self.action_mean) / self.action_std
        
        # Create valid sequences from episodes
        self.sequences = []
        for ep in self.episodes:
            ep_len = len(ep['actions'])
            # Extract all possible sequences of length sequence_length from this episode
            for start_idx in range(ep_len - self.sequence_length + 1):
                end_idx = start_idx + self.sequence_length
                seq = {
                    'states': ep['observations'][start_idx:end_idx + 1],  # T+1 states
                    'actions': ep['actions'][start_idx:end_idx]            # T actions
                }
                self.sequences.append(seq)
        
        self.num_sequences = len(self.sequences)
        self.indices = np.arange(self.num_sequences)
        
        print(f"\nLoaded {len(self.episodes)} episodes")
        print(f"Created {self.num_sequences} sequences of length {self.sequence_length}")
        print(f"Observation shape: [sequence_length + 1, {self.sequences[0]['states'].shape[1]}]")
        print(f"Action shape: [sequence_length, {self.sequences[0]['actions'].shape[1]}]")
        
    def __len__(self):
        """Return number of batches."""
        return (self.num_sequences + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches of sequences."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.num_sequences, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_sequences)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Collect sequences for this batch
            batch_states = []
            batch_actions = []
            
            for idx in batch_indices:
                seq = self.sequences[idx]
                batch_states.append(seq['states'])
                batch_actions.append(seq['actions'])
            
            # Stack into batch arrays
            batch_states = np.stack(batch_states, axis=1)  # [T+1, batch, state_dim]
            batch_actions = np.stack(batch_actions, axis=1)  # [T, batch, action_dim]
            
            if self.use_jax:
                batch_states = jnp.array(batch_states)
                batch_actions = jnp.array(batch_actions)
            
            yield {
                'states': batch_states,
                'actions': batch_actions
            }
    
    def get_random_batch(self, batch_size=None):
        """Get a random batch of sequences."""
        if batch_size is None:
            batch_size = self.batch_size
            
        indices = np.random.choice(self.num_sequences, size=min(batch_size, self.num_sequences), replace=False)
        
        batch_states = []
        batch_actions = []
        
        for idx in indices:
            seq = self.sequences[idx]
            batch_states.append(seq['states'])
            batch_actions.append(seq['actions'])
        
        batch_states = np.stack(batch_states, axis=1)  # [T+1, batch, state_dim]
        batch_actions = np.stack(batch_actions, axis=1)  # [T, batch, action_dim]
        
        if self.use_jax:
            batch_states = jnp.array(batch_states)
            batch_actions = jnp.array(batch_actions)
        
        return {
            'states': batch_states,
            'actions': batch_actions
        }


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