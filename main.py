import os
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import matplotlib.pyplot as plt

from src.model import SoundSM
from src.data_loader import DataLoader, dataset
from src.losses import train_step

# TEMPORARY GPT MAIN CODE, JUST TESTING IF DYNAMICS MODEL CAN WORK





def create_log_dir():
    """Create timestamped log directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(f"Logs/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "plots").mkdir(exist_ok=True)
    (log_dir / "checkpoints").mkdir(exist_ok=True)
    return log_dir


def plot_losses(train_losses, log_dir, epoch):
    """Plot and save loss curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training Progress - Epoch {epoch}')
    
    # Extract loss components
    epochs = np.arange(len(train_losses))
    total_losses = [loss['L'] for loss in train_losses]
    dynamics_losses = [loss['L_d'] for loss in train_losses]
    spectrogram_losses = [loss['L_s'] for loss in train_losses]
    
    # Total loss
    axes[0, 0].plot(epochs, total_losses, 'b-', alpha=0.7, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Dynamics loss
    axes[0, 1].plot(epochs, dynamics_losses, 'r-', alpha=0.7, label='Dynamics Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Dynamics Loss (L_d)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Spectrogram loss (will be 0 for now)
    axes[1, 0].plot(epochs, spectrogram_losses, 'g-', alpha=0.7, label='Spectrogram Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Spectrogram Loss (L_s)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Loss components stacked
    axes[1, 1].plot(epochs, total_losses, 'b-', alpha=0.7, label='Total')
    axes[1, 1].plot(epochs, dynamics_losses, 'r-', alpha=0.7, label='Dynamics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('All Loss Components')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(log_dir / "plots" / f"losses_epoch_{epoch:04d}.png", dpi=150)
    plt.close()
    
    # Also save a "latest" version that gets overwritten
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, 'b-', alpha=0.7, linewidth=2, label='Total Loss')
    plt.plot(epochs, dynamics_losses, 'r-', alpha=0.7, linewidth=2, label='Dynamics Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(log_dir / "plots" / "latest_losses.png", dpi=150)
    plt.close()


def save_checkpoint(model, optimizer, epoch, train_losses, log_dir):
    """Save model checkpoint"""
    checkpoint_path = log_dir / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pkl"
    
    # Save model state
    with open(checkpoint_path, 'wb') as f:
        # NNX models can be saved with their state
        import pickle
        state_dict = {
            'model_state': nnx.state(model),
            'optimizer_state': nnx.state(optimizer),
            'epoch': epoch,
            'train_losses': train_losses
        }
        pickle.dump(state_dict, f)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def prepare_batch(observations, actions, sequence_length=10):
    """
    Convert single-step transitions to sequences for training.
    
    Args:
        observations: [batch, obs_dim] - current observations
        actions: [batch, action_dim] - actions taken
        sequence_length: how many steps to use
        
    Returns:
        batch dict with states and actions in proper format
    """
    # For now, we'll create artificial sequences by treating consecutive
    # samples as if they're from the same trajectory
    # This is not ideal but works for initial testing
    
    batch_size = observations.shape[0] - sequence_length
    
    # Create sequences
    state_sequences = []
    action_sequences = []
    
    for i in range(batch_size):
        state_seq = observations[i:i+sequence_length+1]  # T+1 states
        action_seq = actions[i:i+sequence_length]         # T actions
        state_sequences.append(state_seq)
        action_sequences.append(action_seq)
    
    states = jnp.stack(state_sequences)  # [batch, T+1, state_dim]
    actions = jnp.stack(action_sequences)  # [batch, T, action_dim]
    
    # Transpose to [T+1, batch, state_dim] and [T, batch, action_dim]
    states = states.transpose(1, 0, 2)
    actions = actions.transpose(1, 0, 2)
    
    return {
        'states': states,
        'actions': actions
    }


def train(model, dataloader, optimizer, epochs, sequence_length=10, log_dir=None):
    """
    Main training loop
    
    Args:
        model: SoundSM model
        dataloader: DataLoader instance
        optimizer: NNX optimizer
        epochs: number of epochs to train
        sequence_length: length of sequences for rollout
        log_dir: directory for logs
    """
    train_losses = []
    best_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Sequence Length: {sequence_length}")
    print(f"Log Directory: {log_dir}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        for batch_idx, (observations, actions) in enumerate(dataloader):
            # Prepare batch with sequences
            try:
                batch = prepare_batch(observations, actions, sequence_length)
            except Exception as e:
                print(f"Warning: Skipping batch {batch_idx} due to error: {e}")
                continue
            
            # Training step
            loss, metrics = train_step(model, optimizer, batch)
            epoch_losses.append(metrics)
        
        # Average metrics across batches
        avg_metrics = {
            'L': np.mean([m['L'] for m in epoch_losses]),
            'L_d': np.mean([m['L_d'] for m in epoch_losses]),
            'L_s': np.mean([m['L_s'] for m in epoch_losses]),
        }
        train_losses.append(avg_metrics)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs} Complete | Time: {epoch_time:.2f}s")
        print(f"Average Loss: {avg_metrics['L']:.6f}")
        print(f"Dynamics Loss: {avg_metrics['L_d']:.6f}")
        print(f"Spectrogram Loss: {avg_metrics['L_s']:.6f}")
        print(f"{'='*60}\n")
        
        # Plot every epoch
        plot_losses(train_losses, log_dir, epoch+1)
        
        # Save checkpoint every 5 epochs or if best loss
        if (epoch + 1) % 5 == 0 or avg_metrics['L'] < best_loss:
            save_checkpoint(model, optimizer, epoch+1, train_losses, log_dir)
            if avg_metrics['L'] < best_loss:
                best_loss = avg_metrics['L']
                print(f"New best loss: {best_loss:.6f}")
    
    return train_losses


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Acoustic Self-Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for rollout')
    parser.add_argument('--hidden_width', type=int, default=256, help='Hidden layer width')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    # Create log directory
    log_dir = create_log_dir()
    
    # Save config
    config = vars(args)
    with open(log_dir / "config.txt", 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n{'='*60}")
    print(f"Initializing Training")
    print(f"{'='*60}")
    
    # Create dataloader
    print("\nLoading dataset...")
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        use_jax=True
    )
    
    # Get dimensions from data
    sample_obs, sample_actions = next(iter(dataloader))
    state_size = sample_obs.shape[1]  # Should be 29 (15 qpos + 14 qvel)
    action_size = sample_actions.shape[1]  # Should be 8
    
    print(f"\nData dimensions:")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Initialize model
    print("\nInitializing model...")
    rngs = nnx.Rngs(args.seed)  # Create RNG from seed

    model = SoundSM(
        state_size=state_size,
        action_size=action_size,
        hidden_width=args.hidden_width,
        rngs=rngs  # Pass RNG to model
    )
    
    # Initialize optimizer
    print("Initializing optimizer...")
    learning_rate = args.lr
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    
    print(f"\nModel initialized with {sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))} parameters")
    
    # Train
    train_losses = train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=args.epochs,
        sequence_length=args.seq_len,
        log_dir=log_dir
    )
    
    # Final save
    save_checkpoint(model, optimizer, args.epochs, train_losses, log_dir)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()