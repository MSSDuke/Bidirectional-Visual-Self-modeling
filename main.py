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


# TODO replace temp-GPT main.py later


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
    if len(train_losses) == 0:
        return
        
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


def save_checkpoint(model, optimizer, epoch, train_losses, log_dir):
    """Save model checkpoint"""
    checkpoint_path = log_dir / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pkl"
    
    # Save model state
    with open(checkpoint_path, 'wb') as f:
        import pickle
        state_dict = {
            'model_state': nnx.state(model),
            'optimizer_state': nnx.state(optimizer),
            'epoch': epoch,
            'train_losses': train_losses
        }
        pickle.dump(state_dict, f)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def validate_model(model, dataloader, num_batches=5):
    """Run validation to check model predictions"""
    val_losses = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        states = batch['states']
        actions = batch['actions']
        
        # Compute validation loss (without gradients)
        total_loss = 0.0
        T = states.shape[0] - 1
        
        for t in range(T):
            inputs = jnp.concatenate([states[t], actions[t]], axis=-1)
            pred_next = model(inputs)
            loss = ((pred_next - states[t + 1]) ** 2).mean()
            total_loss += loss
        
        val_losses.append(total_loss / T)
    
    return np.mean(val_losses) if val_losses else 0.0


def train(model, dataloader, optimizer, epochs, log_dir=None):
    """
    Main training loop
    
    Args:
        model: SoundSM model
        dataloader: DataLoader instance
        optimizer: NNX optimizer
        epochs: number of epochs to train
        log_dir: directory for logs
    """
    train_losses = []
    best_loss = float('inf')

    print(f"\nJAX Configuration:")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {dataloader.batch_size}")
    print(f"Sequence Length: {dataloader.sequence_length}")
    print(f"Learning Rate: {optimizer.learning_rate if hasattr(optimizer, 'learning_rate') else 'N/A'}")
    print(f"Log Directory: {log_dir}")
    print(f"{'='*60}\n")
    
    # Initial validation
    initial_val_loss = validate_model(model, dataloader, num_batches=3)
    print(f"Initial validation loss: {initial_val_loss:.6f}\n")
        
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Training step
            loss, metrics = train_step(model, optimizer, batch)
            
            # Convert to Python float for logging
            metrics = {k: float(v) for k, v in metrics.items()}
            epoch_losses.append(metrics)
            
            # Print progress every 10 batches
            if batch_idx % 1000 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {metrics['L']:.6f}")
        
        # Average metrics across batches
        if epoch_losses:
            avg_metrics = {
                'L': np.mean([m['L'] for m in epoch_losses]),
                'L_d': np.mean([m['L_d'] for m in epoch_losses]),
                'L_s': np.mean([m['L_s'] for m in epoch_losses]),
            }
            train_losses.append(avg_metrics)
            
            # Validation
            val_loss = validate_model(model, dataloader, num_batches=3)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs} Complete | Time: {epoch_time:.2f}s")
            print(f"Training Loss: {avg_metrics['L']:.6f}")
            print(f"Dynamics Loss: {avg_metrics['L_d']:.6f}")
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Check if loss is actually changing
            if epoch > 0:
                loss_change = abs(train_losses[-1]['L'] - train_losses[-2]['L'])
                print(f"Loss change from last epoch: {loss_change:.8f}")
                if loss_change < 1e-8:
                    print("WARNING: Loss is not changing! Check gradients and learning rate.")
            
            print(f"{'='*60}\n")
            
            # Plot every epoch
            if log_dir:
                plot_losses(train_losses, log_dir, epoch+1)
            
            # Save checkpoint every 5 epochs or if best loss
            if log_dir and ((epoch + 1) % 5 == 0 or avg_metrics['L'] < best_loss):
                save_checkpoint(model, optimizer, epoch+1, train_losses, log_dir)
                if avg_metrics['L'] < best_loss:
                    best_loss = avg_metrics['L']
                    print(f"New best loss: {best_loss:.6f}")
    
    return train_losses


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Acoustic Self-Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
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
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    
    print(f"\n{'='*60}")
    print(f"Initializing Training")
    print(f"{'='*60}")
    
    # Create dataloader with sequence support
    print("\nLoading dataset...")
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        shuffle=True, 
        use_jax=True
    )
    
    # Get dimensions from a sample batch
    sample_batch = dataloader.get_random_batch(batch_size=2)
    state_size = sample_batch['states'].shape[-1]  # Should be 29
    action_size = sample_batch['actions'].shape[-1]  # Should be 8
    
    print(f"\nData dimensions:")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Sample batch shapes:")
    print(f"  States: {sample_batch['states'].shape} (T+1, batch, state_dim)")
    print(f"  Actions: {sample_batch['actions'].shape} (T, batch, action_dim)")
    
    # Initialize model
    print("\nInitializing model...")
    rngs = nnx.Rngs(args.seed)

    model = SoundSM(
        state_size=state_size,
        action_size=action_size,
        hidden_width=args.hidden_width,
        rngs=rngs
    )
    
    # Initialize optimizer
    print("Initializing optimizer...")
    learning_rate = args.lr
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
    print(f"\nModel initialized with {param_count:,} parameters")
    
    # Quick gradient check
    print("\nPerforming gradient check...")
    test_batch = dataloader.get_random_batch(batch_size=4)
    
    def test_loss(model):
        states = test_batch['states']
        actions = test_batch['actions']
        inputs = jnp.concatenate([states[0], actions[0]], axis=-1)
        pred = model(inputs)
        return ((pred - states[1]) ** 2).mean()
    
    loss_val, grads = nnx.value_and_grad(test_loss)(model)
    grad_norm = jax.tree_util.tree_reduce(
        lambda acc, x: acc + (x ** 2).sum(),
        grads,
        0.0
    )
    print(f"Test loss: {loss_val:.6f}")
    print(f"Gradient norm: {jnp.sqrt(grad_norm):.6f}")
    
    if grad_norm < 1e-10:
        print("WARNING: Gradients are nearly zero! Check model architecture.")
    
    # Train
    train_losses = train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=args.epochs,
        log_dir=log_dir
    )
    
    # Final save
    if log_dir:
        save_checkpoint(model, optimizer, args.epochs, train_losses, log_dir)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Logs saved to: {log_dir}")
    print(f"Final training loss: {train_losses[-1]['L']:.6f}" if train_losses else "No losses recorded")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()