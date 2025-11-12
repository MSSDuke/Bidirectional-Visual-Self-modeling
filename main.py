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

from src.model import AcousticModel, SoundAE, Action2Sound, Sound2Action
from src.data_loader import NPZSequenceDatasetJAX, JAXEpochLoader
from src.losses import train_step

import pickle
from pathlib import Path
import numpy as np

def load_all_shards(path_like):
    path = Path(path_like)

    def _load_npz_list(npz_paths):
        states_list, actions_list, specs_list = [], [], []
        for p in npz_paths:
            with np.load(p, allow_pickle=False) as a:
                states_list.append(a["states"])
                actions_list.append(a["actions"])
                # spectrograms may be empty (shape (0,)), skip those
                if "spectrograms" in a.files:
                    specs_arr = a["spectrograms"]
                    if specs_arr.ndim > 1 and specs_arr.size > 0:
                        specs_list.append(specs_arr)
        states = np.concatenate(states_list, axis=0) if states_list else np.empty((0,0), dtype=np.float32)
        actions = np.concatenate(actions_list, axis=0) if actions_list else np.empty((0,0), dtype=np.float32)
        spectrograms = np.concatenate(specs_list, axis=0) if specs_list else np.empty((0,), dtype=np.uint8)
        return states, actions, spectrograms

    if path.is_file():
        # single-file case: rollout.npz
        with np.load(path, allow_pickle=False) as a:
            states = a["states"]
            actions = a["actions"]
            if "spectrograms" in a.files and a["spectrograms"].ndim > 1 and a["spectrograms"].size > 0:
                spectrograms = a["spectrograms"]
            else:
                spectrograms = np.empty((0,), dtype=np.uint8)
        return states, actions, spectrograms

    # directory case: prefer shards; fall back to single rollout.npz if present
    shard_paths = sorted(path.glob("rollout_shard_*.npz"))
    if shard_paths:
        return _load_npz_list(shard_paths)

    single = path / "rollout.npz"
    if single.exists():
        return load_all_shards(single)

    raise FileNotFoundError(
        f"No dataset found in {path}. "
        "Looked for rollout_shard_*.npz and rollout.npz."
    )


def load_pretrained_autoencoder(checkpoint_path="checkpoints/sound_autoencoder_best.pkl", 
                                latent_dim=128, 
                                input_shape=(128, 128, 3),
                                seed=0):
    """Load pretrained autoencoder from checkpoint and mark parameters as non-trainable"""
    rngs = nnx.Rngs(seed)
    ae = SoundAE(latent_dim, input_shape, rngs=rngs)

    with open(checkpoint_path, "rb") as f:
        saved_state = pickle.load(f)

    nnx.update(ae, saved_state)

    # --- Freeze all parameters recursively ---
    def freeze_tree(tree):
        """Convert all nnx.Param leaves into nnx.Variable (non-trainable)."""
        if isinstance(tree, nnx.Param):
            return nnx.Variable(tree.value)
        elif isinstance(tree, dict):
            return {k: freeze_tree(v) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            ttype = type(tree)
            return ttype(freeze_tree(v) for v in tree)
        return tree

    frozen_state = freeze_tree(nnx.state(ae))
    nnx.update(ae, frozen_state)

    print(f"✓ Loaded pretrained autoencoder from {checkpoint_path}")
    print(f"✓ Autoencoder parameters FROZEN (not trainable)")

    return ae



def a2s_loss(model, batch):
    """
    Compute loss for Action2Sound model
    
    Args:
        model: Action2Sound model
        batch: Dict with 'states', 'actions', 'spectrograms'
    
    Returns:
        loss: MSE between predicted and true spectrograms
    """
    states = batch["states"]  # (T+1, B, state_dim)
    actions = batch["actions"]  # (T, B, action_dim)
    spectrograms = batch["spectrograms"]  # (T, B, C, H, W)
    
    T, B = actions.shape[:2]
    
    # Concatenate state + action for each timestep
    # Use states[:-1] to match action timesteps
    state_action = jnp.concatenate([states[:-1], actions], axis=-1)  # (T, B, state_dim + action_dim)
    
    # Reshape spectrograms: (T, B, C, H, W) -> (T*B, H, W, C)
    C, H, W = spectrograms.shape[2:]
    specs_reshaped = spectrograms.reshape(T * B, C, H, W)
    specs_reshaped = jnp.transpose(specs_reshaped, (0, 2, 3, 1))  # (T*B, H, W, C)
    
    # Reshape state_action for model: (T, B, features) -> (B, T, features)
    state_action = jnp.transpose(state_action, (1, 0, 2))  # (B, T, features)
    
    # Predict spectrograms
    pred_specs = []
    for b in range(B):
        pred = model(state_action[b:b+1])  # (1, H, W, C)
        pred_specs.append(pred)
    
    pred_specs = jnp.concatenate(pred_specs, axis=0)  # (B, H, W, C)
    
    # For now, let's predict the final spectrogram in the sequence
    # Take the last spectrogram from each sequence
    target_specs = specs_reshaped.reshape(T, B, H, W, C)[-1]  # (B, H, W, C)
    
    # MSE loss
    loss = jnp.mean((pred_specs - target_specs) ** 2)
    
    return loss


def evaluate(model, loader, max_batches=None):
    """Evaluate model on dataset"""
    losses = []
    
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        
        loss = a2s_loss(model, batch)
        losses.append(float(loss))
    
    return np.mean(losses) if losses else 0.0


def create_log_dir():
    """Create timestamped log directory under Logs/A2S/"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(f"Logs/A2S/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "plots").mkdir(exist_ok=True)
    (log_dir / "checkpoints").mkdir(exist_ok=True)
    return log_dir


def plot_losses(train_losses, val_losses, log_dir, epoch, best_val_loss, best_epoch):
    """Plot and save loss curves"""
    if len(train_losses) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Linear scale
    axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2, alpha=0.8)
    axes[0].plot(epochs, val_losses, label='Val Loss', linewidth=2, alpha=0.8)
    axes[0].axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val', alpha=0.5)
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch', alpha=0.5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Progress', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Log scale
    axes[1].plot(epochs, train_losses, label='Train Loss', linewidth=2, alpha=0.8)
    axes[1].plot(epochs, val_losses, label='Val Loss', linewidth=2, alpha=0.8)
    axes[1].axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val', alpha=0.5)
    axes[1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch', alpha=0.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss (MSE)', fontsize=12)
    axes[1].set_title('Training Progress (Log Scale)', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Generalization gap
    train_val_gap = np.array(val_losses) - np.array(train_losses)
    axes[2].plot(epochs, train_val_gap, linewidth=2, color='purple')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Val Loss - Train Loss', fontsize=12)
    axes[2].set_title('Generalization Gap', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(log_dir / "plots" / f"losses_epoch_{epoch:04d}.png", dpi=150)
    plt.close()


# def count_parameters(state_dict):
#     """Count parameters from a state dict"""
#     param_count = 0
#     for leaf in jax.tree_util.tree_leaves(state_dict):
#         if hasattr(leaf, 'value'):
#             param_count += leaf.value.size
#         elif hasattr(leaf, 'size'):
#             param_count += leaf.size
#         else:
#             param_count += jnp.array(leaf).size
#     return param_count


# def count_trainable_vs_frozen(model):
#     """Count trainable vs frozen parameters"""
#     state = nnx.state(model)
    
#     trainable_count = 0
#     frozen_count = 0
    
#     # Count Param (trainable) vs Variable (frozen)
#     for path, value in jax.tree_util.tree_leaves_with_path(state):
#         if isinstance(value, nnx.Param):
#             trainable_count += value.value.size
#         elif isinstance(value, nnx.Variable):
#             frozen_count += value.value.size
    
#     return trainable_count, frozen_count


# def train_action2sound(args):
#     """Main training function for Action2Sound model"""
    
#     # Set random seeds
#     np.random.seed(args.seed)
#     jax.random.PRNGKey(args.seed)
    
#     # Create log directory
#     log_dir = create_log_dir()
    
#     # Save config
#     config = vars(args)
#     with open(log_dir / "config.txt", 'w') as f:
#         for k, v in config.items():
#             f.write(f"{k}: {v}\n")
    
#     print(f"\n{'='*60}")
#     print(f"Training Action2Sound Model")
#     print(f"{'='*60}")
#     print(f"Log directory: {log_dir}")
    
#     # Load dataset
#     npz_path = args.data_path
#     dataset = NPZSequenceDatasetJAX(
#         npz_path,
#         steps_per_episode=512,
#         sequence_length=args.seq_len,
#         normalize_sa=True,
#         specs_scale_01=True,
#         specs_downsample_hw=(128, 128)
#     )
    
#     # Create dataloaders (70/20/10 split)
#     train_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
#     val_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 1)
#     test_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 2)
    
#     # Get data dimensions
#     sample_batch = next(iter(train_loader))
#     state_dim = sample_batch['states'].shape[-1]
#     action_dim = sample_batch['actions'].shape[-1]
#     input_dim = state_dim + action_dim
    
#     print(f"\nData dimensions:")
#     print(f"  State dim: {state_dim}")
#     print(f"  Action dim: {action_dim}")
#     print(f"  Input dim (state+action): {input_dim}")
#     print(f"  Sequence length: {args.seq_len}")
    
#     # Load pretrained autoencoder (FROZEN)
#     ae = load_pretrained_autoencoder(
#         checkpoint_path=args.ae_checkpoint,
#         latent_dim=args.latent_dim,
#         input_shape=(128, 128, 3),
#         seed=args.seed
#     )
    
#     # Create Action2Sound model
#     rngs = nnx.Rngs(args.seed)
#     dummy_input = jnp.ones((1, args.seq_len, input_dim))
#     a2s_model = Action2Sound(ae, dummy_input.shape, args.latent_dim, rngs=rngs)
    
#     print(f"\n✓ Created Action2Sound model")    
    
#     # CRITICAL: Create optimizer that ONLY optimizes Action2Sound parameters
#     # This ensures autoencoder stays frozen
#     optimizer = nnx.Optimizer(a2s_model, optax.adam(args.lr), wrt=nnx.Param)
    
#     print(f"\n✓ Optimizer created (autoencoder parameters excluded)")
    
#     # Verify autoencoder is frozen by checking gradients don't flow to it
#     # (gradients will be None or zero for frozen params)
    
#     # Training loop
#     train_losses = []
#     val_losses = []
#     best_val_loss = float('inf')
#     best_epoch = 0
    
#     print(f"\nStarting training for {args.epochs} epochs...")
#     print(f"{'='*60}\n")
    
#     for epoch in range(args.epochs):
#         epoch_start = time.time()
#         epoch_train_losses = []
        
#         # Training
#         for batch_idx, batch in enumerate(train_loader):
#             loss, grads = nnx.value_and_grad(a2s_loss)(a2s_model, batch)
            
#             # Update only trainable parameters (autoencoder is excluded automatically)
#             optimizer.update(grads=grads, model=a2s_model)
#             epoch_train_losses.append(float(loss))
        
#         avg_train_loss = np.mean(epoch_train_losses)
#         train_losses.append(avg_train_loss)
        
#         # Validation
#         avg_val_loss = evaluate(a2s_model, val_loader, max_batches=10)
#         val_losses.append(avg_val_loss)
        
#         # Track best model
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_epoch = epoch + 1
            
#             # Save best model (only Action2Sound parameters)
#             with open(log_dir / "checkpoints" / "action2sound_best.pkl", "wb") as f:
#                 pickle.dump(nnx.state(a2s_model), f)
        
#         epoch_time = time.time() - epoch_start
        
#         print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | "
#               f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
#               f"Best: {best_val_loss:.6f} (ep {best_epoch})")
        
#         # Plot every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             plot_losses(train_losses, val_losses, log_dir, epoch + 1, best_val_loss, best_epoch)
            
#             # Save periodic checkpoint
#             with open(log_dir / "checkpoints" / f"action2sound_epoch_{epoch+1}.pkl", "wb") as f:
#                 pickle.dump(nnx.state(a2s_model), f)
    
#     # Load best model and evaluate on test set
#     print(f"\nLoading best model from epoch {best_epoch}...")
#     with open(log_dir / "checkpoints" / "action2sound_best.pkl", "rb") as f:
#         nnx.update(a2s_model, pickle.load(f))
    
#     test_loss = evaluate(a2s_model, test_loader, max_batches=None)
    
#     # Save final model and results
#     with open(log_dir / "checkpoints" / "action2sound_final.pkl", "wb") as f:
#         pickle.dump(nnx.state(a2s_model), f)
    
#     # Plot final results
#     plot_losses(train_losses, val_losses, log_dir, args.epochs, best_val_loss, best_epoch)
    
#     # Save training history
#     history = {
#         'train_losses': train_losses,
#         'val_losses': val_losses,
#         'test_loss': test_loss,
#         'best_val_loss': best_val_loss,
#         'best_epoch': best_epoch,
#         'args': config
#     }
#     with open(log_dir / "training_history.pkl", "wb") as f:
#         pickle.dump(history, f)
    
#     print(f"\n{'='*60}")
#     print(f"Training Complete!")
#     print(f"{'='*60}")
#     print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
#     print(f"Final train loss: {train_losses[-1]:.6f}")
#     print(f"Final val loss: {val_losses[-1]:.6f}")
#     print(f"Test loss: {test_loss:.6f}")
#     print(f"\nSaved to: {log_dir}")
#     print(f"{'='*60}\n")


# def main():
#     parser = argparse.ArgumentParser(description='Train Action2Sound Model')
#     parser.add_argument('--data_path', type=str, 
#                        default="data/Ant-v5/2025-10-25_13-19-10_seed42/rollout.npz",
#                        help='Path to dataset')
#     parser.add_argument('--ae_checkpoint', type=str,
#                        default="checkpoints/sound_autoencoder_best.pkl",
#                        help='Path to pretrained autoencoder')
#     parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
#     parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#     parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
#     parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
#     args = parser.parse_args()
#     train_action2sound(args)


# if __name__ == "__main__":
#     main()

def s2a_loss(model, batch):
    """
    Compute loss for Sound2Action model
    
    Args:
        model: Sound2Action model
        batch: Dict with 'states', 'actions', 'spectrograms'
    
    Returns:
        loss: MSE between predicted and true actions
    """
    actions = batch["actions"]  # (T, B, action_dim)
    spectrograms = batch["spectrograms"]  # (T, B, C, H, W)
    
    T, B = actions.shape[:2]
    
    # Reshape spectrograms: (T, B, C, H, W) -> (T*B, H, W, C)
    C, H, W = spectrograms.shape[2:]
    specs_reshaped = spectrograms.reshape(T * B, C, H, W)
    specs_reshaped = jnp.transpose(specs_reshaped, (0, 2, 3, 1))  # (T*B, H, W, C)
    
    # Reshape actions: (T, B, action_dim) -> (T*B, action_dim)  
    actions_flat = actions.reshape(T * B, -1)
    
    # Process each spectrogram through the model
    total_loss = 0.0
    for i in range(specs_reshaped.shape[0]):
        spec = specs_reshaped[i]  # (H, W, C)
        target_action = actions_flat[i]  # (action_dim,)
        
        pred_action_seq = model(spec)  # (T_model, action_dim)
        
        # Take first timestep prediction (or mean, or specific timestep)
        pred_action = pred_action_seq[0]  # (action_dim,)
        
        # MSE loss for this sample
        total_loss += jnp.mean((pred_action - target_action) ** 2)
    
    # Average over batch
    loss = total_loss / specs_reshaped.shape[0]
    
    return loss


def create_s2a_log_dir():
    """Create timestamped log directory under Logs/S2A/"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(f"Logs/S2A/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "plots").mkdir(exist_ok=True)
    (log_dir / "checkpoints").mkdir(exist_ok=True)
    return log_dir


def train_sound2action(args):
    """Main training function for Sound2Action model"""
    
    # Set random seeds
    np.random.seed(args.seed)
    jax.random.PRNGKey(args.seed)
    
    # Create log directory
    log_dir = create_s2a_log_dir()
    
    # Save config
    config = vars(args)
    with open(log_dir / "config.txt", 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    
    print(f"\n{'='*60}")
    print(f"Training Sound2Action Model")
    print(f"{'='*60}")
    print(f"Log directory: {log_dir}")
    
    # Load dataset
    npz_path = args.data_path
    dataset = NPZSequenceDatasetJAX(
        npz_path,
        steps_per_episode=512,
        sequence_length=args.seq_len,
        normalize_sa=True,
        specs_scale_01=True,
        specs_downsample_hw=(128, 128)
    )
    
    # Create dataloaders (70/20/10 split)
    train_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 1)
    test_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 2)
    
    # Get data dimensions
    sample_batch = next(iter(train_loader))
    action_dim = sample_batch['actions'].shape[-1]
    
    print(f"\nData dimensions:")
    print(f"  Action dim: {action_dim}")
    print(f"  Sequence length: {args.seq_len}")
    
    # Load pretrained autoencoder (FROZEN)
    ae = load_pretrained_autoencoder(
        checkpoint_path=args.ae_checkpoint,
        latent_dim=args.latent_dim,
        input_shape=(128, 128, 3),
        seed=args.seed
    )
    
    # Create Sound2Action model
    rngs = nnx.Rngs(args.seed)
    s2a_model = Sound2Action(
        sound_autoencoder=ae,
        latent_dim=args.latent_dim,
        T=args.seq_len,
        action_dim=action_dim,
        rngs=rngs
    )
    
    print(f"\n✓ Created Sound2Action model")
    
  
    # Create optimizer that ONLY optimizes Sound2Action parameters
    optimizer = nnx.Optimizer(s2a_model, optax.adam(args.lr), wrt=nnx.Param)
    
    print(f"\n✓ Optimizer created (autoencoder parameters excluded)")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_train_losses = []
        
        # Training
        for batch_idx, batch in enumerate(train_loader):
            loss, grads = nnx.value_and_grad(s2a_loss)(s2a_model, batch)
            
            # Update only trainable parameters (autoencoder is excluded automatically)
            optimizer.update(grads=grads, model=s2a_model)
            epoch_train_losses.append(float(loss))
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        avg_val_loss = evaluate(s2a_model, val_loader, max_batches=10)
        val_losses.append(avg_val_loss)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            
            # Save best model (only Sound2Action parameters)
            with open(log_dir / "checkpoints" / "sound2action_best.pkl", "wb") as f:
                pickle.dump(nnx.state(s2a_model), f)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | "
              f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
              f"Best: {best_val_loss:.6f} (ep {best_epoch})")
        
        # Plot every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_losses(train_losses, val_losses, log_dir, epoch + 1, best_val_loss, best_epoch)
            
            # Save periodic checkpoint
            with open(log_dir / "checkpoints" / f"sound2action_epoch_{epoch+1}.pkl", "wb") as f:
                pickle.dump(nnx.state(s2a_model), f)
    
    # Load best model and evaluate on test set
    print(f"\nLoading best model from epoch {best_epoch}...")
    with open(log_dir / "checkpoints" / "sound2action_best.pkl", "rb") as f:
        nnx.update(s2a_model, pickle.load(f))
    
    test_loss = evaluate(s2a_model, test_loader, max_batches=None)
    
    # Save final model and results
    with open(log_dir / "checkpoints" / "sound2action_final.pkl", "wb") as f:
        pickle.dump(nnx.state(s2a_model), f)
    
    # Plot final results
    plot_losses(train_losses, val_losses, log_dir, args.epochs, best_val_loss, best_epoch)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'args': config
    }
    with open(log_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"\nSaved to: {log_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Action2Sound or Sound2Action Model')
    parser.add_argument('--mode', type=str, choices=['a2s', 's2a'], default='a2s',
                       help='Training mode: a2s (Action2Sound) or s2a (Sound2Action)')
    parser.add_argument('--data_path', type=str, 
                       default="data/Ant-v5/2025-10-25_13-19-10_seed42/rollout.npz",
                       help='Path to dataset')
    parser.add_argument('--ae_checkpoint', type=str,
                       default="checkpoints/sound_autoencoder_best.pkl",
                       help='Path to pretrained autoencoder')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_sound2action(args)


if __name__ == "__main__":
    main()