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

from src.model import SoundAE, Action2Sound, Sound2Action
from src.data_loader import NPZSequenceDatasetJAX, JAXEpochLoader
from src.data_collector import run_data_collection
from src.losses import train_step, a2s_loss, s2a_loss

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
                if "spectrograms" in a.files:
                    specs_arr = a["spectrograms"]
                    if specs_arr.ndim > 1 and specs_arr.size > 0:
                        specs_list.append(specs_arr)
        states = np.concatenate(states_list, axis=0) if states_list else np.empty((0,0), dtype=np.float32)
        actions = np.concatenate(actions_list, axis=0) if actions_list else np.empty((0,0), dtype=np.float32)
        spectrograms = np.concatenate(specs_list, axis=0) if specs_list else np.empty((0,), dtype=np.uint8)
        return states, actions, spectrograms

    if path.is_file():
        with np.load(path, allow_pickle=False) as a:
            states = a["states"]
            actions = a["actions"]
            if "spectrograms" in a.files and a["spectrograms"].ndim > 1 and a["spectrograms"].size > 0:
                spectrograms = a["spectrograms"]
            else:
                spectrograms = np.empty((0,), dtype=np.uint8)
        return states, actions, spectrograms

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

    def freeze_tree(tree):
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
    return ae


def evaluate(model, loader, max_batches=None):
    """Evaluate model on dataset"""
    losses = []
    
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        loss = s2a_loss(model, batch)
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

# def train_sound_ae(args):
#     np.random.seed(args.seed)
#     jax.random.PRNGKey(args.seed)
#     log_dir = create_log_dir()
#     config = vars(args)
#     with open(log_dir / "config.txt", 'w') as f:
#         for k, v in config.items():
#             f.write(f"{k}: {v}\n")
    
#     print(f"\n{'='*60}")
#     print(f"Training Sound Autoencoder")
#     print(f"{'='*60}")
#     print(f"Log directory: {log_dir}")

#     npz_path = args.path_name
#     dataset = NPZSequenceDatasetJAX(
#         path_like=npz_path,
#         steps_per_episode=512,
#         sequence_length=10,
#         normalize_sa=True,
#         specs_scale_01=True,
#         specs_downsample_hw=(128,128)
#     )
    
#     train_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
#     val_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 1)
#     test_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 2)
    
#     sample_batch = next(iter(train_loader))
#     state_dim = sample_batch['states'].shape[-1]
#     action_dim = sample_batch['actions'].shape[-1]
#     input_dim = state_dim + action_dim
    
#     print(f"\nData dimensions:")
#     print(f"  State dim: {state_dim}")
#     print(f"  Action dim: {action_dim}")
#     print(f"  Input dim (state+action): {input_dim}")
#     print(f"  Sequence length: {args.seq_len}")

#     rngs = nnx.Rngs(args.seed)
#     model = SoundAE(
#         latent_dim=args.latent_dim,
#         input_shape=(128, 128, 3),
#         rngs=rngs
#     )
#     optimizer = nnx.Optimizer(model, optax.adam(args.lr), wrt=nnx.Param)

#     train_losses = []
#     val_losses = []
#     best_val_loss = float('inf')
#     best_epoch = 0

#     from src.losses import sound_error

#     for epoch in range(args.epochs):
#         epoch_start = time.time()
#         epoch_train_losses = []
        
#         for batch_idx, batch in enumerate(train_loader):
#             spectrograms = batch["spectrograms"]
#             T, B, C, H, W = spectrograms.shape
            
#             specs_batch = spectrograms.reshape(T * B, C, H, W)
#             specs_batch = jnp.transpose(specs_batch, (0, 2, 3, 1))
            
#             def ae_loss_fn(model):
#                 reconstructed, latent = model(specs_batch)
#                 loss = jnp.mean((reconstructed - specs_batch) ** 2)
#                 return loss
            
#             loss, grads = nnx.value_and_grad(ae_loss_fn)(model)
#             optimizer.update(grads=grads, model=model)
            
#             epoch_train_losses.append(float(loss))
        
#         avg_train_loss = np.mean(epoch_train_losses)
#         train_losses.append(avg_train_loss)
        
#         val_losses_batch = []
#         for val_batch in val_loader:
#             spectrograms = val_batch["spectrograms"]
#             T, B, C, H, W = spectrograms.shape
#             specs_batch = spectrograms.reshape(T * B, C, H, W)
#             specs_batch = jnp.transpose(specs_batch, (0, 2, 3, 1))
            
#             reconstructed, latent = model(specs_batch)
#             val_loss = jnp.mean((reconstructed - specs_batch) ** 2)
#             val_losses_batch.append(float(val_loss))
        
#         avg_val_loss = np.mean(val_losses_batch)
#         val_losses.append(avg_val_loss)
        
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_epoch = epoch + 1      
#             with open(log_dir / "checkpoints" / "sound_autoencoder_best.pkl", "wb") as f:
#                 pickle.dump(nnx.state(model), f)
        
#         epoch_time = time.time() - epoch_start
        
#         print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | "
#               f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
#               f"Best: {best_val_loss:.6f} (ep {best_epoch})")
        
#         if (epoch + 1) % 10 == 0:
#             plot_losses(train_losses, val_losses, log_dir, epoch + 1, best_val_loss, best_epoch)   
#             with open(log_dir / "checkpoints" / f"sound_ae_epoch_{epoch+1}.pkl", "wb") as f:
#                 pickle.dump(nnx.state(model), f)
    
#     print(f"\nLoading best model from epoch {best_epoch}...")
#     with open(log_dir / "checkpoints" / "sound_autoencoder_best.pkl", "rb") as f:
#         nnx.update(model, pickle.load(f))
    
#     test_losses_batch = []
#     for test_batch in test_loader:
#         spectrograms = test_batch["spectrograms"]
#         T, B, C, H, W = spectrograms.shape
#         specs_batch = spectrograms.reshape(T * B, C, H, W)
#         specs_batch = jnp.transpose(specs_batch, (0, 2, 3, 1))
        
#         reconstructed, latent = model(specs_batch)
#         test_loss = jnp.mean((reconstructed - specs_batch) ** 2)
#         test_losses_batch.append(float(test_loss))
    
#     avg_test_loss = np.mean(test_losses_batch)
    
#     with open(log_dir / "checkpoints" / "sound_autoencoder_final.pkl", "wb") as f:
#         pickle.dump(nnx.state(model), f)
    
#     plot_losses(train_losses, val_losses, log_dir, args.epochs, best_val_loss, best_epoch)
    
#     history = {
#         'train_losses': train_losses,
#         'val_losses': val_losses,
#         'test_loss': avg_test_loss,
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
#     print(f"Test loss: {avg_test_loss:.6f}")
#     print(f"\nSaved to: {log_dir}")
#     print(f"{'='*60}\n")


def train_action2sound(args):
    """Main training function for Action2Sound model"""
    
    np.random.seed(args.seed)
    jax.random.PRNGKey(args.seed)
    
    log_dir = create_log_dir()
    
    config = vars(args)
    with open(log_dir / "config.txt", 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    
    print(f"\n{'='*60}")
    print(f"Training Action2Sound Model")
    print(f"{'='*60}")
    print(f"Log directory: {log_dir}")
    
    npz_path = args.data_path
    dataset = NPZSequenceDatasetJAX(
        npz_path,
        steps_per_episode=512,
        sequence_length=args.seq_len,
        normalize_sa=True,
        specs_scale_01=True,
        specs_downsample_hw=(128, 128)
    )
    
    train_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 1)
    test_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 2)
    
    sample_batch = next(iter(train_loader))
    state_dim = sample_batch['states'].shape[-1]
    action_dim = sample_batch['actions'].shape[-1]
    
    print(f"\nData dimensions:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Sequence length: {args.seq_len}")
    
    ae = load_pretrained_autoencoder(
        checkpoint_path=args.ae_checkpoint,
        latent_dim=args.latent_dim,
        input_shape=(128, 128, 3),
        seed=args.seed
    )
    
    rngs = nnx.Rngs(args.seed)
    a2s_model = Action2Sound(
        sound_autoencoder=ae,
        seq_len=args.seq_len,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        rngs=rngs
    )
    
    print(f"\n✓ Created Action2Sound model")    
    
    optimizer = nnx.Optimizer(a2s_model, optax.adam(args.lr), wrt=nnx.Param)
    
    print(f"\n✓ Optimizer created (autoencoder parameters excluded)")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            loss, grads = nnx.value_and_grad(a2s_loss)(a2s_model, batch)
            
            optimizer.update(grads=grads, model=a2s_model)
            epoch_train_losses.append(float(loss))
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        avg_val_loss = evaluate(a2s_model, val_loader, max_batches=10)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            
            with open(log_dir / "checkpoints" / "action2sound_best.pkl", "wb") as f:
                pickle.dump(nnx.state(a2s_model), f)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | "
              f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
              f"Best: {best_val_loss:.6f} (ep {best_epoch})")
        
        if (epoch + 1) % 10 == 0:
            plot_losses(train_losses, val_losses, log_dir, epoch + 1, best_val_loss, best_epoch)
            
            with open(log_dir / "checkpoints" / f"action2sound_epoch_{epoch+1}.pkl", "wb") as f:
                pickle.dump(nnx.state(a2s_model), f)
    
    print(f"\nLoading best model from epoch {best_epoch}...")
    with open(log_dir / "checkpoints" / "action2sound_best.pkl", "rb") as f:
        nnx.update(a2s_model, pickle.load(f))
    
    test_loss = evaluate(a2s_model, test_loader, max_batches=None)
    
    with open(log_dir / "checkpoints" / "action2sound_final.pkl", "wb") as f:
        pickle.dump(nnx.state(a2s_model), f)
    
    plot_losses(train_losses, val_losses, log_dir, args.epochs, best_val_loss, best_epoch)
    
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


# def main():
#     parser = argparse.ArgumentParser(description='Train Action2Sound Model')
#     parser.add_argument('--data_path', type=str, 
#                        default="data/Ant-v5/2025-11-14_13-42-20_seed13/rollout.npz",
#                        help='Path to dataset')
#     parser.add_argument('--ae_checkpoint', type=str,
#                        default="Logs/run_2025-11-14_18-13-00/checkpoints/sound_autoencoder_best.pkl",
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
    
    # Create dataloaders
    train_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 1)
    test_loader = JAXEpochLoader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed + 2)
    
    # Get data dimensions
    sample_batch = next(iter(train_loader))
    action_dim = sample_batch['actions'].shape[-1]
    
    print(f"\nData dimensions:")
    print(f"  Action dim: {action_dim}")
    print(f"  Sequence length: {args.seq_len}")
    
    # Load pretrained, frozen autoencoder
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

    optimizer = nnx.Optimizer(s2a_model, optax.adam(args.lr), wrt=nnx.Param)
    
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
            
            # Save best model
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

# def main():
#     parser = argparse.ArgumentParser(description='Train Models')
#     parser.add_argument('--data_path', type=str, 
#                        default="data/Ant-v5/2025-11-14_13-42-20_seed13/rollout.npz",
#                        help='Path to dataset')
#     parser.add_argument('--ae_checkpoint', type=str,
#                        default="Logs/run_2025-11-14_18-13-00/checkpoints/sound_autoencoder_best.pkl",
#                        help='Path to pretrained autoencoder')
#     parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
#     parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#     parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
#     parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
#     args = parser.parse_args()
    
#     train_sound2action(args)


# if __name__ == "__main__":
#     main()

def test_action2sound_prediction(args):
    """Test Action2Sound model by predicting final image from initial image + actions"""
    
    print(f"\n{'='*60}")
    print(f"Testing Action2Vision Model")
    print(f"{'='*60}")
    
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
    
    # Get a random sample
    random_idx = np.random.randint(0, len(dataset))
    sample = dataset[random_idx]
    
    print(f"\nSample index: {random_idx}")
    print(f"States shape: {sample['states'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Spectrograms shape: {sample['spectrograms'].shape}")
    
    # Extract data
    actions = sample["actions"]  # (T, action_dim)
    specs = sample["spectrograms"]  # (T, C, H, W)
    
    # Get initial and final spectrograms
    initial_spec = specs[0]  # (C, H, W)
    final_spec = specs[-1]   # (C, H, W)
    
    # Convert to (H, W, C) for models
    initial_spec_hwc = jnp.transpose(initial_spec, (1, 2, 0))  # (128, 128, 3)
    final_spec_hwc = jnp.transpose(final_spec, (1, 2, 0))
    
    # Add batch dimension
    actions_batch = actions[None, ...]  # (1, T, action_dim)
    initial_batch = initial_spec_hwc[None, ...]  # (1, 128, 128, 3)
    
    print(f"\nInitial spec range: [{initial_spec_hwc.min():.3f}, {initial_spec_hwc.max():.3f}]")
    print(f"Final spec range: [{final_spec_hwc.min():.3f}, {final_spec_hwc.max():.3f}]")
    print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Load pretrained autoencoder
    ae = load_pretrained_autoencoder(
        checkpoint_path=args.ae_checkpoint,
        latent_dim=args.latent_dim,
        input_shape=(128, 128, 3),
        seed=args.seed
    )
    
    # Load Action2Sound model
    rngs = nnx.Rngs(args.seed)
    action_dim = actions.shape[-1]
    
    a2s_model = Action2Sound(
        sound_autoencoder=ae,
        seq_len=args.seq_len,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        rngs=rngs
    )
    
    # Load trained weights
    print(f"\nLoading A2S model from: {args.a2s_checkpoint}")
    with open(args.a2s_checkpoint, "rb") as f:
        nnx.update(a2s_model, pickle.load(f))
    
    print(f"✓ Loaded Action2Sound model")
    
    # Predict final spectrogram
    # Model expects: (B, T, action_dim) and (B, H, W, C)
    predicted_spec = a2s_model(actions_batch, initial_batch)  # (1, H, W, C)
    predicted_spec = predicted_spec[0]  # Remove batch dim: (H, W, C)
    
    print(f"\nPredicted spec shape: {predicted_spec.shape}")
    print(f"Predicted spec range: [{predicted_spec.min():.3f}, {predicted_spec.max():.3f}]")
    
    # Compute prediction error
    mse = jnp.mean((predicted_spec - final_spec_hwc) ** 2)
    print(f"Prediction MSE: {mse:.6f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Initial, Final (True), Predicted
    axes[0, 0].imshow(np.array(initial_spec_hwc))
    axes[0, 0].set_title('Initial Image (t=0)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.array(final_spec_hwc))
    axes[0, 1].set_title(f'True Final Image (t={args.seq_len})', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    predicted_np = np.array(predicted_spec)
    predicted_np = np.clip(predicted_np, 0, 1)
    axes[0, 2].imshow(predicted_np)
    axes[0, 2].set_title('Predicted Final Image', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    true_change = np.abs(np.array(final_spec_hwc) - np.array(initial_spec_hwc))
    pred_change = np.abs(predicted_np - np.array(initial_spec_hwc))
    
    axes[1, 0].imshow(true_change * 3)
    axes[1, 0].set_title('True Change (3x)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_change * 3)
    axes[1, 1].set_title('Predicted Change (3x)', fontsize=12)
    axes[1, 1].axis('off')
    
    diff = np.abs(np.array(final_spec_hwc) - predicted_np)
    axes[1, 2].imshow(diff * 5)
    axes[1, 2].set_title(f'Prediction Error (5x)\nMSE: {mse:.6f}', fontsize=12)
    axes[1, 2].axis('off')
    
    fig.suptitle(f'Action2Vision Prediction Test (seq_len={args.seq_len})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    save_path = Path("action2sound_test_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {save_path}")
    plt.show()
    
    print(f"\nAction sequence statistics:")
    print(f"  Mean action magnitude: {jnp.abs(actions).mean():.4f}")
    print(f"  Max action magnitude: {jnp.abs(actions).max():.4f}")
    print(f"  Action std: {actions.std():.4f}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Models')
    parser.add_argument('--mode', type=str, default='test_ae',
                       choices=['train_s2a', 'test_ae', 'test_a2s'],
                       help='Mode: train_s2a, test_ae, or test_a2s')
    parser.add_argument('--data_path', type=str, 
                       default="data/Ant-v5/2025-11-14_13-42-20_seed13/rollout.npz",
                       help='Path to dataset')
    parser.add_argument('--ae_checkpoint', type=str,
                       default="Logs/run_2025-11-14_18-13-00/checkpoints/sound_autoencoder_best.pkl",
                       help='Path to pretrained autoencoder')
    parser.add_argument('--a2s_checkpoint', type=str,
                       default="Logs/A2S/run_2025-11-14_22-51-54/checkpoints/action2sound_final.pkl",
                       help='Path to trained Action2Sound model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    test_action2sound_prediction(args)


if __name__ == "__main__":
    main()