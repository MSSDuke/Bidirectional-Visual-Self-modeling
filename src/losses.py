from src.data_loader import *
import optax

def sound_error(model, x):
    out = model(x)
    return jnp.mean((out - x)**2).mean()



@nnx.jit
def train_step(model, optimizer, batch, alpha):
    """
    Computes loss and gradients for a single training step

    Args:
        model: multiheaded dynamics+acoustic prediction model
        optimizer: nnx.Optimizer
        batch: Dict with keys 'states', 'actions', and 'spectrograms'
    """

    # TODO spectrogram loss implementations, maybe have weights (w_d, w_s) be learned parameters

    def loss_fn(model):
        states = batch["states"]
        actions = batch["actions"]
        # spectrograms = batch["spectrograms"]
        T = states.shape[0] - 1

        def rollout(current_state, t):
            pred_state = model(jnp.concatenate([current_state, actions[t]], axis=-1))
            true_state = states[t+1]
            step_loss = ((pred_state - true_state)**2).mean()
            return pred_state, step_loss

        init_d_carry = states[0]
        (_, step_d_losses) = jax.lax.scan(
            rollout,
            init_d_carry,
            jnp.arange(T)
        )
        L_d = step_d_losses.mean()

        # TODO implement spectrogram

        L_s = 0.0

        L = L_d + L_s

        return L, {"L_d": L_d, "L_s": L_s, "L": L}


    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads=grads, model=model)

    return loss, metrics


def a2s_loss(model, batch):
    """
    Compute loss for Action2Sound model
    
    Args:
        model: Action2Sound model
        batch: Dict with 'states', 'actions', 'spectrograms'
    
    Returns:
        loss: MSE between predicted and true spectrograms
    """
    actions = batch["actions"]
    spectrograms = batch["spectrograms"]
    
    T, B = actions.shape[:2]
    C, H, W = spectrograms.shape[2:]
    
    state_action = jnp.concatenate([actions], axis=-1)
    state_action = jnp.transpose(state_action, (1, 0, 2))
    
    initial_specs = spectrograms[0]
    target_specs = spectrograms[-1]

    initial_specs = jnp.transpose(initial_specs, (0, 2, 3, 1))
    target_specs = jnp.transpose(target_specs, (0, 2, 3, 1))
    
    pred_specs = model(state_action, initial_specs)
    
    loss = jnp.mean((pred_specs - target_specs) ** 2)
    return loss

def s2a_loss(model, batch):
    """
    Compute loss for Sound2Action model
    
    Args:
        model: Sound2Action model
        batch: Dict with 'states', 'actions', 'spectrograms'
    
    Returns:
        loss: MSE between predicted and true actions
    """
    actions = batch["actions"]
    spectrograms = batch["spectrograms"]
    
    T, B = actions.shape[:2]
    C, H, W = spectrograms.shape[2:]
    
    initial_specs = spectrograms[0]
    final_specs = spectrograms[-1]
    
    initial_specs = jnp.transpose(initial_specs, (0, 2, 3, 1))
    final_specs = jnp.transpose(final_specs, (0, 2, 3, 1))
    
    pred_actions = model(initial_specs, final_specs)
    target_actions = jnp.transpose(actions, (1, 0, 2))
    
    loss = jnp.mean((pred_actions - target_actions) ** 2)
    return loss