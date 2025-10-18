from src.data_loader import *
import optax

@nnx.jit
def train_step(model, optimizer, batch):
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
            pred_state = jax.vmap(model.dynamics_mlp)(jnp.concatenate([current_state, actions[t]], axis=-1))
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
    optimizer.update(grads)

    return loss, metrics