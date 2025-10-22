from src.data_loader import *
import optax

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

        # jax.debug.print(f"State shape: {states.shape[0]}, Action shape: {actions.shape[0]}")

        # def rollout(current_state, action):
        #     pred_delta = model(jnp.concatenate([current_state, action], axis=-1))
        #     pred_state = current_state + pred_delta
        #     return pred_state, pred_state

        # def one_step_teacher(_, t):
        #     pred_delta = model(jnp.concatenate([states[t], actions[t]], axis=-1))
        #     pred_state = states[t] + pred_delta
        #     true_state = states[t+1]
        #     step_loss = ((pred_state - true_state)**2).mean()
        #     return None, step_loss

        # (_, teacher_d_losses) = jax.lax.scan(
        #     one_step_teacher,
        #     None,
        #     jnp.arange(T)
        # )
        # (_, rollout_pred) = jax.lax.scan(
        #     rollout,
        #     states[0],
        #     actions
        # )
        # true_future_states = states[1:]
        # rollout_d_losses = ((rollout_pred - true_future_states)**2).mean(axis=(1, 2))
        # L_rd, L_tf = rollout_d_losses.mean(), teacher_d_losses.mean()
        # L_d = alpha*L_tf + (1.0 - alpha)*L_rd

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