from src.data_loader import *

"""
Helper functions
"""

class Mish(nnx.Module):
    """
    Mish activation
    """
    def __call__(self, x):
        return jnn.mish(x)


class ConvEncoder_2D(nnx.Module):
    """
    Encodes spectrogram data with 2D convolutions
    """
    def __init__(self, input_dim=None, output_dim=None, rngs=None):
        super(ConvEncoder_2D, self).__init__()

        # TODO IMPLEMENT

    def __call__(self, x):
        pass

# TODO maybe eventually scale to multi-step rollout predictions

class DynamicsNN(nnx.Module):
    """
    Learn mapping from state-action history to future states after rollout
    of some action trajectory

    Args:
        state_size: dimensionality of state space
        action_size: dimensionality of action space
        H: state-action history length (in timesteps) # TODO add later
        T: action trajectory length (in timesteps) # TODO add later
        hidden_width: width of NN hidden layers
    
    """
    def __init__(self, state_size=None, action_size=None, hidden_width=512, rngs=None):
        super(DynamicsNN, self).__init__()

        input_dim = state_size + action_size # shape=(37,)
        output_dim = state_size # shape=(29,)

        self.mlp = nnx.Sequential(
            nnx.SpectralNorm(nnx.Linear(input_dim, hidden_width, rngs=rngs), rngs=rngs),
            Mish(),
            nnx.SpectralNorm(nnx.Linear(hidden_width, hidden_width, rngs=rngs), rngs=rngs),
            Mish(),
            nnx.SpectralNorm(nnx.Linear(hidden_width, hidden_width // 2, rngs=rngs), rngs=rngs),
            Mish(),
            nnx.SpectralNorm(nnx.Linear(hidden_width // 2, hidden_width // 4, rngs=rngs), rngs=rngs),
            Mish(),
            nnx.SpectralNorm(nnx.Linear(hidden_width // 4, hidden_width // 4, rngs=rngs), rngs=rngs),
            Mish(),
            nnx.LayerNorm(hidden_width // 4, rngs=rngs),
            nnx.Linear(hidden_width // 4, output_dim, rngs=rngs),
        )
        
    def __call__(self, x):
        return self.mlp(x)

# TODO audio decoder, maybe FiLM layer after MLP?

class SoundSM(nnx.Module):
    def __init__(self, state_size=None, action_size=None, hidden_width=512, rngs=None):        
        super(SoundSM, self).__init__()

        self.conv_encode = ConvEncoder_2D(rngs=rngs)

        self.dynamics_mlp = DynamicsNN(
            state_size=state_size,
            action_size=action_size,
            hidden_width=hidden_width,
            rngs=rngs
        )

    def __call__(self, x):
        x = self.dynamics_mlp(x)
        return x