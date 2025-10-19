from src.data_loader import *

"""
Helper functions
"""

class Mish(nnx.Module):
    """
    Mish activation function compatible with Flax NNX
    """
    def __call__(self, x):
        return jnn.mish(x)


class ConvEncoder_2D(nnx.Module):
    """
    Encodes spectrogram data with 2D convolutions
    """
    def __init__(self, input_dim=None, output_dim=None, rngs=None):
        super(ConvEncoder_2D, self).__init__()

    def __call__(self, x):
        pass


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
    # TODO include inference mode, maybe add spectral/lipschitz normalizations after initial training + residuals + dropout
    def __init__(self, state_size=None, action_size=None, hidden_width=256, rngs=None):
        super(DynamicsNN, self).__init__()

        input_dim = state_size + action_size
        output_dim = state_size

        self.mlp = nnx.Sequential(
            nnx.Linear(input_dim, hidden_width, rngs=rngs),
            nnx.LayerNorm(hidden_width, rngs=rngs),
            Mish(),
            nnx.Linear(hidden_width, hidden_width, rngs=rngs),
            nnx.LayerNorm(hidden_width, rngs=rngs),
            Mish(),
            nnx.Linear(hidden_width, hidden_width // 2, rngs=rngs),
            nnx.LayerNorm(hidden_width // 2, rngs=rngs),
            Mish(),
            nnx.Linear(hidden_width // 2, output_dim, rngs=rngs)
        )
        
    def __call__(self, x):
        return self.mlp(x)

# TODO audio decoder, maybe FiLM layer after MLP?

class SoundSM(nnx.Module):
    def __init__(self, state_size=None, action_size=None, hidden_width=256, rngs=None):        
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