from data_loader import *


"""
Helper functions
"""

def glorot_init_params():
    """
    Glorot initialization of network weights
    """
    pass

def init_bias(output_dim, dtype=jnp.float32):
    return jnp.zeros(output_dim, dtype=dtype)

class Mish(nnx.Module):
    """
    Mish activation function compatible with Flax NNX
    """
    def __call__(x):
        return jnn.mish(x)
    
class Lipschitz(nnx.Module):
    """
    Lipschitz normalization for linear layers
    """
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)




class ConvEncoder_2D(nnx.Module):
    """
    Encodes spectrogram data with 2D convolutions
    """
    def __init__(self, input_dim=None, output_dim=None):
        super(ConvEncoder_2D, self).__init__()

    def forward(self, x):
        pass


class MLPBase(nnx.Module):
    """
    MLP (maybe include Lipschitz normalization) 
    """
    def __init__(self, input_dim=None, output_dim=None, params=None):
        super(MLPBase, self).__init__()

        self.mlp = nnx.Sequential(
            nnx.Linear(input_dim, 128),
            Mish(),
            Lipschitz(),
            nnx.Linear(128, 128),
            Mish(),
            Lipschitz(),
            nnx.Linear(128, 128),
            Mish(),
            Lipschitz(),
            nnx.Linear(128, 128),
            Mish(),
            Lipschitz(),
            nnx.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

# TODO decoder, maybe FiLM layer after MLP?

class SoundSM(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, inference_mode=False, params=None, H=5, T=1):        
        super(SoundSM, self).__init__()

        self.conv_encode = ConvEncoder_2D()
        self.mlp = MLPBase()

    def forward(self, x):
        pass