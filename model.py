from data_loader import *

states = []
actions = []

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
    MLP with Lipschitz normalization 
    """
    def __init__(self, input_dim=None, output_dim=None, H=5, T=1):
        super(MLPBase, self).__init__()

    def forward(self, x):
        pass

# TODO decoder, maybe FiLM layer after MLP?

class SoundSM(nnx.Module):
    def __init__(self, input_dim=None, output_dim=None, inference_mode=False, params=None):        
        super(SoundSM, self).__init__()

        self.conv_encode = ConvEncoder_2D()
        self.mlp = MLPBase()

        if params is not None:
            self.params = params
            W, _ = self.params