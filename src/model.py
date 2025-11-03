from src.data_loader import *

class Mish(nnx.Module):
    """
    Mish activation
    """
    def __call__(self, x):
        return jnn.mish(x)


class SoundNN(nnx.Module):
    def __init__(self, latent_dim, input_shape, rngs):
        """
        Args:
            latent_dim: size of latent bottleneck
            input_shape: Shape of input (height, width, channels) e.g. (80, 100, 3)
            rngs: Random number generators
        """
        self.rngs = rngs
        self.latent_dim = latent_dim
        
        self.encoder = nnx.Sequential(
            nnx.Conv(in_features=3, out_features=32, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
            nnx.relu,
            nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
            nnx.relu,
            nnx.Conv(in_features=64, out_features=64, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
            nnx.relu,
            nnx.Conv(in_features=64, out_features=64, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
            nnx.relu,
            nnx.Conv(in_features=64, out_features=64, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)
        )

        # TODO change this dummy calculation and explicitly update for proper dimension

        dummy_input = jnp.ones((1, *input_shape))
        encoded = self.encoder(dummy_input)
        self.encoded_shape = encoded.shape[1:]
        flat_dim = jnp.prod(jnp.array(self.encoded_shape))
        
        self.encode_dense = nnx.Linear(flat_dim, latent_dim, rngs=rngs)
        self.decode_dense = nnx.Linear(latent_dim, flat_dim, rngs=rngs)

        self.decoder = nnx.Sequential(
            nnx.ConvTranspose(in_features=64, out_features=64, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
            nnx.relu,
            nnx.ConvTranspose(in_features=64, out_features=64, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
            nnx.relu,
            nnx.ConvTranspose(in_features=64, out_features=64, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
            nnx.relu,
            nnx.ConvTranspose(in_features=64, out_features=32, kernel_size=(3, 3), strides=(2, 2), rngs=rngs),
            nnx.relu,
            nnx.ConvTranspose(in_features=32, out_features=3, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)
        )

    def __call__(self, x):
        batch_size = x.shape[0]
        
        x = self.encoder(x)
        x_flat = x.reshape((batch_size, -1))
        latent = self.encode_dense(x_flat)
        x = self.decode_dense(latent)
        x = x.reshape((batch_size, *self.encoded_shape))
        x = self.decoder(x)
        
        return x, latent

class Action2Sound(nnx.Module):
    def __init__(self, sound_autoencoder, rngs):
        """
        Args:
            sound_autoencoder: pre-trained autoencoder for spectrogram <-> latent
        """

        self.sound_net = nnx.Sequential(

        )

        self.decode = sound_autoencoder.decoder

class Sound2Action(nnx.Module):
    def __init__(self, sound_autoencoder, rngs):
        """
        Args:
            sound_autoencoder: pre-trained autoencoder for spectrogram <-> latent
        """

        self.encode = sound_autoencoder.encoder

        self.action_net = nnx.Sequential(
            
        )


class AcousticModel(nnx.Module):
    def __init__(self, rngs):
        super(AcousticModel, self).__init__()

    def __call__(self, x):
        return x