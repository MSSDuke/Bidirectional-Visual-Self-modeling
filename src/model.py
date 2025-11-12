from src.data_loader import *


class SoundAE(nnx.Module):
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
    
    def encode_latent(self, x, batch_size):
        x = self.encoder(x)
        x = x.reshape((batch_size, -1))
        x = self.encode(x)
        return x

    def decode_latent(self, x, batch_size):
        x = self.decode_dense(x)
        x = x.reshape((batch_size, *self.encoded_shape))
        x = self.decoder(x)
        return x


class Action2Sound(nnx.Module):
    def __init__(self, sound_autoencoder, input_shape, decode_shape, rngs):
        """
        TCN converts concatenated state/action history into latent representation and
        passes through decoder to get predicted spectrogram

        Args:
            sound_autoencoder: pre-trained autoencoder for spectrogram <-> latent
        """

        # TODO try MLP, TCN, LSTM, GRU

        self.sound_net = nnx.Sequential(
            nnx.Conv(37, 64, kernel_size=3, kernel_dilation=1, padding="CAUSAL", rngs=rngs),
            nnx.relu,
            nnx.Conv(64, 64, kernel_size=3, kernel_dilation=2, padding="CAUSAL", rngs=rngs),
            nnx.relu,
            nnx.Conv(64, 128, kernel_size=3, kernel_dilation=4, padding="CAUSAL", rngs=rngs),
            nnx.relu,
            nnx.Conv(128, 128, kernel_size=3, kernel_dilation=8, padding="CAUSAL", rngs=rngs),
            nnx.relu,
            nnx.Conv(128, 256, kernel_size=3, kernel_dilation=16, padding="CAUSAL", rngs=rngs),
            nnx.relu
        )

        # dummy calculations, replace later
        x_one = jnp.ones((1, *input_shape))
        en = self.sound_net(x_one)
        self.en_shape = en.shape[1:]
        flat_dim = jnp.prod(jnp.array(self.en_shape))

        self.to_latent = nnx.Linear(flat_dim, decode_shape, rngs=rngs)

        self.decode = sound_autoencoder.decode_latent

    def __call__(self, x):
        batch_size = x.shape[0]
        x = self.sound_net(x)
        x = jnp.reshape(x, (batch_size, -1))
        x = self.to_latent(x)
        x = self.decode(x, batch_size)
        return x

class Sound2Action(nnx.Module):
    def __init__(self, sound_autoencoder, latent_dim, T, action_dim, rngs):
        """
        Args:
            sound_autoencoder: pre-trained autoencoder for spectrogram <-> latent
            latent_dim: dimensionality of latent representation
            T: number of timesteps
            action_dim: dimensionality of action vector
        """

        self.T = T
        self.action_dim = action_dim
        self.nn_head_dim = self.T * self.action_dim
        
        self.sound_ae = sound_autoencoder

        self.action_net = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(latent_dim, latent_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(latent_dim, self.nn_head_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(self.nn_head_dim, self.nn_head_dim, rngs=rngs)
        )

        self.TCN_head = nnx.Sequential(
            nnx.Conv(action_dim, action_dim, kernel_size=3, kernel_dilation=1, rngs=rngs),
            nnx.relu,
            nnx.Conv(action_dim, action_dim, kernel_size=3, kernel_dilation=2, rngs=rngs),
            nnx.relu
        )

    def __call__(self, x):
        # TODO replace temp __call__ below
        if x.ndim == 3:
            x = x[None, ...]
            
        batch_size = x.shape[0]
        
        x = self.sound_ae.encoder(x)
        x_flat = x.reshape((batch_size, -1))
        latent = self.sound_ae.encode_dense(x_flat)
        
        x = self.action_net(latent)
        x = x.reshape(batch_size, self.T, self.action_dim)
        
        if batch_size == 1:
            x = x[0]
            x = self.TCN_head(x)
        else:
            x_list = []
            for i in range(batch_size):
                xi = self.TCN_head(x[i])
                x_list.append(xi)
            x = jnp.stack(x_list, axis=0)
        
        return x


class AcousticModel(nnx.Module):
    def __init__(self, rngs):
        super(AcousticModel, self).__init__()

    def __call__(self, x):
        return x