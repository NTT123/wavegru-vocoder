"""
WaveGRU model: melspectrogram => mu-law encoded waveform
"""

import jax
import jax.numpy as jnp
import pax
from tqdm.cli import tqdm


def conv_block(in_ft: int, out_ft: int, kernel_size: int, padding: str):
    """
    conv + batchnorm + relu
    """
    return pax.Sequential(
        pax.Conv1D(in_ft, out_ft, kernel_size, padding=padding, with_bias=False),
        pax.BatchNorm1D(out_ft),
        jax.nn.relu,
    )


def residual_block(dim: int, kernel_size) -> pax.Sequential:
    """
    conv >> relu >> conv
    """
    return pax.Sequential(
        conv_block(dim, dim, kernel_size=kernel_size, padding="VALID"),
        jax.nn.relu,
        conv_block(dim, dim, kernel_size=kernel_size, padding="VALID"),
    )


def tile_1d(x, factor):
    """
    Tile tensor of shape N, L, D into N, L*factor, D
    """
    N, L, D = x.shape
    x = x[:, :, None, :]
    x = jnp.tile(x, (1, 1, factor, 1))
    x = jnp.reshape(x, (N, L * factor, D))
    return x


class Upsample(pax.Module):
    """
    Upsample melspectrogram to match raw audio sample rate.
    """

    def __init__(self, input_dim, upsample_factors=(5, 4, 15)):
        super().__init__()
        self.input_conv = pax.Conv1D(input_dim, 512, 1, with_bias=False)
        self.upsample_factors = upsample_factors
        self.residual_blocks = [residual_block(512, 3) for _ in range(10)]
        self.up_factors = upsample_factors[:-1]
        self.up_conv_blocks = [
            conv_block(512, 512, x * 2, "VALID") for x in self.up_factors
        ]
        self.final_tile = upsample_factors[-1]

    def __call__(self, x):
        x = self.input_conv(x)
        for residual in self.residual_blocks:
            y = residual(x)
            pad = (x.shape[1] - y.shape[1]) // 2
            x = x[:, pad:-pad, :] + y
            x = jax.nn.relu(x)

        for i, conv in enumerate(self.up_conv_blocks):
            x = tile_1d(x, self.up_factors[i])
            x = conv(x)

        x = tile_1d(x, self.final_tile)
        return x


class WaveGRU(pax.Module):
    """
    WaveGRU vocoder model
    """

    def __init__(self, mel_dim=80, embed_dim=32, rnn_dim=512):
        super().__init__()
        self.embed = pax.Embed(256, embed_dim)
        self.upsample = Upsample(input_dim=mel_dim)
        self.rnn = pax.GRU(embed_dim + rnn_dim, rnn_dim)
        self.output = pax.Sequential(
            pax.Linear(rnn_dim, 256),
            jax.nn.relu,
            pax.Linear(256, 256),
        )

    def inference(self, mel):
        """
        generate waveform form melspectrogram
        """

        @jax.jit
        def step(rnn_state, mel, rng_key, x):
            x = self.embed(x)
            x = jnp.concatenate((x, mel), axis=-1)
            rnn_state, x = self.rnn(rnn_state, x)
            x = self.output(x)
            rng_key, next_rng_key = jax.random.split(rng_key, 2)
            x = jax.random.categorical(rng_key, x, axis=-1)
            return rnn_state, next_rng_key, x

        y = self.upsample(mel)
        x = jnp.array([127], dtype=jnp.int32)
        rnn_state = self.rnn.initial_state(1)
        output = []
        for i in tqdm(range(y.shape[1])):
            rnn_state, rng_key, x = step(rnn_state, mel[i], rng_key, x)
            output.append(x)

        x = jnp.stack(x, axis=1)
        return x

    def __call__(self, mel, x):
        x = self.embed(x)
        y = self.upsample(mel)
        pad_left = (x.shape[1] - y.shape[1]) // 2
        pad_right = x.shape[1] - y.shape[1] - pad_left
        x = x[:, pad_left:-pad_right]
        x = jnp.concatenate((x, y), axis=-1)
        _, x = pax.scan(
            self.rnn,
            self.rnn.initial_state(x.shape[0]),
            x,
            time_major=False,
        )
        x = self.output(x)
        return x
