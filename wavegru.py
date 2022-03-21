"""
WaveGRU model: melspectrogram => mu-law encoded waveform
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import pax
from pax import GRUState
from tqdm.cli import tqdm


class ReLU(pax.Module):
    def __call__(self, x):
        return jax.nn.relu(x)


def dilated_residual_conv_block(dim, kernel, stride, dilation):
    """
    Use dilated convs to enlarge the receptive field
    """
    return pax.Sequential(
        pax.Conv1D(dim, dim, kernel, stride, dilation, "VALID", with_bias=False),
        pax.LayerNorm(dim, -1, True, True),
        ReLU(),
        pax.Conv1D(dim, dim, 1, 1, 1, "VALID", with_bias=False),
        pax.LayerNorm(dim, -1, True, True),
        ReLU(),
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


def up_block(in_dim, out_dim, factor, relu=True):
    """
    Tile >> Conv >> BatchNorm >> ReLU
    """
    f = pax.Sequential(
        lambda x: tile_1d(x, factor),
        pax.Conv1D(
            in_dim, out_dim, 2 * factor, stride=1, padding="VALID", with_bias=False
        ),
        pax.LayerNorm(out_dim, -1, True, True),
    )
    if relu:
        f >>= ReLU()
    return f


class Upsample(pax.Module):
    """
    Upsample melspectrogram to match raw audio sample rate.
    """

    def __init__(self, input_dim, hidden_dim, rnn_dim, upsample_factors):
        super().__init__()
        self.input_conv = pax.Sequential(
            pax.Conv1D(input_dim, hidden_dim, 1, with_bias=False),
            pax.LayerNorm(hidden_dim, -1, True, True),
        )
        self.upsample_factors = upsample_factors
        self.dilated_convs = [
            dilated_residual_conv_block(hidden_dim, 3, 1, 2 ** i) for i in range(5)
        ]
        self.up_factors = upsample_factors[:-1]
        self.up_blocks = [
            up_block(hidden_dim, hidden_dim, x) for x in self.up_factors[:-1]
        ]
        self.up_blocks.append(
            up_block(hidden_dim, hidden_dim, self.up_factors[-1], relu=False)
        )
        self.x2zrh_fc = pax.Linear(hidden_dim, rnn_dim * 3)

        self.final_tile = upsample_factors[-1]

    def __call__(self, x):
        x = self.input_conv(x)
        for residual in self.dilated_convs:
            y = residual(x)
            pad = (x.shape[1] - y.shape[1]) // 2
            x = x[:, pad:-pad, :] + y

        for f in self.up_blocks:
            x = f(x)

        x = self.x2zrh_fc(x)

        x = tile_1d(x, self.final_tile)
        return x


class GRU(pax.Module):
    """
    A customized GRU module.
    """

    input_dim: int
    hidden_dim: int

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.h_zrh_fc = pax.Linear(
            hidden_dim,
            hidden_dim * 3,
            w_init=jax.nn.initializers.variance_scaling(
                1, "fan_out", "truncated_normal"
            ),
        )

    def initial_state(self, batch_size: int) -> GRUState:
        """Create an all zeros initial state."""
        return GRUState(jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32))

    def __call__(self, state: GRUState, x) -> Tuple[GRUState, jnp.ndarray]:
        hidden = state.hidden
        x_zrh = x
        h_zrh = self.h_zrh_fc(hidden)
        x_zr, x_h = jnp.split(x_zrh, [2 * self.hidden_dim], axis=-1)
        h_zr, h_h = jnp.split(h_zrh, [2 * self.hidden_dim], axis=-1)

        zr = x_zr + h_zr
        zr = jax.nn.sigmoid(zr)
        z, r = jnp.split(zr, 2, axis=-1)

        h_hat = x_h + r * h_h
        h_hat = jnp.tanh(h_hat)

        h = (1 - z) * hidden + z * h_hat
        return GRUState(h), h


class Pruner(pax.Module):
    """
    Base class for pruners
    """

    def compute_sparsity(self, step):
        t = jnp.power(1 - (step * 1.0 - 1_000) / 200_000, 3)
        z = 0.95 * jnp.clip(1.0 - t, a_min=0, a_max=1)
        return z

    def prune(self, step, weights):
        """
        Return a mask
        """
        z = self.compute_sparsity(step)
        x = weights
        H, W = x.shape
        x = x.reshape(H // 4, 4, W // 4, 4)
        x = jnp.abs(x)
        x = jnp.sum(x, axis=(1, 3), keepdims=True)
        q = jnp.quantile(jnp.reshape(x, (-1,)), z)
        x = x >= q
        x = jnp.tile(x, (1, 4, 1, 4))
        x = jnp.reshape(x, (H, W))
        return x


class GRUPruner(Pruner):
    def __init__(self, gru):
        super().__init__()
        self.h_zrh_fc_mask = jnp.ones_like(gru.h_zrh_fc.weight) == 1

    def __call__(self, gru: pax.GRU):
        """
        Apply mask after an optimization step
        """
        zrh_masked_weights = jnp.where(self.h_zrh_fc_mask, gru.h_zrh_fc.weight, 0)
        gru = gru.replace_node(gru.h_zrh_fc.weight, zrh_masked_weights)
        return gru

    def update_mask(self, step, gru: pax.GRU):
        """
        Update internal masks
        """
        z_weight, r_weight, h_weight = jnp.split(gru.h_zrh_fc.weight, 3, axis=1)
        z_mask = self.prune(step, z_weight)
        r_mask = self.prune(step, r_weight)
        h_mask = self.prune(step, h_weight)
        self.h_zrh_fc_mask *= jnp.concatenate((z_mask, r_mask, h_mask), axis=1)


class LinearPruner(Pruner):
    def __init__(self, linear):
        super().__init__()
        self.mask = jnp.ones_like(linear.weight) == 1

    def __call__(self, linear: pax.Linear):
        """
        Apply mask after an optimization step
        """
        return linear.replace(weight=jnp.where(self.mask, linear.weight, 0))

    def update_mask(self, step, linear: pax.Linear):
        """
        Update internal masks
        """
        self.mask *= self.prune(step, linear.weight)


class WaveGRU(pax.Module):
    """
    WaveGRU vocoder model.
    """

    def __init__(self, mel_dim=80, rnn_dim=1024, upsample_factors=(5, 3, 20)):
        super().__init__()
        self.embed = pax.Embed(256, 3 * rnn_dim)
        self.upsample = Upsample(
            input_dim=mel_dim,
            hidden_dim=512,
            rnn_dim=rnn_dim,
            upsample_factors=upsample_factors,
        )
        self.rnn = GRU(rnn_dim)
        self.o1 = pax.Linear(rnn_dim, rnn_dim)
        self.o2 = pax.Linear(rnn_dim, 256)
        self.gru_pruner = GRUPruner(self.rnn)
        self.o1_pruner = LinearPruner(self.o1)
        self.o2_pruner = LinearPruner(self.o2)

    def output(self, x):
        x = self.o1(x)
        x = jax.nn.relu(x)
        x = self.o2(x)
        return x

    def inference(self, mel, no_gru=False, seed=42):
        """
        generate waveform form melspectrogram
        """

        @jax.jit
        def step(rnn_state, mel, rng_key, x):
            x = self.embed(x)
            x = x + mel
            rnn_state, x = self.rnn(rnn_state, x)
            x = self.output(x)
            rng_key, next_rng_key = jax.random.split(rng_key, 2)
            x = jax.random.categorical(rng_key, x, axis=-1)
            return rnn_state, next_rng_key, x

        y = self.upsample(mel)
        if no_gru:
            return y
        x = jnp.array([127], dtype=jnp.int32)
        rnn_state = self.rnn.initial_state(1)
        output = []
        rng_key = jax.random.PRNGKey(seed)
        for i in tqdm(range(y.shape[1])):
            rnn_state, rng_key, x = step(rnn_state, y[:, i], rng_key, x)
            output.append(x)
        x = jnp.concatenate(output, axis=0)
        return x

    def __call__(self, mel, x):
        x = self.embed(x)
        y = self.upsample(mel)
        pad_left = (x.shape[1] - y.shape[1]) // 2
        pad_right = x.shape[1] - y.shape[1] - pad_left
        x = x[:, pad_left:-pad_right]
        x = x + y
        _, x = pax.scan(
            self.rnn,
            self.rnn.initial_state(x.shape[0]),
            x,
            time_major=False,
        )
        x = self.output(x)
        return x
