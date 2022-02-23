"""
Train WaveGRU vocoder
"""

import time
from pathlib import Path
from typing import Deque

import fire
import jax
import jax.numpy as jnp
import opax
import pax
import tensorflow as tf
import yaml

from utils import load_ckpt, load_config, save_ckpt
from wavegru import WaveGRU

CONFIG = load_config()


def double_buffer(ds):
    """
    create a double-buffer iterator
    """
    batch = None

    for next_batch in ds:
        assert next_batch is not None
        next_batch = jax.device_put(next_batch)
        if batch is not None:
            yield batch
        batch = next_batch

    if batch is not None:
        yield batch


def get_data_loader(data_dir: Path, batch_size: int):
    """
    return a data loader of mini-batches
    """
    it = (
        tf.data.experimental.load(str(data_dir), compression="GZIP")
        .repeat()
        .shuffle(200_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )
    return double_buffer(it)


def loss_fn(net, batch):
    """
    return negative log likelihood
    """
    mel, mu = batch
    mel = mel.astype(jnp.float32)
    input_mu, target_mu = mu[:, :-1], mu[:, 1:]
    net, logit = pax.purecall(net, mel, input_mu)
    pad_left = (target_mu.shape[1] - logit.shape[1]) // 2
    pad_right = target_mu.shape[1] - logit.shape[1] - pad_left
    target_mu = target_mu[:, pad_left:-pad_right]
    llh = jax.nn.log_softmax(logit, axis=-1)
    llh = llh * jax.nn.one_hot(target_mu, num_classes=256, axis=-1)
    llh = jnp.sum(llh, axis=-1)
    loss = -jnp.mean(llh)
    return loss, net


@jax.jit
def train_step(net, optim, batch):
    """
    one training step
    """
    (loss, net), grads = pax.value_and_grad(loss_fn, has_aux=True)(net, batch)
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, loss


def train(batch_size: int = CONFIG["batch_size"], lr: float = CONFIG["lr"]):
    """
    train wavegru model
    """
    net = WaveGRU(
        mel_dim=CONFIG["mel_dim"],
        embed_dim=CONFIG["embed_dim"],
        rnn_dim=CONFIG["rnn_dim"],
        upsample_factors=CONFIG["upsample_factors"],
    )

    def lr_decay(step):
        e = jnp.floor(step * 1.0 / 100_000)
        return jnp.exp(-e) * lr

    optim = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale_by_adam(),
        opax.scale_by_schedule(lr_decay),
    ).init(net.parameters())

    data_loader = get_data_loader(CONFIG["tf_data_dir"], batch_size)
    step = -1
    ckpts = sorted(Path(CONFIG["ckpt_dir"]).glob(f"{CONFIG['model_prefix']}_*.ckpt"))
    if len(ckpts) > 0:
        print(f"Load checkpoint at {ckpts[-1]}")
        step, net, optim = load_ckpt(net, optim, ckpts[-1])
        net, optim = jax.device_put((net, optim))

    start = time.perf_counter()
    losses = Deque(maxlen=500)
    for batch in data_loader:
        step += 1
        net, optim, loss = train_step(net, optim, batch)
        losses.append(loss)

        if step % 100 == 0:
            loss = sum(losses).item() / len(losses)
            end = time.perf_counter()
            duration = end - start
            start = end
            print(
                "step  {:07d}  loss {:.3f}  LR {:.3e}  {:.2f}s".format(
                    step, loss, optim[-1].learning_rate, duration
                ),
                flush=True,
            )

        if step % 10_000 == 0:
            save_ckpt(step, net, optim, CONFIG["ckpt_dir"], CONFIG["model_prefix"])


if __name__ == "__main__":
    fire.Fire(train)
