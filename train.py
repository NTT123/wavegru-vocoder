"""
Train WaveGRU vocoder
"""


import time
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import opax
import pax
import tensorflow as tf
import yaml

from utils import load_ckpt, load_config, save_ckpt
from wavegru import WaveGRU

config = load_config()


def get_data_loader(data_dir: Path, batch_size: int):
    """
    return a data loader of mini-batches
    """
    return (
        tf.data.experimental.load(str(data_dir), compression="GZIP")
        .shuffle(500_000, seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )


def loss_fn(net, batch):
    """
    return negative log likelihood
    """
    mel, mu = batch
    input_mu, target_mu = mu[:, :-1], mu[:, 1:]
    net, logit = pax.purecall(net, mel, input_mu)
    llh = jax.nn.log_softmax(logit, axis=-1)
    llh = llh * jax.nn.one_hot(target_mu, num_classes=256, axis=-1)
    llh = jnp.sum(llh, axis=-1)
    loss = -jnp.mean(llh)
    return loss, net


def train_step(net, optim, batch):
    """
    one training step
    """
    (loss, net), grads = pax.value_and_grad(loss_fn, has_aux=True)(net, batch)
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, loss


def train(batch_size: int = config["batch_size"], lr: float = config["lr"]):
    """
    train wavegru model
    """
    net = WaveGRU()

    def lr_decay(step):
        e = jnp.floor(step * 1.0 / 100_000)
        return jnp.exp(-e) * lr

    optim = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale_by_adam(),
        opax.scale_by_schedule(lr_decay),
    )

    data_loader = get_data_loader(config["tf_data_dir"], batch_size)
    step = -1
    ckpts = sorted(Path(config["ckpt_dir"]).glob(f"{config['model_prefix']}_*.ckpt"))
    if len(ckpts) > 0:
        print(f"Load checkpoint at {ckpts[-1]}")
        step, net, optim = load_ckpt(net, optim, ckpts[-1])
        net, optim = jax.device_put((net, optim))

    total_loss = 0.0
    start = time.perf_counter()
    for batch in data_loader:
        step += 1
        batch = jax.device_put(batch)
        net, optim, loss = train_step(net, optim, batch)
        total_loss += loss

        if step % 100 == 0:
            end = time.perf_counter()
            duration = end - start
            start = end
            loss = total_loss / 100
            print("step  {:07d}  loss {:.3f}  {:.2f}s".format(step, loss, duration))
            total_loss = 0.0

        if step % 1000 == 0:
            save_ckpt(step, net, optim, config["ckpt_dir"], config["model_prefix"])


if __name__ == "__main__":
    fire.Fire(train)
