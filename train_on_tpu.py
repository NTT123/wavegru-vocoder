"""
Train WaveGRU vocoder on TPU
"""

import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Deque

import fire
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import numpy as np
import opax
import pax
import tensorflow as tf

from utils import load_ckpt, load_config, save_ckpt, update_gru_mask
from wavegru import WaveGRU

# TPU setup
if "COLAB_TPU_ADDR" in os.environ:
    jax.tools.colab_tpu.setup_tpu()
DEVICES = jax.devices()
NUM_DEVICES = len(DEVICES)
STEPS_PER_CALL = 10
print("Devices:", DEVICES)

CONFIG = load_config()

if "MODEL_PREFIX" in os.environ:
    MODEL_PREFIX = os.environ["MODEL_PREFIX"]
else:
    MODEL_PREFIX = CONFIG["model_prefix"]

NUM_STEPS_PER_CALL = 10


def batch_reshape(x, K):
    """
    add a new first dimension
    """
    N, *L = x.shape
    return np.reshape(x, (K, N // K, *L))


def _device_put_sharded(sharded_tree):
    leaves, treedef = jax.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return jax.device_put_sharded(
        [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)], DEVICES
    )


def pmap_double_buffer(ds):
    """
    create a double buffer iterator for jax.pmap training
    """
    batch = None
    for next_batch in ds:
        assert next_batch is not None
        next_batch = jax.tree_map(partial(batch_reshape, K=NUM_DEVICES), next_batch)
        next_batch = _device_put_sharded(next_batch)
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
    return pmap_double_buffer(it)


def loss_fn(net, batch):
    """
    return negative log likelihood
    """
    mel, mu = batch
    mu = jnp.clip(mu, a_min=0, a_max=255)
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
def train_step(net_optim_step, batch):
    """
    one training step
    """
    net, optim, step = net_optim_step
    step = step + 1
    (loss, net), grads = pax.value_and_grad(loss_fn, has_aux=True)(net, batch)
    jax.lax.pmean(grads, axis_name="i")

    net, optim = opax.apply_gradients(net, optim, grads)
    net = update_gru_mask(step, net)
    net = net.replace(rnn=net.gru_pruner(net.rnn))
    net = net.replace(o1=net.o1_pruner(net.o1))
    net = net.replace(o2=net.o2_pruner(net.o2))
    return (net, optim, step), loss


@partial(jax.pmap, axis_name="i")
def multiple_train_step(step, net, optim, batch):
    """
    multiple training steps
    """
    batch = jax.tree_map(partial(batch_reshape, K=STEPS_PER_CALL), batch)
    (net, optim, step), losses = pax.scan(train_step, (net, optim, step), batch)
    return step, net, optim, jnp.mean(losses)


def train(batch_size: int = CONFIG["batch_size"], lr: float = CONFIG["lr"]):
    """
    train wavegru model
    """
    net = WaveGRU(
        mel_dim=CONFIG["mel_dim"],
        rnn_dim=CONFIG["rnn_dim"],
        upsample_factors=CONFIG["upsample_factors"],
    )

    def lr_decay(step):
        e = jnp.floor(step * 1.0 / 200_000)
        return jnp.exp2(-e) * lr

    optim = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale_by_adam(),
        opax.scale_by_schedule(lr_decay),
    ).init(net.parameters())

    step = -STEPS_PER_CALL
    ckpts = sorted(Path(CONFIG["ckpt_dir"]).glob(f"{MODEL_PREFIX}_*.ckpt"))
    if len(ckpts) > 0:
        print(f"Load checkpoint at {ckpts[-1]}")
        step, net, optim = load_ckpt(net, optim, ckpts[-1])
        net, optim = jax.device_put((net, optim))

    # replicate on multiple devices
    net, optim = jax.device_put_replicated((net, optim), DEVICES)

    start = time.perf_counter()
    losses = Deque(maxlen=500)
    backup = (step, net, optim)
    data_loader = get_data_loader(CONFIG["tf_data_dir"], batch_size * STEPS_PER_CALL)
    pstep = jax.device_put_replicated(jnp.array(step), DEVICES)
    for batch in data_loader:
        step += STEPS_PER_CALL
        pstep, net, optim, loss = multiple_train_step(pstep, net, optim, batch)
        losses.append(loss)

        if step % 100 == 0:
            loss = jnp.mean(sum(losses)).item() / len(losses)
            end = time.perf_counter()
            duration = end - start
            start = end
            sparsity = net.gru_pruner.compute_sparsity(step).item()
            print(
                "step  {:07d}  loss {:.8f}  LR {:.3e}  sparsity {:.2%}  {:.2f}s".format(
                    step, loss, optim[-1].learning_rate[0], sparsity, duration
                ),
                flush=True,
            )

        if step % 1000 == 0:
            if math.isfinite(loss):
                backup = (step, net, optim)
            else:
                step, net, optim = backup
                print("nan loss detected! Use backup checkpoint at step", step)

        if step % 10_000 == 0:
            net_, optim_ = jax.tree_map(lambda x: x[0], (net, optim))
            save_ckpt(step, net_, optim_, CONFIG["ckpt_dir"], MODEL_PREFIX)


if __name__ == "__main__":
    fire.Fire(train)
