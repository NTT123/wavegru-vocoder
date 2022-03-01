"""
Create tensorflow dataset from mel/mu files.
"""

import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm.cli import tqdm

from utils import load_config

parser = ArgumentParser(description="Prepare tf dataset for training")
parser.add_argument("ft_dir", type=Path, help="Path to feature directory")
config = load_config()
args = parser.parse_args()
hop_length = int(config["sample_rate"] * config["hop_length"] * 1e-3)

mu_files = sorted(args.ft_dir.glob("*.mu.npy"))
mel_files = [f.parent / f"{f.name[:-7]}.mel.npy" for f in mu_files]

frames_per_sequence = config["frames_per_sequence"]
num_pad_frames = config["num_pad_frames"]
assert num_pad_frames % 2 == 0
num_center_frames = frames_per_sequence - num_pad_frames
pad = num_pad_frames // 2
mel_min = config["mel_min"]


def data_generator():
    """
    yield mel, mu
    """
    data_files = list(zip(mu_files, mel_files))
    random.Random(42).shuffle(data_files)

    for mu_file, mel_file in tqdm(data_files):
        mu = np.load(mu_file)
        mel = np.load(mel_file)

        L = len(mu) // hop_length + 1
        mel = mel[: L]
        mu = mu[: L * hop_length]
        mu = np.pad(mu, [(1, 0)], mode="constant", constant_values=127)
        mel = np.pad(mel, [(pad, pad), (0, 0)], constant_values=np.log(mel_min))

        L = mel.shape[0]
        for i in range(0, L - frames_per_sequence, num_center_frames):
            j = i + frames_per_sequence
            l = i * hop_length
            r = (i + num_center_frames) * hop_length + 1
            x = mel[i:j]
            y = mu[l:r]
            yield x, y


mel_shape = (frames_per_sequence, config["mel_dim"])
mu_shape = (1 + num_center_frames * hop_length,)
output_signature = (
    tf.TensorSpec(shape=mel_shape, dtype=tf.float16, name="mel"),
    tf.TensorSpec(shape=mu_shape, dtype=tf.int32, name="mu"),
)
dataset = tf.data.Dataset.from_generator(
    data_generator, output_signature=output_signature
)
print("Dataset definition:", dataset)


def custom_shard_func(*a):
    """
    return random shard index
    """
    return tf.random.uniform((), minval=0, maxval=20, dtype=tf.dtypes.int64)


tf.data.experimental.save(
    dataset, config["tf_data_dir"], compression="GZIP", shard_func=custom_shard_func
)
