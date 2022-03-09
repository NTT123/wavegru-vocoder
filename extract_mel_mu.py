"""
Extract mel-spectrogram features and mu-law encoded waveforms from wav files
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import jax
import librosa
import numpy as np
from tqdm.cli import tqdm

from dsp import MelFilter
from utils import load_config


def extract_mel_mu(
    mel_filter, wav_file, sample_rate, maxlen, window_length, hop_length
):
    """
    We use np.float16 to save memory.
    """
    y, _ = librosa.load(wav_file, sr=sample_rate, res_type="soxr_hq")
    scale = max(1, np.max(np.abs(y)))
    y = y / scale  # rescale to avoid overflow [-1, 1] interval
    num_frames = len(y) // hop_length + 1
    padded_y = np.pad(y, [(0, maxlen - len(y))])
    # scale by 0.5 to avoid overflow when doing preemphasis
    y = librosa.effects.preemphasis(y * 0.5, coef=0.86)
    mu = librosa.mu_compress(y, mu=255, quantize=True) + 127
    mel = mel_filter(padded_y[None])[0].astype(np.float16)
    return mel[:num_frames], mu


parser = ArgumentParser(description="Preprocess audio data")
parser.add_argument("wav_dir", type=Path, help="Input data directory")
parser.add_argument("out_dir", type=Path, help="Output data directory")
args = parser.parse_args()
config = load_config()
wav_files = sorted(args.wav_dir.glob("*.wav"))
args.out_dir.mkdir(parents=True, exist_ok=True)

sr = config["sample_rate"]
window_length = int(sr * config["window_length"] / 1000)
hop_length = int(sr * config["hop_length"] / 1000)
assert np.prod(config["upsample_factors"]) == hop_length

mel_filter = MelFilter(
    sr,
    config["n_fft"],
    window_length,
    hop_length,
    config["mel_dim"],
    0,
    sr // 2,
    config["mel_min"],
)
mel_filter = jax.jit(mel_filter)

sorted_files = sorted(wav_files, key=os.path.getsize)
y, _ = librosa.load(sorted_files[-1], sr=sr)
maxlen = len(y)

for wav_file in tqdm(wav_files):
    mel, mu = extract_mel_mu(
        mel_filter, wav_file, config["sample_rate"], maxlen, window_length, hop_length
    )
    mel_file = args.out_dir / (wav_file.stem + ".mel")
    mu_file = mel_file.with_suffix(".mu")
    np.save(mel_file, mel)
    np.save(mu_file, mu)


print(f"Preprocessed data is located at {args.out_dir}")
print()
print(f"Run 'python tf_data.py {args.out_dir}' to create tf dataset")
