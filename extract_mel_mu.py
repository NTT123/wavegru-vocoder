"""
Extract mel-spectrogram features and mu-law encoded waveforms from wav files
"""

from argparse import ArgumentParser
from pathlib import Path

import librosa
import numpy as np
from tqdm.cli import tqdm
from utils import load_config


def extract_mel_mu(
    wav_file, n_fft, window_length, hop_length, mel_dim, sample_rate, mel_min
):
    """
    We use np.float16 to save memory.
    """
    y, rate = librosa.load(wav_file, sr=sample_rate)
    y = librosa.effects.preemphasis(y, coef=0.86)
    y = np.clip(y, a_min=-1.0, a_max=1.0)
    mu = librosa.mu_compress(y, mu=255, quantize=True) + 127

    hop = int(hop_length * rate / 1000)
    window = int(window_length * rate / 1000)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=rate,
        n_mels=mel_dim,
        n_fft=n_fft,
        hop_length=hop,
        win_length=window,
        window="hann",
        center=False,
        pad_mode="reflect",
        power=2.0,
        fmin=0,
        fmax=rate // 2,
    )
    mel = np.log(mel + mel_min)
    mel = mel.astype(np.float16).T
    return mel, mu


parser = ArgumentParser(description="Preprocess audio data")
parser.add_argument("wav_dir", type=Path, help="Input data directory")
parser.add_argument("out_dir", type=Path, help="Output data directory")
args = parser.parse_args()
config = load_config()
wav_files = sorted(args.wav_dir.glob("*.wav"))
args.out_dir.mkdir(parents=True, exist_ok=True)

for wav_file in tqdm(wav_files):
    mel, mu = extract_mel_mu(
        wav_file,
        config["n_fft"],
        config["window_length"],
        config["hop_length"],
        config["mel_dim"],
        config["sample_rate"],
        config["mel_min"],
    )
    mel_file = args.out_dir / (wav_file.stem + ".mel")
    mu_file = mel_file.with_suffix(".mu")
    np.save(mel_file, mel)
    np.save(mu_file, mu)


print(f"Preprocessed data is located at {args.out_dir}")
print()
print(f"Run 'python tf_data.py {args.out_dir}' to create tf dataset")
