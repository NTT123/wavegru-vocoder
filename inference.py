from argparse import ArgumentParser
from pathlib import Path

import jax
import librosa
import numpy as np
import pax
from scipy.io import wavfile

from utils import load_ckpt, load_config
from wavegru import WaveGRU

parser = ArgumentParser(description="Mel to waveform")
parser.add_argument(
    "--model", type=Path, required=True, help="Path to model checkpoint"
)
parser.add_argument("--mel", type=Path, required=True, help="Path to mel file")
parser.add_argument(
    "--output", type=Path, required=True, help="Path to output wav file"
)
args = parser.parse_args()
CONFIG = load_config()
net = WaveGRU(
    mel_dim=CONFIG["mel_dim"],
    embed_dim=CONFIG["embed_dim"],
    rnn_dim=CONFIG["rnn_dim"],
    upsample_factors=CONFIG["upsample_factors"],
)
_, net, _ = load_ckpt(net, None, args.model)
net = net.eval()
net = jax.device_put(net)
mel = np.load(args.mel).astype(np.float32)
if len(mel.shape) == 2:
    mel = mel[None]
pad = CONFIG["num_pad_frames"] // 2
mel = np.pad(
    mel,
    [(0, 0), (pad, pad), (0, 0)],
    constant_values=np.log(CONFIG["mel_min"]),
)
wav = pax.pure(lambda net, mel: net.inference(mel))(net, mel)
wav = jax.device_get(wav)
wav = librosa.mu_expand(wav - 127, mu=255)
# wav = librosa.effects.deemphasis(wav, coef=0.86)
wav = wav / np.max(np.abs(wav))
wavfile.write(str(args.output), CONFIG["sample_rate"], wav)

print(f"Write output to file '{args.output}'")
