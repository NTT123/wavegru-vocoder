"""
Mel to waveform / upsample features

Usage:

    poetry run python inference.py --model model.ckpt --mel mel.npy --output ft.npz --no-gru

"""
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
parser.add_argument("--output", type=Path, required=True, help="Path to output file")
parser.add_argument(
    "--no-gru",
    default=False,
    action="store_true",
    help="Return the upsample feature only",
)
args = parser.parse_args()
CONFIG = load_config()
net = WaveGRU(
    mel_dim=CONFIG["mel_dim"],
    rnn_dim=CONFIG["rnn_dim"],
    upsample_factors=CONFIG["upsample_factors"],
)
step, net, _ = load_ckpt(net, None, args.model)
print("Loaded checkpoint step", step)
net = net.eval()
net = jax.device_put(net)
mel = np.load(args.mel).astype(np.float32)
if len(mel.shape) == 2:
    mel = mel[None]
pad = CONFIG["num_pad_frames"] // 2 + 2
mel = np.pad(
    mel,
    [(0, 0), (pad, pad), (0, 0)],
    constant_values=np.log(CONFIG["mel_min"]),
)
x = pax.pure(lambda net, mel: net.inference(mel, no_gru=args.no_gru))(net, mel)
x = jax.device_get(x)

if args.no_gru:
    np.savez_compressed(args.output, mel=x)
else:
    wav = librosa.mu_expand(x - 127, mu=255)
    wav = librosa.effects.deemphasis(wav, coef=0.86)
    wav = wav * 2.0
    wav = wav / max(1.0, np.max(np.abs(wav)))
    wav = wav * 2 ** 15
    wav = np.clip(wav, a_min=-(2 ** 15), a_max=(2 ** 15) - 1)
    wav = wav.astype(np.int16)

    wavfile.write(str(args.output), CONFIG["sample_rate"], wav)

print(f"Write output to file '{args.output}'")
