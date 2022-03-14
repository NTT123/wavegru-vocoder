"""
Extract Embed + GRU + O1 + O2 weights and masks from a pretrained model

Usage:

    python extract_weight_and_mask.py --model model.ckpt --output gru_weight.npz

"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from utils import load_ckpt, load_config
from wavegru import WaveGRU

parser = ArgumentParser(description="Extract weights and masks")
parser.add_argument(
    "--model", type=Path, required=True, help="Path to model checkpoint"
)
parser.add_argument("--output", type=Path, required=True, help="Path to output file")
args = parser.parse_args()
CONFIG = load_config()
net = WaveGRU(
    mel_dim=CONFIG["mel_dim"],
    embed_dim=CONFIG["embed_dim"],
    rnn_dim=CONFIG["rnn_dim"],
    upsample_factors=CONFIG["upsample_factors"],
)
_, net, _ = load_ckpt(net, None, args.model)
data = {}
data["embed_weight"] = net.embed.weight
data["gru_xh_zr_weight"] = net.rnn.xh_zr_fc.weight
data["gru_xh_zr_mask"] = net.gru_pruner.xh_zr_fc_mask
data["gru_xh_zr_bias"] = net.rnn.xh_zr_fc.bias

data["gru_xh_h_weight"] = net.rnn.xh_h_fc.weight
data["gru_xh_h_mask"] = net.gru_pruner.xh_h_fc_mask
data["gru_xh_h_bias"] = net.rnn.xh_h_fc.bias

data["o1_weight"] = net.o1.weight
data["o1_mask"] = net.o1_pruner.mask
data["o1_bias"] = net.o1.bias
data["o2_weight"] = net.o2.weight
data["o2_mask"] = net.o2_pruner.mask
data["o2_bias"] = net.o2.bias
net = net.eval()
np.savez_compressed(args.output, **data)
print(f"Write output to file '{args.output}'")
