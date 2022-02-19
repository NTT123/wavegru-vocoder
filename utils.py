"""
Utility functions
"""
import yaml
from pathlib import Path
import pickle


def load_config():
    """
    Load project configurations
    """
    with open("./config.yaml", "r") as f:
        return yaml.safe_load(f)


def save_ckpt(step, net, optim, ckpt_dir, model_prefix):
    """
    save checkpoint to disk
    """
    out_file = Path(ckpt_dir) / f"{model_prefix}_{step:07d}.ckpt"
    dic = {
        "step": step,
        "net_state_dict": net.state_dict(),
        "optim_state_dict": optim.state_dict(),
    }
    with open(out_file, "wb") as f:
        pickle.dump(dic, f)


def load_ckpt(net, optim, ckpt_file):
    """
    load training checkpoint from file
    """
    with open(ckpt_file, "rb") as f:
        dic = pickle.load(f)

    net = net.load_state_dict(dic["net_state_dict"])
    optim = optim.load_state_dict(dic["optim_state_dict"])
    return dic["step"], net, optim
