import os
import json
import torch

from .hifigan_model import AttrDict, Generator

def torch_version_orig_mod_remove(state_dict):
    new_state_dict = {}
    new_state_dict["generator"] = {}
    for key in state_dict["generator"].keys():
        if "_orig_mod." in key:
            new_state_dict["generator"][key.replace("_orig_mod.", "")] = state_dict[
                "generator"
            ][key]
        else:
            new_state_dict["generator"][key] = state_dict["generator"][key]
    return new_state_dict

def get_vocoder(sr, ckpt_path):

    with open(os.path.join(ckpt_path, "hifigan_16k_64bins.json"), "r") as f:
        config = json.load(f)
    config = AttrDict(config)
    vocoder = Generator(config)

    ckpt = torch.load(os.path.join(ckpt_path, "hifigan_16k_64bins.ckpt"), map_location='cpu')
    ckpt = torch_version_orig_mod_remove(ckpt)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    
    return vocoder
