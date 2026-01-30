import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from transformers import RobertaModel
from .htsat import create_htsat_model

BASE_DIR = Path(__file__).resolve().parent

class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


# Audio Config Class
@dataclass
class CLAPAudioCfp:
    model_type: str = "PANN"
    model_name: str = "Cnn14"
    sample_rate: int = 48000
    # Param
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 1024
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000


@dataclass
class CLAPTextCfg:
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int
    model_type: str


class CLAP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        audio_cfg: CLAPAudioCfp,
        text_cfg: CLAPTextCfg,
    ):
        super().__init__()

        audio_cfg = CLAPAudioCfp(**audio_cfg)
        text_cfg = CLAPTextCfg(**text_cfg)
        self.context_length = text_cfg.context_length

        mlp_act_layer = nn.ReLU()

        # audio branch
        self.audio_branch = create_htsat_model(audio_cfg)

        # audio branch parameters
        self.audio_transform = MLPLayers(units=[512,
                                                512,
                                                512], dropout=0.1)

        self.audio_projection = nn.Sequential(
                nn.Linear(embed_dim, 512),
                mlp_act_layer,
                nn.Linear(512, 512)
            )
        
        # text branch
        self.text_branch = RobertaModel.from_pretrained('roberta-base')
            
        # text branch parameters
        self.text_transform = MLPLayers(units=[512,
                                               512,
                                               512], dropout=0.1)
        self.text_projection = nn.Sequential(
            nn.Linear(768, 512),
            mlp_act_layer,
            nn.Linear(512, 512)
        )

        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.init_text_branch_parameters()

    def init_text_branch_parameters(self):
        nn.init.constant_(self.logit_scale_a, np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale_t, np.log(1 / 0.07))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def get_word_embedding(self, data):
        device = next(self.parameters()).device
        for k in data:
            data[k] = data[k].to(device)
        
        word_embeds = self.text_branch.embeddings.word_embeddings(
            data['input_ids'].to(device=device, non_blocking=True)
        )

        return word_embeds

    def get_text_embedding(self, data, normalize=False):
        
        device = next(self.parameters()).device
        for k in data:
            data[k] = data[k].to(device)
            
        x = self.text_branch(
            input_ids=data["input_ids"].to(device=device, non_blocking=True),
            attention_mask=data["attention_mask"].to(
                device=device, non_blocking=True
            ),
        )["pooler_output"]
        text_embeds = self.text_projection(x)
        
        if normalize:
            text_embeds = F.normalize(text_embeds, dim=-1)
        
        return text_embeds

    def get_audio_embedding(self, data, normalize=False):
        
        device = next(self.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        
        audio_embeds = self.audio_branch(input_dict, mixup_lambda=None, device=device)["embedding"]
        audio_embeds = self.audio_projection(audio_embeds)
        
        if normalize:
            audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

