"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: https://arxiv.org/abs/2211.06687
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""
import os
import json
import torch
import librosa
import torchaudio
import transformers
import numpy as np
from pathlib import Path
from packaging import version

from .data import get_audio_features
from .data import int16_to_float32, float32_to_int16
from .clap_model import CLAP

from transformers import RobertaTokenizer
import wget

BASE_DIR = Path(__file__).resolve().parent

class CLAP_Module(torch.nn.Module):
    def __init__(self, amodel='HTSAT-tiny', tmodel='roberta') -> None:
        super(CLAP_Module, self).__init__()
        
        config_path = os.path.join(BASE_DIR, 'model_configs', f'{amodel}.json')
        with open(config_path, "r") as f:
            model_cfg = json.load(f)
        
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")
                
        model_cfg["text_cfg"]["model_type"] = tmodel
        model = CLAP(**model_cfg)
        
        self.model = model
        self.model_cfg = model_cfg

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return result

    def load_ckpt(self, ckpt_folder_path, ckpt_name):
        ckpt_path = os.path.join(ckpt_folder_path, ckpt_name)
        
        if os.path.exists(ckpt_path):
            print(f'Load checkpoint from {ckpt_path}')
        else:
            download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
            print(f'Download checkpoint from {download_link + ckpt_name}.')
            ckpt_path = wget.download(download_link + ckpt_name, ckpt_folder_path)
            print('Download completed!')
        print()
        
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
             
        if version.parse(transformers.__version__) >= version.parse("4.31.0"): 
            del state_dict["text_branch.embeddings.position_ids"]
    
        self.model.load_state_dict(state_dict)
    
    def get_audio_embedding(self, x, sr=16000, normalize=False, use_tensor=True):
        self.model.eval()
        if isinstance(x, str):
            x = [x]

        audio_input = []
        for audio_waveform in x:
            
            if isinstance(audio_waveform, str):
                # load the waveform of the shape (T,), should resample to 48000
                audio_waveform, _ = librosa.load(audio_waveform, sr=48000)
            elif sr != 48000:
                audio_waveform = torchaudio.functional.resample(audio_waveform, orig_freq=sr, new_freq=48000)                
                
            if isinstance(audio_waveform, torch.Tensor):
                audio_waveform = audio_waveform.numpy()
                
            # quantize
            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()

            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000, 
                data_truncating='rand_trunc', 
                data_filling='repeatpad',
                audio_cfg=self.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            
            audio_input.append(temp_dict)
            
        audio_embed = self.model.get_audio_embedding(audio_input, normalize)
        
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu().numpy()

        return audio_embed

    def get_text_embedding(self, x, normalize=False, use_tensor=True):
        self.model.eval()
        if isinstance(x, str):
            x = [x]
            
        token_data = self.tokenizer(x)
        sequence_lengths = (torch.ne(token_data['attention_mask'], 0).sum(-1) - 1)
        setence_embeds = self.model.get_text_embedding(token_data, normalize)
        word_embeds = self.model.get_word_embedding(token_data)
        
        if not use_tensor:
            setence_embeds = setence_embeds.detach().cpu().numpy()
            word_embeds = word_embeds.detach().cpu().numpy()
            
        return setence_embeds, word_embeds, sequence_lengths
        
    def get_clap_score(self, text, audio, sr=16000):
        setence_embeds, word_embeds, sequence_lengths = self.get_text_embedding(text, normalize=True)
        audio_embeds = self.get_audio_embedding(audio, sr=16000, normalize=True)
        
        clap_score = torch.nn.functional.cosine_similarity(setence_embeds, audio_embeds, dim=-1)

        return clap_score