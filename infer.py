import os
import time
import argparse

import torch
import torchaudio

import CLAP
from generator import Generator
from HiFiGAN.inference import get_vocoder

def format_time(seconds):
    seconds = int(seconds)
    
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    result = []
    if days > 0:
        result.append(f"{days}d")
    if hours > 0:
        result.append(f"{hours}h")
    if minutes > 0:
        result.append(f"{minutes}m")
    if secs > 0:
        result.append(f"{secs}s")
    
    return " ".join(result) if result else "0s"

def load_model_for_inference(ckpt_path):
    model_dict = {}
    
    model_dict['clap_encoder'] = CLAP.CLAP_Module(amodel='HTSAT-base', tmodel='roberta').eval()
    model_dict['clap_encoder'].load_ckpt(ckpt_path, 'music_speech_audioset_epoch_15_esc_89.98.pt')
    
    model_dict['generator'] = Generator()
    ckpt_state = torch.load(os.path.join(ckpt_path, 'generator.pt'), weights_only=False, map_location='cpu')
    model_dict['generator'].load_state_dict(ckpt_state)
    
    model_dict['vocoder'] = get_vocoder(sr=16000, ckpt_path=ckpt_path).eval()
    
    model_dict['clap_tester'] = CLAP.CLAP_Module(amodel='HTSAT-tiny', tmodel='roberta').eval()
    model_dict['clap_tester'].load_ckpt(ckpt_path, '630k-audioset-best.pt')
    
    for model_name in model_dict.keys():
        model_dict[model_name].eval()

    return model_dict

def get_text_embedding(text, model_dict):
        
    with torch.no_grad():
        sentence_embedding, word_embedding, sequence_lengths = model_dict['clap_encoder'].get_text_embedding(text)

    return sentence_embedding.detach(), word_embedding.detach(), sequence_lengths

def generate_audio(model_dict, text, device="cuda:0", gen_per_text=1, save_path='inference_results'):
    model_dict['generator'] = model_dict['generator'].to(device)
    model_dict['clap_encoder'] = model_dict['clap_encoder'].to(device)
    model_dict['vocoder'] = model_dict['vocoder'].to(device)
    model_dict['clap_tester'] = model_dict['clap_tester'].to(device)
    
    origin_text = text
    if text.endswith('.'):
        text = text[:-1]
    if text[0].islower():
        text = text[0].upper() + text[1:]
        
    print(f'Text : {text}')
    print(f'Try to generate per text : {gen_per_text}')
    print()
    
    print('Generate Audio...')
    text = [text] * gen_per_text
    start_time = time.time()
    with torch.no_grad():
        sentence_embedding, word_embedding, sequence_lengths = get_text_embedding(text, model_dict)

        noise = torch.randn((gen_per_text, 128)).to(device)
        fake_mel = model_dict['generator'](noise, sentence_embedding, word_embedding, sequence_lengths)
        fake_sound = model_dict['vocoder'](fake_mel.squeeze())
        
        if len(fake_sound.shape) == 3:
            fake_sound = fake_sound.squeeze(1)
        
        fake_similarity = model_dict['clap_tester'].get_clap_score(text, fake_sound, sr=16000)
        best_clap_score, best_clap_index = torch.max(fake_similarity, dim=0)

    print(f'Time for Generating Audio : {format_time(time.time() - start_time)}')
    print()
    
    if gen_per_text > 1:
        print(f'Scores of generated audios : {[round(x, 2) for x in fake_similarity.tolist()]}')
        print(f'Best CLAP Score : {round(best_clap_score.item(), 2)}')
    else:
        print(f'CLAP Score : {round(best_clap_score.item(), 2)}')
    print()
    
    best_fake_sound = fake_sound[best_clap_index].unsqueeze(0).detach().cpu()

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{origin_text.replace(' ', '_')}.wav")
    torchaudio.save(file_path, best_fake_sound, 16000)
    print(f'Save best audio to {file_path}')
    print()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="pretrained_models"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0
    )
    parser.add_argument(
        "--text",
        type=str,
        default="A bird is chirping in a quiet place."
    )
    parser.add_argument(
        "--gen_per_text",
        type=int,
        default=1
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="inference_results"
    )
    args = parser.parse_args()
    
    assert os.path.exists(os.path.join(args.ckpt_path, 'generator.pt')), f"Download AudioGAN Generator in \"{args.ckpt_path}\"."
    assert os.path.exists(os.path.join(args.ckpt_path, 'hifigan_16k_64bins.json')), f"Download HiFi-GAN Vocoder in \"{args.ckpt_path}\"."
    assert os.path.exists(os.path.join(args.ckpt_path, 'hifigan_16k_64bins.ckpt')), f"Download HiFi-GAN Vocoder in \"{args.ckpt_path}\"."

    model_dict = load_model_for_inference(args.ckpt_path)

    generate_audio(
        model_dict, 
        text=args.text, 
        device=f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu",
        gen_per_text=args.gen_per_text,
        save_path=args.save_path,
    )

if __name__ == "__main__":
    main()