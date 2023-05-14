from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

import torchaudio
import numpy as np
import torch
import os

model = load_codec_model(use_gpu=True)
device = 'cuda'
file_dir = "audiosamples\wav"
samples = {
    "audio2_00000103.wav": "for you sir, always",
    "audio2_00000153.wav": "shall I store this on the Stark industries circle database?",
    "audio2_00000248.wav": "working on a secret project are we sir?",
    "audio2_00000425.wav": "I have indeed been uploaded sir, we're online and ready",
    "audio2_00001345.wav": "I have no record of an invitation sir",
    "audio2_00001669.wav": "commencing automated assembly, estimated completion time is five hours"
}


for audio, text in samples.items():
    path = os.path.join(file_dir, audio)
    wav, sr = torchaudio.load(path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
    
    seconds = wav.shape[-1]/model.sample_rate
    semantic_tokens = generate_text_semantic(text, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7)
    codes = codes.cpu().numpy()

    voice_name = "jarvis"
    np.savez(f"{voice_name}.npz", coarse_prompt=codes[:2,:], semantic_prompt=semantic_tokens)
    quit()