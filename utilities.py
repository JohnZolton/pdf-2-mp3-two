

import os
import time
import torch
from playsound import playsound


from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

import numpy as np

from PyPDF2 import PdfReader
from tqdm import tqdm
import nltk
import wave

from PyPDF2 import PdfReader
from tqdm import tqdm
import nltk
import wave

checkpoint_file = "checkpoint.txt"

# returns int for starting sentence
def get_start_point(file_path):
    # Look for the latest saved checkpoint
    latest_checkpoint = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            for line in f:
                latest_checkpoint += 1
        print("Resuming from checkpoint:", latest_checkpoint)
        
    return latest_checkpoint

#returns array of sentences 
def read_pdf(input_path):
    reader = PdfReader(input_path)
    
    # controls content read on page
    def visitor_body(text, cm, tm, fontDict, fontSize):
        y = tm[5]
        # adjust height range to ignore headers/footers
        #print(f"{y}: {text}")
        if y < 800:
            parts.append(text)
    
    parts = []
    for i in tqdm(range(len(reader.pages))):
        text = reader.pages[i].extract_text(visitor_text=visitor_body)
    text = "".join(parts).replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(text)
    return sentences

def generate_audio_high():
        # semantic tokens method was 2x slower than using generate audio, but allegedly has less hallucinations
        semantic_tokens = generate_text_semantic(
            line,
            history_prompt = SPEAKER,
            temp = GEN_TEMP,
            min_eos_p = 0.05,
        )
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER,)
        return audio_array