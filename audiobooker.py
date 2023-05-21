import os
import time
import torch
from playsound import playsound
import webbrowser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

from utilities import *
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav, read as read_wav
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
import streamlit as st

checkpoint_file = "checkpoint.txt"


def convert_to_wav(sentences, start_point, progress_bar, existing_audio):
    start_time = time.time()
    SPEAKER = f"v2/en_speaker_3"
    preload_models()

    GEN_TEMP = 0.6
    silence = np.zeros(int(0.25*SAMPLE_RATE))

    pieces = []
    if np.any(existing_audio):
        pieces.append(existing_audio)

    line_count = len(sentences)
    print(f"total lines: {line_count}")
    for i, line in enumerate(sentences):
        if i < start_point: continue
        # seems like 12 seconds max. 200 char too much sometimes, going with 25 word chunks
        words = nltk.tokenize.word_tokenize(line)
        num_chunks = (len(words)+24)//25
        for j in range(num_chunks):
            start = j*25
            end = min(start+25, len(words))
            chunk = " ".join(words[start:end])
            print(chunk)
            audio_array = generate_audio(chunk, history_prompt=SPEAKER)
            pieces.append(audio_array)

        write_wav("output.wav", SAMPLE_RATE, np.concatenate(pieces))
        with open(checkpoint_file, "a") as f:
            # its hacky, but tracks progress with a textfile in case its interrupted
            f.write(f"{i}\n")
        progress_bar.progress((i+1)/line_count)
    os.remove(checkpoint_file) # remove on completion

    print("completed.")
    print("---%s seconds ---" % (time.time()-start_time))

st.set_page_config(page_title="pdf 2 MP3")
st.header("Convert your pdf to Audio")
pdf = st.file_uploader("Upload your PDF", type="pdf")
completed = False
    
file_path = "output.wav"
if os.path.exists(file_path):
    # load existing file, if it was interrupted will resume where left off
    sample_rate, existing_audio = read_wav(file_path)

else: existing_audio = None


if pdf is not None:
    sentences = read_pdf(pdf)
    start_point = get_start_point(pdf)
    progress_bar = st.progress(0)
    if not completed:
        convert_to_wav(sentences, start_point, progress_bar, existing_audio)
        completed = True

    if completed:
        st.write("completed, file: output.wav")


