import os
import time
import torch
from playsound import playsound

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

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

start_time = time.time()

SPEAKER = f"v2/en_speaker_3"
preload_models()

GEN_TEMP = 0.6
silence = np.zeros(int(0.25*SAMPLE_RATE))


checkpoint_dir = "checkpoints"
checkpoint_interval = 2
checkpoint_file = "checkpoint.txt"

# Create the checkpoint directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Look for the latest saved checkpoint
latest_checkpoint = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        latest_checkpoint = int(f.read())
    print("Resuming from checkpoint:", latest_checkpoint)

pieces = []
sentences = read_pdf("The Crypto Anarchist Manifesto.pdf")
#sentences = ["And just as a seemingly minor invention like barbed wire made possible the fencing-off of vast ranches and farms, thus altering forever the concepts of land and property rights in the frontier West, so too will the seemingly minor discovery out of an arcane branch of mathematics come to be the wire clippers which dismantle the barbed wire around intellectual property."]

print(len(sentences))
for i, line in enumerate(sentences):
    if i < latest_checkpoint*checkpoint_interval: continue
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
    # semantic tokens method was 2x slower than using generate audio, but allegedly has less hallucinations

    '''semantic_tokens = generate_text_semantic(
        line,
        history_prompt = SPEAKER,
        temp = GEN_TEMP,
        min_eos_p = 0.05,
    )
    audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER,)'''
    #pieces += [audio_array]
    # saving after each line saves progress, its inherently slow so idk if saving periodically is better
    i += 1
    #if i % 
    write_wav("current.wav", SAMPLE_RATE, np.concatenate(pieces))
    if i % checkpoint_interval == 0: 
        checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{i + 1}.txt")

        write_wav(f"{checkpoint_filename}.wav", SAMPLE_RATE, np.concatenate(pieces))


print("completed.")
print("---%s seconds ---" % (time.time()-start_time))
