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


import nltk

start_time = time.time()
# download and load all models
preload_models()

# generate audio from text
text_prompt = """
This is a test to see the different voices that bark has.
"""
# speaker 0-9
for i in range(10):
    SPEAKER = f"v2/en_speaker_{i}"
    audio_array = generate_audio(text_prompt, history_prompt=SPEAKER)

    # save audio to disk
    write_wav(f"bark_generation{i}.wav", SAMPLE_RATE, audio_array)
    time.sleep(1)
    print(f"Speaker: {SPEAKER}")
    playsound(f"bark_generation{i}.wav")

"""
voice 0 - okay
voice 1 - awful
voice 2 - okay-bad
voice 3 - english accent
voice 4 - sound quality feels bad
voice 5 - sound quality feels bad
voice 6 - decent
voice 7 - pretty good
voice 8 - okay
voice 9 - pam from the office?
"""
#print("---%s seconds ---" % (time.time()-start_time))
  

print("completed.")
