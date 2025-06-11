import torch
from TTS.api import TTS
import numpy

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

rng = numpy.random.default_rng()
rand = int(rng.random()*1000)
filename = "data/" + rand.__str__() + "_bark_out.wav"