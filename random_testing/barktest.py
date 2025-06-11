from transformers import BarkModel
from transformers import AutoProcessor
import torch
from IPython.display import Audio
import scipy
import time
import numpy
import datetime
datetime.time
time_before_model = time.time()
print(time_before_model)
model = BarkModel.from_pretrained('suno/bark-small')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

processor = AutoProcessor.from_pretrained('suno/bark')
voice_preset = 'v2/de_speaker_3'

time_after_model = time.time()
print("Loading Model took: ", time_after_model-time_before_model)
# prepare the inputs
text_prompt = "Herr Mustermann bitte heben sie das Smartphone auf und folgen sie mir ins Behandlungszimmer!"
inputs = processor(text_prompt, voice_preset=voice_preset)
time_after_processing = time.time()
print("Processing Input took: ", time_after_processing-time_after_model)
# generate speech
speech_output = model.generate(**inputs.to(device))

sampling_rate = model.generation_config.sample_rate
print('done')
time_finished = time.time()
print("Time to finish with startup: ", time_finished-time_before_model)
print("Time for generation: ", time_finished-time_after_model)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)
rng = numpy.random.default_rng()
rand = int(rng.random()*1000)
filename = "data/" + rand.__str__() + "_bark_out.wav"
scipy.io.wavfile.write(filename=filename, rate=sampling_rate, data=speech_output[0].cpu().numpy())

