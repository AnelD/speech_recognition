import time

from pywhispercpp.model import Model

#model = Model('large-v3-turbo-q5_0', n_threads=12)
#segments = model.transcribe('data/melissa_clean.wav', language='de')
#print(segments)
#for segment in segments:
 #   print(segment.text)


def timed_transcription(model_path, audio_path):
    t0 = time.time()
    model = Model(model_path, language='de', n_threads=12)
    t1 = time.time()
    print(f"Model loaded in {t1 - t0:.2f} seconds.")

    segments = model.transcribe(audio_path)
    t2 = time.time()
    print(f"Audio transcribed in {t2 - t1:.2f} seconds.")
    print(f"Total time: {t2 - t0:.2f} seconds.\n")

    for segment in segments:
        print(segment.text)

    return segments


if __name__ == '__main__':
    timed_transcription('medium-q5_0', '../../testggml/data/melissa_clean.wav')