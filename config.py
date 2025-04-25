WEBSOCKET_URI = "ws://localhost:8080"

# Directory where audio files with patient data are uploaded
DATA_AUDIO_DIR = ""

# Directory where audio files with voice commands are uploaded
COMMAND_AUDIO_DIR = ""


# Variables for TTS
# Directory where your downloaded piper.exe is
PIPER_DIR = "/home/anel/Desktop/piper"

# The Name of the voice model you want to use for tts
# If it's not inside the piper_dir give the full path
VOICE_NAME = "de_DE-thorsten-high.onnx"
# VOICE_NAME = "Thorsten-Voice_Hessisch_Piper_high-Oct2023.onnx"

# Where the generated audio will be saved to,
# for relative paths keep in mind that it starts in piper_dir
GENERATE_AUDIO_DIR = "/home/anel/PycharmProjects/speech_recognition/data/generated"
