"""Configuration file for setting application parameters."""

import logging

# WebSocket settings
WEBSOCKET_URI = "ws://localhost:8080"

# Audio directories
# Make sure that in and out don't point to the same folder
AUDIO_IN_DIR = r"/home/anel/PycharmProjects/speech_recognition/data/in"
AUDIO_OUT_DIR = r"/home/anel/PycharmProjects/speech_recognition/data/out"


# TTS (Text-to-Speech) settings
PIPER_DIR = r"C:\Users\dervi\Desktop\piper"
# Voice model name for Piper
# If it's not inside the PIPER_DIR, provide the full absolute path
VOICE_NAME = "de_DE-thorsten-high.onnx"
# VOICE_NAME = "Thorsten-Voice_Hessisch_Piper_high-Oct2023.onnx"
# Directory where generated audio files are stored
GENERATE_AUDIO_DIR = r"/home/anel/PycharmProjects/speech_recognition/data/generated"

# ASR (Automatic Speech Recognition) settings
# Default: openai/whisper-large-v3-turbo
ASR_MODEL_NAME = "openai/whisper-large-v3-turbo"
ASR_LANGUAGE = "german"

# LLM (Large Language Model) settings
# Default: Qwen/Qwen2.5-0.5B-Instruct
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Logging settings available: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = logging.DEBUG

# Log file name and location
LOG_FILE = r"logs/app.log"
