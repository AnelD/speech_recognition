"""Configuration file for setting application parameters."""

import logging

# WebSocket settings
WEBSOCKET_URI = "ws://localhost:8080"

# Audio directories
DATA_AUDIO_DIR = "../data/in"
COMMAND_AUDIO_DIR = "data/commands"

# TTS (Text-to-Speech) settings
PIPER_DIR = "/home/anel/Desktop/piper"

# Voice model name for Piper
# If it's not inside the PIPER_DIR, provide the full absolute path
VOICE_NAME = "de_DE-thorsten-high.onnx"
# VOICE_NAME = "Thorsten-Voice_Hessisch_Piper_high-Oct2023.onnx"

# Directory where generated audio files are stored
GENERATE_AUDIO_DIR = "/data/generated"

# ASR (Automatic Speech Recognition) settings
ASR_MODEL_NAME = "openai/whisper-large-v3-turbo"  # default Whisper model
ASR_LANGUAGE = "german"

# LLM (Large Language Model) settings
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # default LLM model

# Logging settings available: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = logging.INFO

# Log file name and location
LOG_FILE = "logs/app.log"
