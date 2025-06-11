# speech_recognition

Speech Recognition for Uni project

# How to Install

## Prerequisites

python 🐍 (Tested with 3.11 and 3.12)  
piper 🗣️  
FFmpeg 🔁

### 🛠️ Install piper

Follow the instructions here https://github.com/rhasspy/piper  
You can Download voices here https://huggingface.co/rhasspy/piper-voices/tree/main

### 🛠️ Install FFmpeg on Linux(Debian)

```
sudo apt update
sudo apt install ffmpeg
```

### 🛠️ Install FFmpeg on Windows

**With a package manager (Chocolatey)**

``choco install ffmpeg``

**If you don't have Chocolatey (Binary + PATH Setup)**

Download a static build from the official site:
👉 https://www.gyan.dev/ffmpeg/builds/
Choose:
ffmpeg-release-essentials.zip

Extract the zip to a location like:
``
C:\ffmpeg
``

Inside that folder, go to ``C:\ffmpeg\bin``, and copy the full path.

Add FFmpeg to your system PATH:

```
Open Start Menu → search for Environment Variables

Click "Edit the system environment variables"

In the dialog, click "Environment Variables..."

Under "System variables", find and select Path, then click Edit

Click New and paste:

C:\ffmpeg\bin

Click OK on all windows.
```

Open a new terminal and check if it works:

``ffmpeg -version``

After cloning the repository, go inside the project folder and create a virtual environment

*Windows*
<br>
``python -m venv venv``
<br>
``venv\Scripts\activate``
<br>
<br>
Once you have the venv ready and activated, your terminal should look like:
<br>
``(venv) C:\Users\you\...``
<br>
<br>
*Linux*
<br>
``python -m venv venv``
<br>
``venv/bin/activate``
<br>
<br>
Once you have the venv ready and activated, your terminal should look like:
<br>
``(venv) ~/...``
<br>
<br>
Now upgrade pip and install the required packages
<br>
``pip install --upgrade pip``
<br>
``pip install -r requirements.txt``
<br>
⚠️ Note: Only CPU version of PyTorch is in the requirements.txt.
If you have an nvidia GPU, please install the appropriate version manually.

Install PyTorch (Choose Your CUDA Version)

Go to the official PyTorch installation guide:
👉 https://pytorch.org/get-started/locally/

Then run the appropriate installation command.
For example,
If using CUDA 12.6:

``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126``

You should now be able to run
``pytest``
in the Root directory and it should run all tests.

# Starting the application

Once everything is installed, you can go into the speech_recognition/config.py and configure it how you want it.

To start it, you need to have the venv activated and execute

``python run.py``

from the root directory

If you do not have/want an external server there is a mock server that can be started with

``python -m tests.mock_server``

This is a basic server, you can use it to send messages to generate audio files for.
The server will take care of properly wrapping it you only need
to send the sentence you want the text-to-speech to generate.
The server will also receive the results of the speech recognition if you put an audio file the specified
in directory.
