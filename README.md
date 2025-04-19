# speech_recognition

Speech Recognition for Uni project

# 🇬🇧 🇺🇸 – How to Install

Prerequisites

Works with Python 3.11, 3.12 

This project requires FFmpeg to be installed and available in your system's PATH.

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

After cloning the repository go inside the project folder and create a virtual environment

*Windows*
<br>
``python -m venv venv``
<br>
``venv\Scripts\activate``
<br>
<br>
Once you have the venv ready and activated your terminal should look like:
<br>
``(venv) C:\Users\you\...``
<br>
<br>
Now upgrade pip and install the required packages
<br>
``pip install --upgrade pip``
<br>
``pip install -r requirements.txt``
<br>
⚠️ Note: PyTorch is not included in requirements.txt. Please install the appropriate version for your system manually.

Install PyTorch (Choose Your CUDA Version)

Go to the official PyTorch installation guide:
👉 https://pytorch.org/get-started/locally/

Then run the appropriate install command. For example:
If using CUDA 12.6:

``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126``

If using CPU-only:

``pip install torch torchvision torchaudio``
