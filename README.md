# speech_recognition
Speech Recognition for Uni project

# 🇬🇧 🇺🇸 – How to Install


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
