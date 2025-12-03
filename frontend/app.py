# frontend/app.py
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import os

# Make sure your models folder is accessible
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "models"))
from cnn_model import CNNEmotionClassifier  # your CNN class

# Flask app
app = Flask(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CNN model
# model = CNNEmotionClassifier(num_classes=8)
# model_path = os.path.join(ROOT, "models", "best_cnn_model.pt")
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)
# model.eval()
model = CNNEmotionClassifier(num_classes=8)
model_path = os.path.join(ROOT, "models", "best_cnn_model.pt")  # path to your trained weights
state_dict = torch.load(model_path, map_location=device)  # load weights
model.load_state_dict(state_dict)  # apply weights
model.to(device)
model.eval()

# Labels
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# # Preprocessing function
# def preprocess_audio(file_path, sr=48000, n_mels=40, n_fft=1024, hop_length=256, fixed_length=400):
#     y, _ = librosa.load(file_path, sr=sr)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
#     S_db = librosa.power_to_db(S, ref=np.max)
    
#     # pad or truncate to fixed_length
#     if S_db.shape[1] < fixed_length:
#         pad_width = fixed_length - S_db.shape[1]
#         S_db = np.pad(S_db, ((0,0),(0,pad_width)), mode='constant')
#     else:
#         S_db = S_db[:, :fixed_length]

#     # Convert to tensor and add batch & channel dims: (1,1,40,400)
#     return torch.tensor(S_db).unsqueeze(0).unsqueeze(0).float()
def preprocess_audio(file_path, sr=48000, n_mfcc=40, fixed_length=400):
    y, _ = librosa.load(file_path, sr=sr)
    
    # extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # pad or truncate time dimension
    if mfcc.shape[1] < fixed_length:
        pad_width = fixed_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :fixed_length]

    # normalize per sample
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)

    # add channel dimension for CNN: (1, n_mfcc, time)
    mfcc = np.expand_dims(mfcc, axis=0)

    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]

            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                file.save(tmp.name)
                x = preprocess_audio(tmp.name).to(device)

            with torch.no_grad():
                logits = model(x)
                pred_idx = torch.argmax(logits, dim=1).item()
                result = LABELS[pred_idx]

    return render_template("index.html", result=result)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
