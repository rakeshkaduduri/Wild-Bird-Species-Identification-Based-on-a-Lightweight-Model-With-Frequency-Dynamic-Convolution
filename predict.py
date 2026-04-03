import torch
import librosa
import numpy as np
from PIL import Image
from torchvision import transforms

from model import BirdCNN

# -----------------------------------
# Bird classes (same order as training)
# -----------------------------------
classes = [
    'supsta1','tafpri1','tamdov1','thrnig1','trobou1',
    'varsun2','vibsta2','vilwea1','vimwea1','walsta1','wbgbir1'
]

# -----------------------------------
# Load trained model
# -----------------------------------
model = BirdCNN(num_classes=len(classes))
model.load_state_dict(torch.load("models/bird_model.pth", map_location="cpu"))
model.eval()

# -----------------------------------
# Image transformation
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# -----------------------------------
# Prediction function
# -----------------------------------
def predict_audio(audio_path):

    try:

        # Load audio
        y, sr = librosa.load(audio_path)

        # Create Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)

        mel_db = librosa.power_to_db(mel)

        # Normalize
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        mel_db = (mel_db * 255).astype(np.uint8)

        # Convert to image
        img = Image.fromarray(mel_db).convert("RGB")

        img = transform(img)

        img = img.unsqueeze(0)

        # Model prediction
        with torch.no_grad():

            output = model(img)

            probabilities = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(probabilities, 1)

        species = classes[predicted.item()]

        return species, confidence.item()

    except Exception as e:

        return "Error", 0