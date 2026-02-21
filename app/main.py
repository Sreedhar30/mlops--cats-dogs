import io
import torch
import logging
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from src.model import SimpleCNN

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# ---------------------------
# Load Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/model.pt", map_location=device))
model.eval()

# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

# ---------------------------
# Health Endpoint
# ---------------------------
@app.get("/health")
def health():
    return {"status": "OK"}

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    label = "Cat" if predicted_class == 0 else "Dog"

    logging.info(f"Prediction made: {label}")

    return {
        "prediction": label,
        "probabilities": probabilities.cpu().numpy().tolist()
    }