from fastapi import FastAPI, UploadFile, File
import torch
from src.model import SimpleCNN
from PIL import Image
import io
from torchvision import transforms
import os

app = FastAPI()

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)

    return {
        "cat_probability": float(probs[0][0]),
        "dog_probability": float(probs[0][1])
    }
