import numpy as np
import cv2
import torch
from app.main import transform

def test_preprocessing_pipeline():
    # Create dummy RGB image
    dummy = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    resized = cv2.resize(dummy, (224, 224))
    tensor_img = transform(resized).unsqueeze(0)

    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (1, 3, 224, 224)