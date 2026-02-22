import torch
from app.main import model

def test_model_output_shape():
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape[1] == 2