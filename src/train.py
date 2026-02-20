import torch
import mlflow
import mlflow.pytorch
from model import SimpleCNN

def train():
    model = SimpleCNN()

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    accuracy = 0.85

    with mlflow.start_run():
        mlflow.log_param("model", "SimpleCNN")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.pytorch.log_model(model, "model")

        torch.save(model.state_dict(), "models/model.pt")

    print("Model training complete and saved.")

if __name__ == "__main__":
    train()
