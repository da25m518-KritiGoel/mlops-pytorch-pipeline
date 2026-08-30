import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import get_model


app = FastAPI(title="CIFAR-10 Classifier API")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/checkpoints/classifier_v1.pt",
)

model = None


class PredictionRequest(BaseModel):
    image: list[list[list[float]]]


def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        return False

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model = get_model(
        architecture=checkpoint.get("architecture", "resnet18"),
        num_classes=checkpoint.get("num_classes", 10),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        tensor = torch.tensor(
            request.image,
            dtype=torch.float32,
        )

        if tuple(tensor.shape) != (3, 32, 32):
            raise ValueError(
                "image must have shape [3, 32, 32]"
            )

        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(
                probabilities.argmax(dim=1).item()
            )
            probability = float(
                probabilities[0, predicted_class].item()
            )

        return {
            "class_id": predicted_class,
            "probability": probability,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


load_model()