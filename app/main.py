"""
FastAPI inference service for IT ticket classification using a fine-tuned DistilBERT model.
"""

from pathlib import Path
from typing import Dict

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path("models/run_03")
MAX_LENGTH = 128

ID_TO_LABEL = {
    0: "Network",
    1: "Hardware",
    2: "Software",
    3: "Access",
    4: "Security",
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

app = FastAPI(
    title="IT Ticket Classification API",
    description="Inference API for classifying IT support tickets using a fine-tuned DistilBERT model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    category: str
    confidence: float


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "IT Ticket Classification API is running."}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    inputs = tokenizer(
        request.text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # DistilBERT does not use token_type_ids.
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    predicted_class = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_class].item())

    category = ID_TO_LABEL.get(predicted_class, "Unknown")

    return PredictionResponse(
        category=category,
        confidence=round(confidence, 4),
    )
