"""
FastAPI inference service for IT ticket classification.

This API loads the best fine-tuned Hugging Face model and exposes
simple endpoints for health checks and text classification.
"""

# Import Path to handle file paths across different operating systems.
from pathlib import Path

# Import torch to run model inference.
import torch

# Import FastAPI to create the web API.
from fastapi import FastAPI

# Import BaseModel to define request and response schemas.
from pydantic import BaseModel

# Import Hugging Face utilities to load the tokenizer and model.
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Create the FastAPI application instance.
app = FastAPI(
    title="IT Ticket Classification API",
    description="Inference API for classifying IT support tickets using a fine-tuned DistilBERT model.",
    version="1.0.0",
)


# Define the project root based on the location of this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the path to the best trained model.
MODEL_PATH = PROJECT_ROOT / "models" / "run_03"

# Define the maximum number of tokens used during inference.
MAX_LENGTH = 128

# Define the label mapping used by the trained model.
# Update this dictionary only if your label encoding changes.
ID_TO_LABEL = {
    0: "Hardware",
    1: "HR Support",
    2: "Network",
    3: "Storage",
    4: "Software",
}


# Define the expected input schema for prediction requests.
class PredictionRequest(BaseModel):
    # Text from the IT ticket that will be classified.
    text: str


# Define the output schema returned by the prediction endpoint.
class PredictionResponse(BaseModel):
    # Predicted category name.
    category: str

    # Model confidence score for the predicted category.
    confidence: float


# Load the tokenizer once when the application starts.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load the fine-tuned sequence classification model once when the application starts.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Move the model to CPU for portable inference.
model.to("cpu")

# Set the model to evaluation mode.
model.eval()


@app.get("/")
def root():
    """
    Root endpoint to confirm that the API is running.
    """

    # Return a simple welcome message.
    return {
        "message": "IT Ticket Classification API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint used to verify that the API and model are available.
    """

    # Return API and model status.
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict the IT ticket category from raw ticket text.
    """

    # Tokenize the input text using the saved tokenizer.
    inputs = tokenizer(
        request.text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Disable gradient calculation because this is inference only.
    with torch.no_grad():

        # Run the model and obtain raw logits.
        outputs = model(**inputs)

        # Convert logits into probabilities.
        probabilities = torch.softmax(outputs.logits, dim=1)

        # Select the class with the highest probability.
        predicted_class_id = int(torch.argmax(probabilities, dim=1).item())

        # Extract the confidence score for the predicted class.
        confidence = float(probabilities[0][predicted_class_id].item())

    # Convert the predicted class ID into a readable category label.
    predicted_category = ID_TO_LABEL.get(predicted_class_id, f"Unknown label {predicted_class_id}")

    # Return the prediction response.
    return PredictionResponse(
        category=predicted_category,
        confidence=round(confidence, 4),
    )
