from fastapi import FastAPI
from gradio_client import Client
import os

app = FastAPI()

# Map your model IDs to your actual HF Space URLs
MODEL_MAP = {
    "1": ("E-motionAssistant/Space4", "/classify_english"),
    "2": ("E-motionAssistant/Space4", "/classify_sinhala"),
    "3": ("E-motionAssistant/Space4", "/classify_tamil"),
    "4": ("E-motionAssistant/Space5", "/detect_emotion"),

    
}

@app.get("/api/python/predict")
def predict(model_id: str, message: str):
    model_info = MODEL_MAP.get(model_id)

    if not model_info:
        return {"error": "Invalid Model ID"}
    
    space_path, api_endpoint = model_info
    
    try:
        # Connect to the specific HF Space
        client = Client(space_path)
        result = client.predict(message, api_name=api_endpoint)
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}
