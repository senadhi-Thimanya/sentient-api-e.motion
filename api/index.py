from fastapi import FastAPI
from gradio_client import Client
import os

app = FastAPI()

# Map your model IDs to your actual HF Space URLs
MODEL_MAP = {
    "1": "... ",
    "2": "... ",
    "3": "... ",
    "4": "... ",
    "5": "... ",
    
}

@app.get("/api/python/predict")
def predict(model_id: str, message: str):
    space_path = MODEL_MAP.get(model_id)
    if not space_path:
        return {"error": "Invalid Model ID"}
    
    try:
        # Connect to the specific HF Space
        client = Client(space_path)
        result = client.predict(message, api_name="/predict")
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}
