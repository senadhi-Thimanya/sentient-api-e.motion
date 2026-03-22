from flask import Flask, request, jsonify
from gradio_client import Client
import os
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


HF_TOKEN = os.getenv("HF_TOKEN")

# --- 1. CONFIGURATION: YOUR HUGGING FACE SPACES ---
SPACE_TEXT_EMO  = "E-motionAssistant/Space4"
SPACE_AUDIO_EMO = "E-motionAssistant/Space5"
SPACE_LLM       = "E-motionAssistant/TherapyEnglish"
SPACE_TTS       = "E-motionAssistant/Space3"

@app.route("/", methods=["GET"])
def health_check():
    return "Flask AI Server is running. Send POST requests to /api/python/predict"

@app.route("/api/python/predict", methods=["POST"])
def unified_ai_pipeline():
    try:
        # Get data from frontend
        data = request.json
        user_input = data.get("message")  # Text string or Audio file path/URL
        lang = data.get("language", "english").lower() 
        mode = data.get("type", "text")   # "text" or "voice"

        # --- STEP 1: DETECT EMOTION ---
        if mode == "text":
            client_emo = Client(SPACE_TEXT_EMO, hf_token=HF_TOKEN)
            api_name = f"/predict"
            emotion_result = client_emo.predict(user_input, api_name=api_name)
        else:
            client_emo = Client(SPACE_AUDIO_EMO, hf_token=HF_TOKEN)
            emotion_result = client_emo.predict(user_input, api_name="/predict")

        # Clean emotion result
        final_emotion = emotion_result if isinstance(emotion_result, str) else emotion_result.get('label', 'neutral')
        
        # --- STEP 2: GENERATE LLM RESPONSE ---
        client_llm = Client(SPACE_LLM, hf_token=HF_TOKEN)
        client_llm.timeout = 360

        prompt = f"Language: {lang}. Detected Emotion: {final_emotion}. User said: {user_input}"
        llm_reply = client_llm.predict(prompt, api_name="/chat")

        # --- STEP 3: CONVERT TEXT TO SPEECH ---
        client_tts = Client(SPACE_TTS, hf_token=HF_TOKEN)
        # TTS usually returns a local file path to a .wav or .mp3
        temp_audio_path = client_tts.predict(llm_reply, lang, api_name=f"/{lang}_tts")

        # --- STEP 4: CONVERT AUDIO FILE TO BASE64 ---
        # This ensures the audio data survives the Vercel serverless hop
        audio_base64 = ""
        if os.path.exists(temp_audio_path):
            with open(temp_audio_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        # --- STEP 5: RETURN CONSOLIDATED RESPONSE ---
        return jsonify({
            "status": "success",
            "emotion": final_emotion,
            "reply_text": llm_reply,
            "reply_audio_base64": f"data:audio/wav;base64,{audio_base64}"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

