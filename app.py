import os
import uuid
import json
import base64
import time
import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import config
import predict
import requests

app = Flask(__name__)

def fetch_wikipedia_summary(landmark_name):
    """
    Fetches the summary text for a landmark name from Wikipedia's REST API.
    """
    try:
        title = landmark_name.replace(' ', '_')
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        headers = {'User-Agent': 'LandmarkLensAI/1.0 (contact: support@landmarklensai.local)'}
        
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', '')
    except Exception as e:
        print(f"[!] Error fetching Wikipedia summary for {landmark_name}: {e}")
    return ''

def fetch_local_weather(latitude, longitude):
    """
    Queries current temperature and WMO weather codes from Open-Meteo API.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            temp = current.get('temperature_2m')
            code = current.get('weather_code')
            
            condition, icon = map_wmo_code(code)
            return {
                "temp": f"{round(temp)}°C" if temp is not None else "N/A",
                "condition": condition,
                "icon": icon
            }
    except Exception as e:
        print(f"[!] Error fetching weather for ({latitude}, {longitude}): {e}")
    return None

def map_wmo_code(code):
    """
    Maps WMO Weather Interpretation Codes (WW) to condition text and font-awesome icon.
    """
    if code is None:
        return "Unknown", "fa-question"
        
    if code == 0:
        return "Clear Sky", "fa-sun"
    elif code in [1, 2, 3]:
        return "Partly Cloudy", "fa-cloud-sun"
    elif code in [45, 48]:
        return "Foggy", "fa-smog"
    elif code in [51, 53, 55]:
        return "Drizzle", "fa-cloud-rain"
    elif code in [61, 63, 65]:
        return "Rainy", "fa-cloud-showers-heavy"
    elif code in [66, 67]:
        return "Freezing Rain", "fa-cloud-meatball"
    elif code in [71, 73, 75, 77]:
        return "Snowy", "fa-snowflake"
    elif code in [80, 81, 82]:
        return "Rain Showers", "fa-cloud-showers-water"
    elif code in [85, 86]:
        return "Snow Showers", "fa-snowflake"
    elif code in [95, 96, 99]:
        return "Thunderstorm", "fa-cloud-bolt"
        
    return "Overcast", "fa-cloud"
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

HISTORY_FILE = os.path.join(config.DATA_DIR, 'history.json')

def load_history():
    """
    Loads prediction history from history.json.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_history(history):
    """
    Saves prediction history to history.json.
    """
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[!] Error saving history: {e}")

def add_to_history(landmark_name, confidence, is_detected, image_url, heatmap_url=None, latitude=None, longitude=None, weather=None, wikipedia_summary=None):
    """
    Adds a new record to the top of the history list. Keeps the history capped at 20 entries.
    """
    history = load_history()
    new_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "landmark_name": landmark_name,
        "confidence": confidence,
        "is_detected": is_detected,
        "image_url": image_url,
        "heatmap_url": heatmap_url,
        "latitude": latitude,
        "longitude": longitude,
        "weather": weather,
        "wikipedia_summary": wikipedia_summary
    }
    history.insert(0, new_entry)
    save_history(history[:20])  # Cap at 20 entries

@app.route('/')
def home():
    """
    Renders the main SPA dashboard.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    Accepts file upload or webcam base64 string, runs prediction, and returns results.
    """
    # 1. Handle File Upload
    file_path = None
    original_filename = ""
    
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            original_filename = file.filename
            
    # 2. Handle Webcam Base64 Data
    elif request.json and 'image' in request.json:
        image_data = request.json['image']
        if image_data.startswith('data:image'):
            # Extract base64 header
            header, encoded = image_data.split(",", 1)
            # Determine extension
            ext = header.split(";")[0].split("/")[1]
            filename = f"webcam_{uuid.uuid4().hex}.{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            with open(file_path, "wb") as fh:
                fh.write(base64.b64decode(encoded))
            original_filename = "webcam_capture.jpg"

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No image provided or failed to save file"}), 400

    try:
        # Run inference
        start_time = time.time()
        results = predict.predict_landmark(file_path)
        inference_time = time.time() - start_time
        
        # Add details
        image_url = f"/static/uploads/{os.path.basename(file_path)}"
        results["image_url"] = image_url
        results["inference_time"] = f"{inference_time:.3f}s"
        
        # Save to history
        lat = None
        lng = None
        weather_data = None
        wiki_summary = None
        
        if results.get("is_landmark_detected") and results.get("details"):
            lat = results["details"].get("latitude")
            lng = results["details"].get("longitude")
            
            # 1. Fetch real-time weather details
            if lat is not None and lng is not None:
                weather_data = fetch_local_weather(lat, lng)
                
            # 2. Fetch live Wikipedia summary description
            wiki_summary = fetch_wikipedia_summary(results["landmark_name"])
            if wiki_summary:
                # Override static dictionary summary text with dynamic Wikipedia summary
                results["details"]["historical_background"] = wiki_summary

        results["weather"] = weather_data
        results["wikipedia_summary"] = wiki_summary

        add_to_history(
            landmark_name=results["landmark_name"],
            confidence=results["confidence"],
            is_detected=results["is_landmark_detected"],
            image_url=image_url,
            heatmap_url=results["heatmap_url"],
            latitude=lat,
            longitude=lng,
            weather=weather_data,
            wikipedia_summary=wiki_summary
        )
        
        return jsonify(results)
    except Exception as e:
        print(f"[!] Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

@app.route('/history', methods=['GET'])
def get_prediction_history():
    """
    Endpoint to retrieve the recent history of uploads.
    """
    return jsonify(load_history())

@app.route('/clear-history', methods=['POST'])
def clear_prediction_history():
    """
    Endpoint to clear the history.json.
    """
    save_history([])
    return jsonify({"status": "success", "message": "History cleared successfully"})

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Returns system and model parameters for display on the statistics panel.
    """
    model_exists = os.path.exists(config.MODEL_PATH)
    classes = predict.get_class_names()
    
    # Simple CPU/RAM check
    import platform
    system_info = {
        "os": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }
    
    # Check if a custom model is loaded or if it's the ImageNet template
    is_trained = False
    if model_exists and predict.HAS_TENSORFLOW:
        if os.path.exists(config.DATASET_DIR) and len(os.listdir(config.DATASET_DIR)) >= 2:
            is_trained = True

    model_type = "EfficientNetB0 (Transfer Learning)" if is_trained else "EfficientNetB0 (ImageNet Base - Demo Mode)"
    status_label = "Ready" if model_exists else "Awaiting Training"
    
    if not predict.HAS_TENSORFLOW:
        model_type = "EfficientNetB0 (Emulation Mode - DLL Blocked)"
        status_label = "Emulated"

    return jsonify({
        "model_loaded": model_exists and predict.HAS_TENSORFLOW,
        "model_type": model_type,
        "classes_count": len(classes),
        "classes": classes,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "system": system_info,
        "status": status_label,
        "tensorflow_loaded": predict.HAS_TENSORFLOW,
        "tensorflow_error": predict.TENSORFLOW_ERROR
    })

# Add custom route to serve uploaded images (in case static folder isn't default configured)
@app.route('/static/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    print("[*] Pre-loading model...")
    predict.load_or_create_model()
    print("[*] Starting Flask server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)


