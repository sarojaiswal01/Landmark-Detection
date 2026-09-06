import os

# Base Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'landmark_model.keras')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LANDMARKS_JSON_PATH = os.path.join(DATA_DIR, 'landmarks.json')

# Upload & Temp folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Training Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 15
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5

# Prediction threshold for "Unknown Landmark"
CONFIDENCE_THRESHOLD = 0.10

# Default landmarks database (fallback / list of classes)
DEFAULT_CLASSES = []
if os.path.exists(LANDMARKS_JSON_PATH):
    try:
        import json
        with open(LANDMARKS_JSON_PATH, 'r', encoding='utf-8') as f:
            DEFAULT_CLASSES = list(json.load(f).keys())
    except Exception:
        pass

if not DEFAULT_CLASSES:
    DEFAULT_CLASSES = [
        "Eiffel_Tower",
        "Taj_Mahal",
        "Colosseum",
        "Pyramids_of_Giza",
        "Statue_of_Liberty",
        "Machu_Picchu",
        "Great_Wall_of_China",
        "Sydney_Opera_House",
        "Stonehenge",
        "Christ_the_Redeemer",
        "Golden_Gate_Bridge",
        "Burj_Khalifa"
    ]
