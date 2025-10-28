from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import threading
import base64  # ← ADD THIS
from collections import deque
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========================================================================
# CONFIGURATION
# ========================================================================
TARGET_SIZE = 200
CONFIDENCE_THRESHOLD = 0.70
TRACKING_CONFIDENCE = 0.70
SMOOTHING_WINDOW = 5

# ========================================================================
# GLOBALS
# ========================================================================
last_prediction = None
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
lock = threading.Lock()

# ========================================================================
# MEDIAPIPE + MODEL LOADING
# ========================================================================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,  # ← CHANGED TO FALSE for video stream
    max_num_hands=1,
    min_detection_confidence=CONFIDENCE_THRESHOLD,
    min_tracking_confidence=TRACKING_CONFIDENCE
)
mp_drawing = mp.solutions.drawing_utils

try:
    model = tf.keras.models.load_model('model/best_asl_model.h5')
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model, label_encoder = None, None


# ========================================================================
# FUNCTIONS
# ========================================================================
def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    for hand_landmarks in result.multi_hand_landmarks:
        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])
        return np.nan_to_num(np.array(landmark_list, dtype=np.float32))
    return None


def smooth_predictions(new_pred):
    prediction_history.append(new_pred)
    if len(prediction_history) < SMOOTHING_WINDOW:
        return new_pred
    labels = [p['label'] for p in prediction_history]
    most_common_label = max(set(labels), key=labels.count)
    confidences = [p['confidence'] for p in prediction_history if p['label'] == most_common_label]
    return {'label': most_common_label, 'confidence': float(np.mean(confidences))}


# ========================================================================
# ROUTES
# ========================================================================

@app.route('/')
def home():
    return jsonify({"message": "ASL backend is running!"}), 200


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None
    }), 200


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Receives base64-encoded image from frontend, returns label + confidence"""
    global last_prediction

    if request.method == 'GET':
        return jsonify({"message": "Predict endpoint is working! Send a POST request with image data."}), 200

    if model is None or label_encoder is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        try:
            img_data = base64.b64decode(data['image'].split(',')[1])
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image format: {str(e)}"}), 400

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
            
        frame = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))

        # Extract landmarks
        landmarks = extract_landmarks(frame)
        if landmarks is None:
            return jsonify({"label": None, "confidence": 0.0})

        # Predict
        input_data = np.expand_dims(landmarks, axis=0)
        pred = model.predict(input_data, verbose=0)
        pred_class = np.argmax(pred)
        label = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(np.max(pred))

        result = smooth_predictions({'label': label, 'confidence': confidence})
        with lock:
            last_prediction = result

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========================================================================
# MAIN
# ========================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
