from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import threading
from collections import deque

app = Flask(__name__)

# ============================================================================
# CONFIGURATION - Match your training setup
# ============================================================================
TARGET_SIZE = 200  # Match training resolution
CONFIDENCE_THRESHOLD = 0.70  # Higher threshold for stability
TRACKING_CONFIDENCE = 0.70
SMOOTHING_WINDOW = 5  # Number of predictions to average (set to 1 to disable)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
last_frame = None
last_prediction = None
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
lock = threading.Lock()
stop_thread = False

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=CONFIDENCE_THRESHOLD,
    min_tracking_confidence=TRACKING_CONFIDENCE
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================================================================
# LOAD MODEL & LABEL ENCODER
# ============================================================================
try:
    model = tf.keras.models.load_model('model/best_asl_model.h5')
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✅ Model loaded! Classes: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, label_encoder = None, None


# ============================================================================
# LANDMARK EXTRACTION (SAME AS YOUR TESTING CODE)
# ============================================================================
def extract_landmarks(frame):
    """
    Extract 63D landmarks from frame resized to TARGET_SIZE (200x200).
    Returns: (landmarks_array or None, annotated_frame)
    """
    # Resize to match training size
    resized = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, frame

    for hand_landmarks in result.multi_hand_landmarks:
        # Draw landmarks on ORIGINAL frame for display
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extract landmarks (x, y, z) flattened to 63D
        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

        # Convert to numpy and normalize (CRITICAL for stability)
        landmarks = np.array(landmark_list, dtype=np.float32)
        landmarks = np.nan_to_num(landmarks, nan=0.0)  # Replace NaNs
        landmarks = np.clip(landmarks, -1.0, 1.0)  # Clip to valid range

        return landmarks, frame

    return None, frame


# ============================================================================
# PREDICTION SMOOTHING (OPTIONAL BUT RECOMMENDED)
# ============================================================================
def smooth_predictions(new_prediction):
    """
    Apply moving average smoothing to reduce flickering.
    Uses majority voting for label and averages confidence.
    """
    prediction_history.append(new_prediction)

    if len(prediction_history) < SMOOTHING_WINDOW:
        return new_prediction  # Not enough data yet

    # Majority voting for label
    labels = [p['label'] for p in prediction_history]
    most_common_label = max(set(labels), key=labels.count)

    # Average confidence for the most common label
    confidences = [p['confidence'] for p in prediction_history if p['label'] == most_common_label]
    avg_confidence = np.mean(confidences) if confidences else new_prediction['confidence']

    return {
        'label': most_common_label,
        'confidence': float(avg_confidence)
    }


# ============================================================================
# BACKGROUND PREDICTION THREAD
# ============================================================================
def prediction_loop():
    """Background thread for continuous predictions."""
    global last_prediction

    while not stop_thread:
        with lock:
            frame_copy = None if last_frame is None else last_frame.copy()

        if frame_copy is None or model is None or label_encoder is None:
            continue

        # Extract landmarks
        landmarks, _ = extract_landmarks(frame_copy)

        if landmarks is None:
            with lock:
                last_prediction = None
                prediction_history.clear()  # Reset smoothing on no detection
            continue

        # Prepare input for model
        input_data = np.expand_dims(landmarks, axis=0)

        # Predict
        pred = model.predict(input_data, verbose=0)
        pred_class = np.argmax(pred)
        label = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(np.max(pred))

        # Apply smoothing
        raw_prediction = {'label': label, 'confidence': confidence}
        smoothed = smooth_predictions(raw_prediction)

        with lock:
            last_prediction = smoothed


# ============================================================================
# VIDEO FEED GENERATOR
# ============================================================================
def generate_frames():
    """Stream webcam with hand landmarks drawn."""
    global last_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # Mirror effect

            # Extract and draw landmarks
            _, annotated_frame = extract_landmarks(frame.copy())

            with lock:
                last_frame = frame  # Store raw frame for prediction

            # Encode and stream
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        cap.release()


# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict', methods=['GET'])
def predict():
    """Return current prediction as JSON."""
    with lock:
        if last_prediction:
            return jsonify(last_prediction)
        return jsonify({'label': None, 'confidence': 0.0})


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    # Start prediction thread
    thread = threading.Thread(target=prediction_loop, daemon=True)
    thread.start()

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    finally:
        stop_thread = True
        thread.join()
