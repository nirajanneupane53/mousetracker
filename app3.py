from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import threading
import mediapipe as mp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables
lock = threading.Lock()
hand_detector = mp.solutions.hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
drawing_utils = mp.solutions.drawing_utils

DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080

frameR = 100
smooth_factor = 0.2
prev_x, prev_y = 0, 0
last_index_y = None
scroll_threshold = 20  # Pixel threshold for scroll direction

@app.route('/')
def index():
    return render_template('remote_controller.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global prev_x, prev_y, last_index_y

    if 'frame' not in request.json:
        return jsonify({"error": "No frame data provided"}), 400

    screen_width = request.json.get('screen_width', DEFAULT_SCREEN_WIDTH)
    screen_height = request.json.get('screen_height', DEFAULT_SCREEN_HEIGHT)

    try:
        img_data = base64.b64decode(request.json['frame'].split(',')[1] if ',' in request.json['frame'] else request.json['frame'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400
    except Exception as e:
        return jsonify({"error": f"Image decoding error: {str(e)}"}), 400

    cam_w, cam_h = img.shape[1], img.shape[0]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb_img)

    response_data = {
        "mouse_move": None,
        "mouse_gesture": "none",
        "is_hand_detected": False
    }

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        response_data["is_hand_detected"] = True

        # Extract landmarks
        finger_tips = [4, 8, 12, 16, 20]
        finger_dips = [3, 6, 10, 14, 18]
        finger_states = [1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y else 0
                         for tip, dip in zip(finger_tips, finger_dips)]
        total_fingers = sum(finger_states)

        thumb, index, middle, ring, pinky = finger_states

        # Get index fingertip position
        ind_x = int(hand_landmarks.landmark[8].x * cam_w)
        ind_y = int(hand_landmarks.landmark[8].y * cam_h)

        # Cursor conversion
        conv_x = int(np.interp(ind_x, (frameR, cam_w - frameR), (0, screen_width)))
        conv_y = int(np.interp(ind_y, (frameR, cam_h - frameR), (0, screen_height)))
        smooth_x = int(prev_x + (conv_x - prev_x) * smooth_factor)
        smooth_y = int(prev_y + (conv_y - prev_y) * smooth_factor)
        prev_x, prev_y = smooth_x, smooth_y

        response_data["mouse_move"] = {"x": smooth_x, "y": smooth_y}

        # Gesture detection
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        pinch_dist = np.linalg.norm(
            np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
        )

        gesture = "none"

        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            if last_index_y is not None:
                dy = ind_y - last_index_y
                if abs(dy) > scroll_threshold:
                    gesture = "scroll_up" if dy < 0 else "scroll_down"
            last_index_y = ind_y
        elif total_fingers == 5:
            gesture = "move"
            last_index_y = None
        elif index == 1 and all(f == 0 for i, f in enumerate(finger_states) if i != 1):
            gesture = "left_click"
            last_index_y = None
        elif pinch_dist < 0.05 and index == 1 and thumb == 1:
            gesture = "left_hold"
            last_index_y = None
        elif all(f == 0 for f in finger_states):
            gesture = "right_click"
            last_index_y = None
        else:
            gesture = "unknown"
            last_index_y = None

        response_data["mouse_gesture"] = gesture
        print(response_data)

    return jsonify(response_data)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "frameR": frameR,
        "smooth_factor": smooth_factor,
        "default_screen_width": DEFAULT_SCREEN_WIDTH,
        "default_screen_height": DEFAULT_SCREEN_HEIGHT
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
