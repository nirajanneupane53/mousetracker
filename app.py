from flask import Flask, render_template, Response, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
import threading
import mouse
from deepface import DeepFace
import mediapipe as mp
import threading
import numpy as np  

# Global variables
camera = None  
lock = threading.Lock()
frame = None
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils
mouse_down = False
running = False  
frame_skip_interval = 2 
last_processed_time = 0

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('handcontroller.html')

@app.route("/handcontroller")
def emotion_match():
    return render_template("handcontroller.html")

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, running
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'status': 'Error initializing camera'})
        
        running = True  # processing thread
        threading.Thread(target=process_frame, daemon=True).start()  # frame processing in a separate thread
    return jsonify({'status': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, running
    if camera:
        running = False 
        camera.release()
        camera = None
    return jsonify({'status': 'Camera stopped'})

def process_frame():
    global frame, running, prev_x, prev_y, mouse_down

    detector = HandDetector(detectionCon=0.9, maxHands=1)

    cam_w, cam_h = 640, 480
    camera.set(3, cam_w)
    camera.set(4, cam_h)

    frameR = 100  # Frame reduction 
    smooth_factor = 0.2 

    prev_x, prev_y = 0, 0

    while running:
        success, img = camera.read()
        if not success:
            continue
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)
        cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)

        if hands:
            lmlist = hands[0]['lmList']
            ind_x, ind_y = lmlist[8][0], lmlist[8][1]
            thumb_x, thumb_y = lmlist[4][0], lmlist[4][1]

            cv2.circle(img, (ind_x, ind_y), 5, (0, 255, 255), 2)

            conv_x = int(np.interp(ind_x, (frameR, cam_w - frameR), (0, 1536)))
            conv_y = int(np.interp(ind_y, (frameR, cam_h - frameR), (0, 864)))

            smooth_x = int(prev_x + (conv_x - prev_x) * smooth_factor)
            smooth_y = int(prev_y + (conv_y - prev_y) * smooth_factor)

            mouse.move(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

            # Distance between ind thumb
            distance = np.sqrt((ind_x - thumb_x) ** 2 + (ind_y - thumb_y) ** 2)

            # Pinching gesture
            pinch_threshold = 35  # threshold adjustment
            if distance < pinch_threshold:
                if not mouse_down:
                    mouse.press(button="left") 
                    mouse_down = True
                    print("click")
            else:
                if mouse_down:
                    mouse.release(button="left") 
                    mouse_down = False
                    print("released")

def generate_frames():
    global camera
    while camera is not None:
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame")
            break
        else:
            frame = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if camera is None:
        return jsonify({'error': 'Please wait for a moment...'})
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

def generate_frames():
    while running:
        with lock:
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
