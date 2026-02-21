"""
FYP Driver Monitoring System - DASHBOARD CONNECTED V2
Logic: Unchanged (2 Modes, Adjusted Thresholds) + HEAD POSE ESTIMATION
Output: Formatted specifically for the new "Hybrid Driver Monitoring Dashboard"
"""

import cv2
import numpy as np
import asyncio
import websockets
import json
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
import tensorflow as tf
from keras import layers, models
import signal
import sys
import logging
import threading
from queue import Queue
import time
from datetime import datetime
import winsound  # For Windows sound alerts

# ==========================================
# 1. CONFIGURATION
# ==========================================
logging.getLogger('websockets').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# --- PATHS ---
EYE_MODEL_PATH = r"C:\Users\USER\Desktop\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\CODING\eyes_fixed_model_20251209_232128.h5"
YAWN_MODEL_PATH = r"C:\Users\USER\Desktop\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\CODING\yawn_model_final_20260126_114325.h5"
LSTM_MODEL_PATH = r"C:\Users\USER\Desktop\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\CODING\optimized_drowsiness_checkpoints\best_model.pth"
YOLO_MODEL_PATH = 'yolov8n.pt'

# --- UI COLORS (BGR) ---
COLORS = {
    'alert': (80, 200, 80),      # Green
    'warning': (0, 165, 255),    # Orange
    'danger': (50, 50, 255),     # Red
    'drowsy': (0, 0, 200),       # Dark Red
    'bg_dark': (30, 30, 30),     # Dark background
    'bg_medium': (50, 50, 50),   # Medium background
    'text_white': (255, 255, 255),
    'text_gray': (180, 180, 180),
    'accent': (255, 180, 0)      # Blue accent
}

# --- DETECTION SETTINGS ---
CAMERA_ID = 0
SEQUENCE_LENGTH = 24
FRAME_SIZE = (144, 144)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# --- ADJUSTED THRESHOLDS ---
EYE_THRESHOLD = 0.30            # < 0.30 = Closed
YAWN_THRESHOLD = 0.60           # Less Sensitive
YAWN_ASPECT_RATIO_THRESHOLD = 0.60 

# Distraction Thresholds
PHONE_CLASS = 67
BOTTLE_CLASS = 39
PHONE_CONFIDENCE_THRESHOLD = 0.05   
BOTTLE_CONFIDENCE_THRESHOLD = 0.15  
MIN_PHONE_AREA = 0.003
MAX_PHONE_AREA = 0.30
MIN_BOTTLE_AREA = 0.01
MAX_BOTTLE_AREA = 0.30

# Head Pose Thresholds
HEAD_YAW_THRESHOLD = 20      # degrees (left/right rotation)
HEAD_PITCH_THRESHOLD = 15    # degrees (up/down rotation)
HEAD_POSE_BUFFER_SIZE = 3    # frames to confirm head pose
NO_FACE_DETECTION_FRAMES = 5  # frames without face = distracted

# --- MODES ---
class DetectionMode:
    DROWSINESS = "drowsiness"
    DISTRACTION = "distraction"

MODE_CONFIG = {
    DetectionMode.DROWSINESS: {
        'name': 'DROWSINESS MODE',
        'color': COLORS['accent'],
        'detect_eyes': True, 'detect_yawn': True, 'detect_lstm': True,
        'detect_yolo': False, 'detect_head_pose': False
    },
    DetectionMode.DISTRACTION: {
        'name': 'DISTRACTION MODE',
        'color': COLORS['warning'],
        'detect_eyes': False, 'detect_yawn': False, 'detect_lstm': False,
        'detect_yolo': True, 'detect_head_pose': True
    }
}

# Global State
should_exit = False
connected_clients = set()
data_queue = Queue(maxsize=10)
models_loaded = False
current_mode = DetectionMode.DROWSINESS

# ==========================================
# 2. UI HELPER FUNCTIONS
# ==========================================
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_progress_bar(img, x, y, width, height, value, max_value, color, bg_color):
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    if max_value > 0:
        progress_width = int(min(1.0, value / max_value) * width)
        if progress_width > 0:
            cv2.rectangle(img, (x, y), (x + progress_width, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 80), 1)

def draw_metric_card(img, x, y, width, height, title, value, color):
    cv2.rectangle(img, (x, y), (x + width, y + height), COLORS['bg_medium'], -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 80), 1)
    cv2.putText(img, title, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_gray'], 1)
    cv2.putText(img, str(value), (x + 10, y + height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ==========================================
# 3. MODELS
# ==========================================
def build_cnn_model(img_size=96):
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout(0.2),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout(0.2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout(0.3),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.GlobalAveragePooling2D(), layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

class OptimizedLSTMDrowsinessDetector(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(OptimizedLSTMDrowsinessDetector, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            self._make_depthwise_block(32, 64, stride=2),
            self._make_depthwise_block(64, 128, stride=2),
            self._make_depthwise_block(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.2)
        )
        self.feature_size = 256 * 4 * 4
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5), nn.Linear(32, num_classes))
    
    def _make_depthwise_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        return self.classifier(lstm_out[:, -1, :])

def load_models():
    global models_loaded, face_cascade, eye_cascade, eye_model, yawn_model, lstm_model, device, yolo_model
    print("Loading models... Please wait.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_model = build_cnn_model(); eye_model.load_weights(EYE_MODEL_PATH)
    eye_model.predict(np.zeros((1, 96, 96, 3), dtype=np.float32), verbose=0) 
    yawn_model = build_cnn_model(); yawn_model.load_weights(YAWN_MODEL_PATH)
    yawn_model.predict(np.zeros((1, 96, 96, 3), dtype=np.float32), verbose=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = OptimizedLSTMDrowsinessDetector(HIDDEN_SIZE, NUM_LAYERS, dropout=DROPOUT)
    try:
        checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device)
        lstm_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        lstm_model.to(device)
        lstm_model.eval()
    except Exception as e: print(f"LSTM Load Error: {e}")
    try: yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e: print(f"YOLO Load Error: {e}")
    
    models_loaded = True
    print("✅ System Ready!")

# ==========================================
# 4. STATE & UTILS
# ==========================================
class StateCache:
    def __init__(self):
        self.ear = 0.50
        self.mar = 0.0
        self.mouth_aspect_ratio = 0.0
        self.lstm_drowsy = 0.0
        self.lstm_ready = False
        self.has_phone = False
        self.has_bottle = False
        self.phone_conf = 0.0
        self.bottle_conf = 0.0
        self.yawn_cooldown = 0
        self.phone_buffer = deque(maxlen=10)
        self.bottle_buffer = deque(maxlen=10)
        # Head pose / Face detection
        self.no_face_counter = 0
        self.face_not_detected = False
        self.head_pitch = 0.0
        self.head_yaw = 0.0
        self.head_roll = 0.0
        self.head_pose_method = "none"
        self.head_pose_buffer = deque(maxlen=HEAD_POSE_BUFFER_SIZE)
        self.looking_away = False
        self.looking_down = False
        # Sound alert tracking
        self.last_alert_time = 0
        self.alert_cooldown = 0.1  # Very short cooldown for continuous checking
        self.is_playing_alert = False
        self.alert_sound_thread = None
        self.should_stop_alert = False

cache = StateCache()

def calculate_mouth_aspect_ratio(mouth_region):
    try:
        gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(largest)
            if w > 0: return h / w
    except: pass
    return 0.0

def play_alert_sound(alert_type="danger"):
    """
    Play CONTINUOUS alert sound that keeps going until the alert state ends.
    Uses winsound for Windows systems.
    """
    global cache
    
    # If already playing, don't start another thread
    if cache.is_playing_alert:
        return
    
    def continuous_beep_thread():
        try:
            cache.is_playing_alert = True
            cache.should_stop_alert = False
            
            # Keep beeping continuously until told to stop
            while not cache.should_stop_alert and not should_exit:
                if alert_type == "danger":
                    # Loud urgent beep pattern
                    winsound.Beep(1500, 400)  # 1500Hz for 400ms (LOUD)
                    time.sleep(0.1)  # Very short gap
                    winsound.Beep(1500, 400)  # Double beep pattern
                    time.sleep(0.3)  # Short pause before repeating
                elif alert_type == "warning":
                    winsound.Beep(1000, 300)
                    time.sleep(0.2)
            
        except Exception as e:
            print(f"Alert sound error: {e}")
        finally:
            cache.is_playing_alert = False
    
    # Start continuous beeping in separate thread
    cache.alert_sound_thread = threading.Thread(target=continuous_beep_thread, daemon=True)
    cache.alert_sound_thread.start()

def stop_alert_sound():
    """
    Stop the continuous alert sound.
    """
    global cache
    cache.should_stop_alert = True
    # Give it a moment to stop
    time.sleep(0.1)

def estimate_head_pose(frame, face_box=None):
    """
    Estimate head pose using face landmarks.
    Returns: pitch, yaw, roll, visualization_data, method
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get face region
        if face_box is not None:
            fx, fy, fw, fh = face_box
            face_center_x = fx + fw // 2
            face_center_y = fy + fh // 2
            
            # Simple estimation based on face position in frame
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate yaw (left-right) based on horizontal position
            # Center = 0, Left = negative, Right = positive
            yaw = ((face_center_x / frame_w) - 0.5) * 60  # Scale to ±30 degrees
            
            # Calculate pitch (up-down) based on vertical position
            # Center = 0, Up = negative, Down = positive
            pitch = ((face_center_y / frame_h) - 0.45) * 50  # Scale to ±25 degrees
            
            roll = 0.0  # Not estimating roll with this simple method
            
            viz_data = {
                'face_center': (face_center_x, face_center_y),
                'frame_center': (frame_w // 2, frame_h // 2)
            }
            
            return pitch, yaw, roll, viz_data, "simple"
        else:
            # No face detected
            return 0.0, 0.0, 0.0, None, "none"
            
    except Exception as e:
        return 0.0, 0.0, 0.0, None, "error"

# ==========================================
# 5. CORE DETECTION LOGIC
# ==========================================
def detect_driver_state(frame, lstm_buffer, frame_count, mode_config):
    global cache
    
    # --- FACE DETECTION FIRST (needed for multiple features) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    face_box, eye_boxes, mouth_box = None, [], None
    
    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_box = (fx, fy, fw, fh)
        cache.no_face_counter = 0
        cache.face_not_detected = False
    else:
        cache.no_face_counter += 1
        if cache.no_face_counter >= NO_FACE_DETECTION_FRAMES:
            cache.face_not_detected = True
    
    # --- HEAD POSE ESTIMATION (INDEPENDENT, with optional face_box for fallback) ---
    head_viz_data = None
    if mode_config['detect_head_pose']:
        pitch, yaw, roll, head_viz_data, method = estimate_head_pose(frame, face_box)
        cache.head_pitch = pitch
        cache.head_yaw = yaw
        cache.head_roll = roll
        cache.head_pose_method = method
        
        # Determine if looking away or down
        looking_away = abs(yaw) > HEAD_YAW_THRESHOLD
        looking_down = pitch < -HEAD_PITCH_THRESHOLD
        
        # Buffer for stability
        cache.head_pose_buffer.append((looking_away, looking_down))
        
        # Confirm if most recent frames agree (reduced threshold for faster response)
        if len(cache.head_pose_buffer) >= 2:
            away_votes = sum(1 for la, _ in cache.head_pose_buffer if la)
            down_votes = sum(1 for _, ld in cache.head_pose_buffer if ld)
            cache.looking_away = away_votes >= 2
            cache.looking_down = down_votes >= 2
        else:
            cache.looking_away = False
            cache.looking_down = False
    
    # --- YOLO DISTRACTION (INDEPENDENT) ---
    if mode_config['detect_yolo'] and yolo_model is not None:
        try:
            results = yolo_model(frame, verbose=False, classes=[PHONE_CLASS, BOTTLE_CLASS], 
                               conf=0.03, imgsz=640, device=device, iou=0.25, max_det=5)[0]
            
            frame_has_phone, frame_has_bottle = False, False
            curr_phone_conf, curr_bottle_conf = 0, 0
            
            for box in results.boxes.data:
                conf, cls = float(box[4]), int(box[5])
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                area_ratio = box_area / frame_area
                center_x = (x1 + x2) / 2
                
                if cls == PHONE_CLASS and conf > PHONE_CONFIDENCE_THRESHOLD:
                    if MIN_PHONE_AREA <= area_ratio <= MAX_PHONE_AREA:
                        margin = frame.shape[1] * 0.02
                        if margin < center_x < frame.shape[1] - margin:
                            frame_has_phone = True
                            curr_phone_conf = max(curr_phone_conf, conf)

                elif cls == BOTTLE_CLASS and conf > BOTTLE_CONFIDENCE_THRESHOLD:
                     if MIN_BOTTLE_AREA <= area_ratio <= MAX_BOTTLE_AREA:
                        frame_has_bottle = True
                        curr_bottle_conf = max(curr_bottle_conf, conf)
            
            cache.phone_buffer.append(frame_has_phone)
            cache.bottle_buffer.append(frame_has_bottle)
            
            cache.has_phone = sum(cache.phone_buffer) >= 2
            cache.has_bottle = sum(cache.bottle_buffer) >= 3
            
            cache.phone_conf = curr_phone_conf if frame_has_phone else 0.0
            cache.bottle_conf = curr_bottle_conf if frame_has_bottle else 0.0
            
            if cache.has_phone and cache.has_bottle:
                if cache.phone_conf > cache.bottle_conf: cache.has_bottle = False
                else: cache.has_phone = False

        except Exception: pass
    else:
        cache.has_phone, cache.has_bottle = False, False
        cache.phone_conf, cache.bottle_conf = 0.0, 0.0

    # --- EYES & YAWN (if face detected) ---
    if face_box:
        fx, fy, fw, fh = face_box
        
        # Eyes
        if mode_config['detect_eyes']:
            eye_roi = gray[fy:fy+int(fh*0.55), fx:fx+fw]
            eye_color_roi = frame[fy:fy+int(fh*0.55), fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(eye_roi, 1.1, 3, minSize=(25, 25))
            preds = []
            for ex, ey, ew, eh in eyes[:2]:
                eye_img = eye_color_roi[ey:ey+eh, ex:ex+ew]
                if eye_img.size > 0:
                    rgb_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
                    eye_input = np.expand_dims(cv2.resize(rgb_eye, (96, 96)).astype(np.float32), axis=0)
                    preds.append(eye_model.predict(eye_input, verbose=0)[0][0])
                    eye_boxes.append((fx+ex, fy+ey, ew, eh))
            if preds: cache.ear = sum(preds) / len(preds)

        # Yawn
        if mode_config['detect_yawn']:
            my, mx, mx2 = int(fh*0.50), int(fw*0.15), int(fw*0.85)
            mouth_img = frame[fy+my:fy+fh, fx+mx:fx+mx2]
            mouth_box = (fx+mx, fy+my, mx2-mx, fh-my)
            
            if mouth_img.size > 0:
                rgb_mouth = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB)
                mouth_input = np.expand_dims(cv2.resize(rgb_mouth, (96, 96)).astype(np.float32), axis=0)
                cache.mar = float(yawn_model.predict(mouth_input, verbose=0)[0][0])
                cache.mouth_aspect_ratio = calculate_mouth_aspect_ratio(mouth_img)

    # --- LSTM ---
    if mode_config['detect_lstm'] and frame_count % 6 == 0:
        resized = cv2.resize(frame, FRAME_SIZE)
        normalized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        lstm_buffer.append(np.transpose(normalized, (2, 0, 1)))
        if len(lstm_buffer) == SEQUENCE_LENGTH:
            try:
                seq_tensor = torch.from_numpy(np.expand_dims(np.array(list(lstm_buffer)), axis=0)).float().to(device)
                with torch.no_grad():
                    outputs = lstm_model(seq_tensor)
                    cache.lstm_drowsy = torch.nn.functional.softmax(outputs, dim=1)[0][1].item()
            except: pass

    # --- STATUS LOGIC ---
    status = "ALERT"
    status_color = COLORS['alert']
    alert_level = 0
    
    # FIXED YAWN LOGIC: Use ONLY the CNN model (cache.mar) that is shown in the meter
    # This ensures the alert matches what the user sees in the yawn meter display
    yawn_detected = cache.mar > YAWN_THRESHOLD
    
    # Decision Tree (PRIORITY ORDER)
    if mode_config['detect_yolo'] and cache.has_phone and cache.phone_conf > 0.01:
        status = f"PHONE: {cache.phone_conf:.2f}"
        status_color = COLORS['danger']
        alert_level = 3
    elif mode_config['detect_yolo'] and cache.has_bottle and cache.bottle_conf > 0.01:
        status = f"DRINKING: {cache.bottle_conf:.2f}"
        status_color = COLORS['warning']
        alert_level = 2
    elif mode_config['detect_head_pose'] and cache.face_not_detected:
        status = "NOT FACING FORWARD"
        status_color = COLORS['danger']
        alert_level = 3
    elif cache.lstm_drowsy > 0.85 and mode_config['detect_lstm']:
        status = "DROWSY (LSTM)"
        status_color = COLORS['drowsy']
        alert_level = 3
    elif cache.ear < EYE_THRESHOLD and yawn_detected and mode_config['detect_eyes']:
        status = "DROWSY (EYES+YAWN)"
        status_color = COLORS['drowsy']
        alert_level = 3
    elif cache.ear < EYE_THRESHOLD and mode_config['detect_eyes']:
        status = "EYES CLOSED"
        status_color = COLORS['danger']
        alert_level = 2
    elif yawn_detected and mode_config['detect_yawn']:
        status = "YAWNING"
        status_color = COLORS['warning']
        alert_level = 1

    return status, status_color, alert_level, face_box, eye_boxes, mouth_box

# ==========================================
# 6. MAIN THREAD
# ==========================================
def camera_thread_function():
    global should_exit, current_mode, cache
    if not models_loaded: return
    
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    lstm_buffer = deque(maxlen=SEQUENCE_LENGTH)
    drowsy_count, yawn_count, dist_count = 0, 0, 0
    prev_status = "ALERT"
    frame_count = 0
    
    # FPS Calculation Variables
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    while not should_exit:
        ret, frame = cap.read()
        if not ret: continue
        frame_count += 1
        
        # Calculate FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()
        
        process_frame = cv2.resize(frame, (640, 480))
        display = cv2.resize(frame, (1280, 720))
        
        mode_config = MODE_CONFIG[current_mode]
        status, color, level, fbox, eboxes, mbox = detect_driver_state(
            process_frame, lstm_buffer, frame_count, mode_config
        )
        
        # --- SOUND ALERTS for RED (Danger) conditions ---
        # Play CONTINUOUS alert for high-priority dangerous conditions
        # Alert keeps playing until condition is resolved
        if level >= 2:  # Alert level 2 or 3 (Red text conditions)
            if "PHONE" in status or "NOT FACING" in status or "DROWSY" in status or "EYES CLOSED" in status:
                play_alert_sound("danger")
        else:
            # Stop the alert when returning to safe state
            if cache.is_playing_alert:
                stop_alert_sound()
        
        # --- COUNTERS ---
        if "DROWSY" in status and "DROWSY" not in prev_status: drowsy_count += 1
        if "PHONE" in status and "PHONE" not in prev_status: dist_count += 1
        if "NOT FACING" in status and "NOT FACING" not in prev_status: dist_count += 1
        
        if "YAWNING" in status:
            if "YAWNING" not in prev_status and cache.yawn_cooldown == 0:
                yawn_count += 1
                cache.yawn_cooldown = 90
        if cache.yawn_cooldown > 0: cache.yawn_cooldown -= 1
        
        prev_status = status
        
        # --- UI DRAWING (LOCAL WINDOW) ---
        cv2.rectangle(display, (0, 0), (1280, 100), COLORS['bg_dark'], -1)
        pulse = int(abs(np.sin(time.time() * 5) * 50)) if level >= 2 else 0
        final_color = tuple(min(255, c + pulse) for c in color)
        
        cv2.putText(display, status, (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, final_color, 3)
        cv2.putText(display, f"MODE: {mode_config['name']}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_gray'], 1)

        if fbox:
            fx, fy, fw, fh = fbox
            fx, fy, fw, fh = fx*2, fy*1.5, fw*2, fh*1.5 
            draw_rounded_rect(display, (fx, fy), (fx+fw, fy+fh), COLORS['accent'], 2, 15)
            
            if mode_config['detect_eyes']:
                for ex, ey, ew, eh in eboxes:
                    ex, ey, ew, eh = ex*2, ey*1.5, ew*2, eh*1.5
                    ec = COLORS['danger'] if cache.ear < EYE_THRESHOLD else COLORS['alert']
                    cv2.circle(display, (int(ex+ew//2), int(ey+eh//2)), int(ew//2), ec, 2)
            
            if mode_config['detect_yawn'] and mbox:
                mx, my, mw, mh = mbox
                mx, my, mw, mh = mx*2, my*1.5, mw*2, mh*1.5
                mc = COLORS['warning'] if "YAWN" in status else COLORS['alert']
                draw_rounded_rect(display, (mx, my), (mx+mw, my+mh), mc, 2, 5)

        panel_y = 600
        cv2.rectangle(display, (0, panel_y), (1280, 720), COLORS['bg_dark'], -1)
        
        curr_x = 20
        if mode_config['detect_eyes']:
            ec = COLORS['danger'] if cache.ear < EYE_THRESHOLD else COLORS['alert']
            draw_metric_card(display, curr_x, panel_y+20, 180, 80, "EYE SCORE", f"{cache.ear:.2f}", ec)
            draw_progress_bar(display, curr_x+10, panel_y+80, 170, 5, cache.ear, 1.0, ec, COLORS['bg_medium'])
            curr_x += 200
            
        if mode_config['detect_yawn']:
            yc = COLORS['warning'] if cache.mar > YAWN_THRESHOLD else COLORS['alert']
            draw_metric_card(display, curr_x, panel_y+20, 180, 80, "YAWN SCORE", f"{cache.mar:.2f}", yc)
            draw_progress_bar(display, curr_x+10, panel_y+80, 170, 5, cache.mar, 1.0, yc, COLORS['bg_medium'])
            curr_x += 200
            
        if mode_config['detect_yolo']:
            pc = COLORS['danger'] if cache.has_phone and cache.phone_conf > 0 else COLORS['text_gray']
            draw_metric_card(display, curr_x, panel_y+20, 180, 80, "PHONE CONF", f"{cache.phone_conf:.2f}", pc)
            curr_x += 200
            
            bc = COLORS['warning'] if cache.has_bottle and cache.bottle_conf > 0 else COLORS['text_gray']
            draw_metric_card(display, curr_x, panel_y+20, 180, 80, "BOTTLE CONF", f"{cache.bottle_conf:.2f}", bc)
            curr_x += 200

        cv2.imshow("Driver Monitoring System", display)
        
        # --- DATA FOR DASHBOARD ---
        # NOTE: Formatting data exactly for the NEW HTML dashboard structure
        
        # Separate Statuses
        drowsiness_status = "Alert"
        if "DROWSY" in status: drowsiness_status = "Drowsy"
        elif "YAWN" in status: drowsiness_status = "Yawning"
        elif "EYES" in status: drowsiness_status = "Eyes Closed"
        
        distraction_status = "✅ SAFE DRIVING"
        if "PHONE" in status: distraction_status = f"📱 PHONE USAGE"
        elif "DRINKING" in status: distraction_status = f"🍶 DRINKING"
        elif "NOT FACING" in status: distraction_status = f"⚠️ NOT FACING FORWARD"
        
        dashboard_data = {
            "status": drowsiness_status,
            "distractionStatus": distraction_status,
            "fps": current_fps,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "ear": float(cache.ear),
                "mar": float(cache.mar),
                "lstmDrowsy": float(cache.lstm_drowsy),
                "lstmReady": len(lstm_buffer) == SEQUENCE_LENGTH,
                "drowsyCount": drowsy_count // 10, # Slow down count for UI
                "yawnCount": yawn_count,
                "distractionCount": dist_count,
                "hasPhone": bool(cache.has_phone and cache.phone_conf > 0.01),
                "hasBottle": bool(cache.has_bottle and cache.bottle_conf > 0.01),
                "lookingAway": bool(cache.face_not_detected),
                "lookingDown": False
            }
        }
        
        if not data_queue.full():
            data_queue.put(dashboard_data)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): should_exit = True; break
        elif key == ord('m'):
            current_mode = DetectionMode.DISTRACTION if current_mode == DetectionMode.DROWSINESS else DetectionMode.DROWSINESS
            print(f"SWITCHED MODE: {current_mode}")

    # Stop any playing alerts before exiting
    if cache.is_playing_alert:
        stop_alert_sound()
    
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 7. ASYNC SERVER
# ==========================================
async def handle_client(websocket):
    connected_clients.add(websocket)
    try:
        async for _ in websocket: pass
    except: pass
    finally: connected_clients.discard(websocket)

async def broadcast_loop():
    while not should_exit:
        if not data_queue.empty():
            data = json.dumps(data_queue.get())
            if connected_clients:
                await asyncio.gather(*[client.send(data) for client in connected_clients], return_exceptions=True)
        await asyncio.sleep(0.01)

async def main():
    global should_exit
    load_models()
    t = threading.Thread(target=camera_thread_function, daemon=True)
    t.start()
    
    async with websockets.serve(handle_client, "localhost", 8765):
        print("\nSERVER STARTED on ws://localhost:8765")
        try: await broadcast_loop()
        except KeyboardInterrupt: pass
        finally: should_exit = True; t.join()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())