# ai-drowsiness-distraction-cnn-lstm-yolov8
# 🚗 Driver Monitoring System
**Final Year Project (FYP) — Real-Time Drowsiness & Distraction Detection**

A real-time AI-powered driver monitoring system that detects drowsiness and distraction using computer vision and deep learning. It features a live web dashboard connected via WebSocket.

---

## 📋 Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Models](#training-the-models)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Technologies Used](#technologies-used)

---

## ✨ Features

### Drowsiness Detection Mode
- 👁️ **Eye State Detection** — CNN model classifies eyes as open or closed
- 🥱 **Yawn Detection** — CNN model detects yawning in real-time
- 🧠 **LSTM Temporal Analysis** — Analyses sequences of frames for sustained drowsiness patterns
- 🔊 **Audio Alerts** — Continuous beep alarm triggered on danger state (Windows)

### Distraction Detection Mode
- 📱 **Phone Usage Detection** — YOLOv8 detects phone in driver's hand
- 🍶 **Drinking Detection** — YOLOv8 detects bottle/drink
- 👤 **Head Pose Estimation** — Detects if driver is not facing forward
- 🎯 **Face Absence Detection** — Alerts when driver's face is not visible

### Dashboard
- 📊 Real-time metrics (EAR, MAR, LSTM score)
- 🔔 Live alert log with timestamps
- 📡 WebSocket connection to Python backend
- 📱 Responsive design for desktop and mobile

---

## 🏗️ System Architecture

```
Camera Feed
    │
    ▼
[main.py] ─── Face Detection (Haar Cascade)
    │               ├── Eye CNN Model (.h5)
    │               ├── Yawn CNN Model (.h5)
    │               └── LSTM Model (.pth)
    │
    ├── YOLOv8 (Phone / Bottle Detection)
    ├── Head Pose Estimation
    │
    └── WebSocket Server (port 8765)
              │
              ▼
      [dashboard.html] — Live Browser Dashboard
```

---

## 📁 Project Structure

```
driver-monitoring-system/
│
├── main.py                     # Main detection system + WebSocket server
├── dashboard.html              # Real-time web dashboard
│
├── training/
│   ├── TRAIN_EYES_CNN.py       # Train eye state CNN model
│   ├── TRAIN_YAWN_CNN_2.0.py   # Train yawn detection CNN model
│   └── TRAIN_LSTM.py           # Train LSTM temporal model
│
├── models/
│   ├── eyes_model.h5           # ✅ Included in repo
│   ├── yawn_model.h5           # ✅ Included in repo
│   └── best_model.pth          # ⚠️ Too large — download separately (see below)
│
├── yolov8n.pt                  # YOLOv8 nano weights (auto-downloaded)
├── requirements.txt
└── README.md
```

> ⚠️ **LSTM model** (`best_model.pth`) is not included due to file size. 
---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- Webcam
- Windows OS (for audio alerts via `winsound`)
- NVIDIA GPU recommended (CUDA support)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/driver-monitoring-system.git
cd driver-monitoring-system
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. For GPU support (optional but recommended)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**5. Download the LSTM model** and place it inside the `models/` folder:
```
models/
├── eyes_model.h5       ← already included in the repo
└──yawn_model.h5       ← already included in the repo
```

**6. Update model paths in `main.py`**
```python
EYE_MODEL_PATH  = "models/eyes_model.h5"
YAWN_MODEL_PATH = "models/yawn_model.h5"
LSTM_MODEL_PATH = "models/best_model.pth"
```

---

## 🚀 Usage

**1. Start the system**
```bash
python main.py
```

**2. Open the dashboard**  
Open `dashboard.html` in your browser. It will automatically connect to the WebSocket server at `ws://localhost:8765`.

**3. Keyboard shortcuts** (in the OpenCV window)
| Key | Action |
|-----|--------|
| `M` | Toggle between Drowsiness / Distraction mode |
| `Q` | Quit the system |

---

## 🏋️ Training the Models

### Eye State CNN
```bash
python training/TRAIN_EYES_CNN.py
```
- Dataset: Images of open/closed eyes
- Expected folders: `open/`, `closed/`

### Yawn Detection CNN
```bash
python training/TRAIN_YAWN_CNN_2.0.py
```
- Dataset: Images of yawning/not yawning mouths
- Expected folders: `yawn/`, `no_yawn/`

### LSTM Drowsiness Detector
```bash
python training/TRAIN_LSTM.py
```
- Dataset: Video clips (SUST-DDD format)
- Expected folders: `drowsy/`, `not_drowsy/`

> Update the dataset paths at the top of each training script before running.

---

## 📊 Dashboard

The dashboard (`dashboard.html`) connects to the Python backend via WebSocket and displays:

| Section | Description |
|---------|-------------|
| Drowsiness Status | Current drowsiness state with severity level |
| Distraction Status | Phone, drinking, or head pose alerts |
| Eye Score (EAR) | Eye aspect ratio from CNN model |
| Yawn Score (MAR) | Mouth aspect ratio from CNN model |
| LSTM Score | Temporal drowsiness probability |
| Event Counters | Total drowsy events, yawns, and distractions |
| Distraction Indicators | Visual indicators for phone, bottle, gaze |
| Alert Log | Timestamped history of all alerts |

---

## 🔧 Configuration

Key settings can be adjusted in `main.py`:

```python
# Detection Thresholds
EYE_THRESHOLD  = 0.30   # Below this = eyes closed
YAWN_THRESHOLD = 0.60   # Above this = yawning

# Head Pose
HEAD_YAW_THRESHOLD   = 20   # degrees left/right
HEAD_PITCH_THRESHOLD = 15   # degrees up/down

# YOLO Confidence
PHONE_CONFIDENCE_THRESHOLD  = 0.05
BOTTLE_CONFIDENCE_THRESHOLD = 0.15

# Camera
CAMERA_ID = 0   # Change if using external webcam
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core language |
| OpenCV | Camera feed, face/eye detection |
| TensorFlow / Keras | CNN models (eye, yawn) |
| PyTorch | LSTM temporal model |
| YOLOv8 (Ultralytics) | Phone & bottle detection |
| WebSockets | Real-time data to dashboard |
| HTML / CSS / JS | Live web dashboard |


---

## 👤 Author
**Haniszaim**  

