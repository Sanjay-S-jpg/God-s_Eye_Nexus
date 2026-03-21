# config.py - Central Configuration for Crowd Intelligence System v6.0
# All settings in one place. Tokens loaded from .env file for security.

import os

# Load .env file for secrets (Telegram tokens, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — will read from OS env directly

# ============================================================
# MODEL SETTINGS
# ============================================================
MODEL_NAME = 'yolo11n.pt'               # YOLO11 Nano — fastest, GPU-optimized
WEAPON_MODEL_PATH = 'weapon_yolo.pt'     # Custom YOLO model for gun detection (optional)

# Detection confidence
CONFIDENCE_THRESHOLD = 0.25
WEAPON_CONF_THRESHOLD = 0.25

# ============================================================
# GPU / PERFORMANCE
# ============================================================
GPU_AUTO_DETECT = True          # Automatically use CUDA/MPS if available
GPU_ONLY = True
TARGET_FPS = 20                 # Target FPS for adaptive frame skipping
ADAPTIVE_FRAME_SKIP = True      # Auto-adjust frame skip based on current FPS
MIN_FRAME_SKIP = 1
MAX_FRAME_SKIP = 5
PROCESS_EVERY_N_FRAMES = 2     # Initial frame skip (adapts at runtime if enabled)
AI_INPUT_SIZE = 480             # AI input resolution (lower = faster: 320/416/512/640)
TARGET_DISPLAY_WIDTH = 960      # Display window width

# ============================================================
# PERSON DETECTION
# ============================================================
PERSON_CLASS_ID = 0

# ============================================================
# DETECTION SETTINGS
# ============================================================
GRID_SIZE = (4, 4)
DENSITY_LIMIT_PER_SECTOR = 4
MAX_LOG_ENTRIES = 15

# Alert thresholds
CROWD_DENSITY_ALERT_THRESHOLD = 15
STAMPEDE_VELOCITY_THRESHOLD = 80.0  # pixels/second (time-normalized)

# ============================================================
# FIGHT DETECTION (MediaPipe Pose)
# ============================================================
FIGHT_DETECTION_ENABLED = True
FIGHT_PROXIMITY_THRESHOLD = 200     # pixels — max distance between people to check
FIGHT_AGGRESSION_THRESHOLD = 0.4    # 0.0 to 1.0 — score above this = fight
FIGHT_ANALYSIS_INTERVAL = 3         # Run pose estimation every N frames

# ============================================================
# ALERT SETTINGS
# ============================================================
ALERT_COOLDOWN = 30
ENABLE_TELEGRAM = True
ENABLE_DESKTOP = True

# Telegram credentials — loaded from .env file (see .env.example)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# CAMERA SOURCE
# ============================================================
CAMERA_SOURCE = "Videos/stamp.mp4"
#CAMERA_SOURCE = 0  # Webcam
#CAMERA_SOURCE = "http://192.168.1.62:8080/video"  # IP Camera
