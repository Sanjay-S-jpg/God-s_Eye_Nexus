# config.py - Central Configuration for Crowd Intelligence System

# ============================================================
# MODEL SETTINGS
# ============================================================
MODEL_NAME = 'yolov8n.pt'  # Nano model (fastest)

# Detection confidence
CONFIDENCE_THRESHOLD = 0.20
WEAPON_CONF_THRESHOLD = 0.25  # Lower for weapons (catch more)

# ============================================================
# WEAPON CLASSES (from COCO dataset in YOLO)
# ============================================================
# YOLO can detect these weapon-like objects:
PERSON_CLASS_ID = 0

WEAPON_CLASSES = {
    43: {"name": "Knife", "danger": "HIGH", "color": (0, 0, 255)},
    76: {"name": "Scissors", "danger": "MEDIUM", "color": (0, 165, 255)},
    39: {"name": "Baseball Bat", "danger": "MEDIUM", "color": (0, 165, 255)},
    # These can be suspicious in certain contexts:
    # 77: {"name": "Teddy Bear", "danger": "LOW"},  # Not a weapon, just example
}

# All weapon class IDs for quick lookup
WEAPON_CLASS_IDS = list(WEAPON_CLASSES.keys())

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================
PROCESS_EVERY_N_FRAMES = 1  # Skip frames for speed
AI_INPUT_SIZE = 640  # AI resolution
TARGET_DISPLAY_WIDTH = 960  # Display resolution

# ============================================================
# DETECTION SETTINGS
# ============================================================
GRID_SIZE = (4, 4)
DENSITY_LIMIT_PER_SECTOR = 4
MAX_LOG_ENTRIES = 10

# Alert thresholds
CROWD_DENSITY_ALERT_THRESHOLD = 15
STAMPEDE_VELOCITY_THRESHOLD = 15.0

# ============================================================
# ALERT SETTINGS
# ============================================================
ALERT_COOLDOWN = 30
ENABLE_TELEGRAM = True
ENABLE_DESKTOP = True

# ============================================================
# CAMERA SOURCE
# ============================================================
#CAMERA_SOURCE = "Videos/stamp.mp4"
CAMERA_SOURCE = 0  # Webcam
#CAMERA_SOURCE = "http://192.168.1.62:8080/video"  # IP Camera