# main.py - Crowd Intelligence System v6.0 (God's Eye Nexus)
# 
# ALL FEATURES:
# 1. STAMPEDE INTELLIGENCE v2.0 (FPS-normalized, counterflow, flow reversal)
# 2. REAL-TIME HEATMAP
# 3. WEAPON DETECTION (COCO + Custom gun model)
# 4. FIGHT/AGGRESSION DETECTION (MediaPipe Pose)
# 5. TARGET TRACKING (Enhanced with search & re-identification)
# 6. NIGHT VISION
# 7. DEMO MODE
# 8. ALERTS (Desktop + Telegram)
# 9. HTML REPORT
# 10. PROFESSIONAL DASHBOARD
# 11. GPU AUTO-DETECT + ADAPTIVE FPS
# 12. PEOPLE COUNT TREND (rise/drop)
#
# CONTROLS:
# Q - Quit | N - Night Vision | M - Heatmap | H - Save Heatmap
# R - Reset Target | T - Test Alerts | D - Demo Mode
# 1,2,3 - Demo Scenarios (press D first!) | C - Clear Heatmap
# B - Toggle Bounding Boxes | F - Toggle Fight Detection

import cv2
from ultralytics import YOLO
import numpy as np
import time
import os
import platform
import base64
from collections import deque

from config import *
from stampede_intel import StampedeDetector, DemoDataGenerator, HeatmapGenerator, PersonTracker
from alerts import AlertManager
from weapon_detector_v2 import WeaponDetectorV2
from fight_detector import FightDetector, MEDIAPIPE_AVAILABLE

# --- Global State ---
locked_target_id = None
mouse_click_pos = None
target_memory = {"last_seen": 0, "box": None, "stats": {}}
incident_log = []
people_count_history = deque(maxlen=90)  # Track people count for trend analysis
session_stats = {
    "start_time": time.time(),
    "max_crowd": 0,
    "weapons_found": 0,
    "fights_detected": 0,
    "max_stampede_risk": 0,
    "fps_list": [],
    "total_persons_tracked": 0
}


def add_log(message):
    global incident_log
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    incident_log.insert(0, entry)
    if len(incident_log) > MAX_LOG_ENTRIES:
        incident_log.pop()
    print(entry)


def mouse_handler(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)


def apply_night_vision(frame):
    gamma = 1.6
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(frame, table)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.medianBlur(enhanced, 3)


def generate_html_report(heatmap_base64=None, stampede_data=None, fight_data=None):
    """Creates a professional HTML report"""
    global session_stats, incident_log
    try:
        duration = int(time.time() - session_stats["start_time"])
        avg_fps = round(np.mean(session_stats["fps_list"]), 1) if session_stats["fps_list"] else "N/A"
        
        fights = fight_data.get('fights_detected', 0) if fight_data else 0
        base_risk = (session_stats["max_crowd"] * 3) + (session_stats["weapons_found"] * 30) + (fights * 15)
        stampede_risk = session_stats.get("max_stampede_risk", 0)
        risk_score = min(100, base_risk + stampede_risk)
        
        risk_level = "LOW"
        risk_color = "#00ff88"
        if risk_score > 30:
            risk_level = "ELEVATED"
            risk_color = "#ffcc00"
        if risk_score > 60:
            risk_level = "CRITICAL"
            risk_color = "#ff4444"

        heatmap_html = ""
        if heatmap_base64:
            heatmap_html = f"""
            <h2 style="margin-top:40px; font-size:18px; color:var(--primary);">FORENSIC HEATMAP VISUALIZATION</h2>
            <div class="card" style="text-align:center;">
                <img src="data:image/png;base64,{heatmap_base64}" style="max-width:100%; border-radius:10px; border: 2px solid #333;">
                <p style="font-size:12px; color:#666; margin-top:10px;">Cumulative Movement Density Map - Red indicates high traffic zones</p>
            </div>
            """

        stampede_html = f"""
        <h2 style="margin-top:40px; font-size:18px; color:var(--primary);">STAMPEDE INTELLIGENCE ANALYSIS</h2>
        <div class="grid">
            <div class="card">
                <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Peak Stampede Risk</span>
                <span class="stat-val" style="color: {'#ff4444' if stampede_data.get('max_risk', 0) > 50 else '#00ff88'};">{stampede_data.get('max_risk', 0)}%</span>
                <p style="font-size:12px; margin-top:10px;">Highest stampede probability detected.</p>
            </div>
            <div class="card">
                <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Weapons Detected</span>
                <span class="stat-val" style="color: {'#ff4444' if session_stats['weapons_found'] > 0 else '#00ff88'};">{session_stats['weapons_found']}</span>
                <p style="font-size:12px; margin-top:10px;">Total weapon alerts triggered.</p>
            </div>
            <div class="card">
                <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Persons Tracked</span>
                <span class="stat-val">{session_stats.get('total_persons_tracked', 0)}</span>
                <p style="font-size:12px; margin-top:10px;">Unique individuals identified.</p>
            </div>
            <div class="card">
                <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Fights Detected</span>
                <span class="stat-val" style="color: {'#ff4444' if fights > 0 else '#00ff88'};">{fights}</span>
                <p style="font-size:12px; margin-top:10px;">Physical altercation events detected.</p>
            </div>
        </div>
        """

        log_entries = ""
        for log in incident_log:
            color = "#888"
            if "STAMPEDE" in log:
                color = "#ff9500"
            if "WEAPON" in log:
                color = "#ff4444"
            if "FIGHT" in log:
                color = "#ff6600"
            if "TARGET" in log:
                color = "#00f2ff"
            log_entries += f'<div class="log-entry" style="color:{color};">{log}</div>'
        
        if not log_entries:
            log_entries = '<div class="log-entry">No incidents recorded during session.</div>'

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CROWD-INTEL | Surveillance Report</title>
    <style>
        :root {{ --primary: #00f2ff; --bg: #0a0a0c; --card: #16161a; --text: #e0e0e0; --accent: {risk_color}; }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: var(--bg); color: var(--text); padding: 40px; line-height: 1.6; }}
        .container {{ max-width: 1100px; margin: auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 40px; }}
        h1 {{ font-size: 28px; letter-spacing: 2px; color: var(--primary); }}
        .badge {{ background: var(--accent); color: #000; padding: 8px 20px; border-radius: 25px; font-weight: 800; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; margin-bottom: 30px; }}
        .card {{ background: var(--card); border: 1px solid #2a2a2a; padding: 25px; border-radius: 15px; transition: all 0.3s; }}
        .card:hover {{ border-color: var(--primary); transform: translateY(-3px); }}
        .stat-val {{ font-size: 48px; font-weight: 900; color: var(--primary); display: block; margin: 10px 0; }}
        .log-box {{ background: #0d0d0f; padding: 20px; border-radius: 12px; font-family: 'Consolas', monospace; font-size: 13px; max-height: 300px; overflow-y: auto; border: 1px solid #222; }}
        .log-entry {{ padding: 8px 0; border-bottom: 1px solid #1a1a1a; }}
        .risk-meter {{ height: 12px; background: #222; border-radius: 6px; margin-top: 15px; overflow: hidden; }}
        .risk-fill {{ height: 100%; width: {risk_score}%; background: linear-gradient(90deg, var(--accent), var(--accent)); border-radius: 6px; }}
        .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #222; color: #555; font-size: 12px; }}
        h2 {{ color: var(--primary); margin-bottom: 20px; font-size: 18px; text-transform: uppercase; letter-spacing: 1px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>CROWD-INTEL REPORT</h1>
                <p style="color: #666; margin-top: 5px;">Session: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            <div class="badge">{risk_level} RISK</div>
        </header>

        <div class="grid">
            <div class="card">
                <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Security Assessment</span>
                <span class="stat-val">{risk_score}%</span>
                <div class="risk-meter"><div class="risk-fill"></div></div>
                <p style="font-size:12px; margin-top:10px; color:#888;">Combined threat score based on all factors.</p>
            </div>
            <div class="card">
                <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Peak Crowd Count</span>
                <span class="stat-val">{session_stats['max_crowd']}</span>
                <p style="font-size:12px; margin-top:10px; color:#888;">Maximum concurrent people detected.</p>
            </div>
            <div class="card">
                <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Session Duration</span>
                <span class="stat-val">{duration // 60}m {duration % 60}s</span>
                <p style="font-size:12px; margin-top:10px; color:#888;">Total monitoring time.</p>
            </div>
        </div>

        {stampede_html}
        {heatmap_html}

        <h2 style="margin-top:40px;">INCIDENT CHRONOLOGY</h2>
        <div class="log-box">
            {log_entries}
        </div>

        <div class="footer">
            <p>ENGINE: YOLOv8 + Stampede Intelligence v2.0 | AVG FPS: {avg_fps}</p>
            <p>HARDWARE: {platform.processor()} | OS: {platform.system()} {platform.release()}</p>
            <p style="margin-top:10px;">&copy; 2026 Developed by Sanjay & Harlin AI</p>
        </div>
    </div>
</body>
</html>
"""
        
        report_path = "surveillance_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("\n" + "=" * 50)
        print("[SUCCESS] HTML REPORT GENERATED")
        print(f"[PATH] {os.path.abspath(report_path)}")
        print("=" * 50)
        
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    global locked_target_id, mouse_click_pos, target_memory, session_stats, people_count_history

    print("\n" + "=" * 60)
    print("  GOD'S EYE NEXUS - Crowd Intelligence System v6.0")
    print("  Stampede | Weapons | Fight Detection | Target Tracking")
    print("=" * 60)
    
    # --- GPU Auto-Detection ---
    device = 'cpu'
    if GPU_AUTO_DETECT:
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[GPU] CUDA detected: {gpu_name}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                print("[GPU] Apple MPS detected")
            else:
                print("[GPU] No GPU found, using CPU")
        except ImportError:
            print("[GPU] PyTorch not available for GPU check, using CPU")
    
    if GPU_ONLY and device == 'cpu':
        print("[ERROR] GPU_ONLY is enabled but no GPU was detected.")
        print("[INFO] Please run on a machine with CUDA GPU. Exiting.")
        return
    
    print(f"\n[CONFIG] Model: {MODEL_NAME} | Device: {device.upper()} | AI Size: {AI_INPUT_SIZE}px")
    print(f"[CONFIG] Frame Skip: {PROCESS_EVERY_N_FRAMES} (adaptive: {ADAPTIVE_FRAME_SKIP})")
    print(f"[CONFIG] Fight Detection: {'ON' if FIGHT_DETECTION_ENABLED and MEDIAPIPE_AVAILABLE else 'OFF'}")
    print(f"[CONFIG] Gun Model: {WEAPON_MODEL_PATH if os.path.exists(WEAPON_MODEL_PATH) else 'Not found (COCO-only)'}")
    print("\nCONTROLS:")
    print("  Q - Quit              N - Night Vision")
    print("  M - Heatmap Toggle    H - Save Heatmap")
    print("  R - Reset Target      T - Test Alerts")
    print("  D - Demo Mode         C - Clear Heatmap")
    print("  B - Bounding Boxes    F - Fight Detection")
    print("  1,2,3 - Demo Scenarios (press D first!)")
    print("=" * 60 + "\n")

    # Source setup
    source = CAMERA_SOURCE
    is_file = isinstance(source, str) and not source.startswith("http")
    is_stream = isinstance(source, str) and source.startswith("http")
    is_webcam = isinstance(source, int)

    # If source points to a directory, pick the first video file
    if is_file and os.path.isdir(source):
        try:
            exts = (".mp4", ".mkv", ".avi", ".mov", ".webm")
            entries = sorted(os.listdir(source))
            video_files = [f for f in entries if f.lower().endswith(exts)]
            if len(video_files) > 0:
                source = os.path.join(source, video_files[0])
                print(f"[INFO] Using video: {source}")
            else:
                print(f"[ERROR] No video files found in folder: {CAMERA_SOURCE}")
                source, is_file, is_webcam = 0, False, True
        except Exception as e:
            print(f"[ERROR] Failed to read folder '{CAMERA_SOURCE}': {e}")
            source, is_file, is_webcam = 0, False, True

    if is_file and not os.path.exists(source):
        print(f"[ERROR] Video not found: {source}, using webcam")
        source, is_file, is_webcam = 0, False, True

    # Load model
    print(f"[INFO] Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    if device != 'cpu':
        model.to(device)
        print(f"[INFO] Model moved to {device.upper()}")
    print("[INFO] Model loaded!")

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source!")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    
    # Window setup
    win_name = 'GOD\'S EYE NEXUS v6.0'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_handler)

    # Initialize all systems
    stampede_detector = StampedeDetector()
    stampede_result = {'risk_score': 0, 'alert_level': 0, 'alert_name': 'SAFE', 'components': {}}
    heatmap_gen = None
    person_tracker = PersonTracker(max_lost_frames=30)
    alert_manager = AlertManager(cooldown=ALERT_COOLDOWN, enable_telegram=ENABLE_TELEGRAM, enable_desktop=ENABLE_DESKTOP)
    
    # Weapon detector with custom gun model support
    weapon_detector = WeaponDetectorV2(
        alert_cooldown=5.0,
        proximity_threshold=150,
        min_weapon_size=15,
        max_weapon_size=500,
        require_person_nearby=False,
        weapon_model_path=WEAPON_MODEL_PATH
    )
    
    # Fight detector (MediaPipe Pose)
    fight_detector = None
    fight_result = {'fights': [], 'aggressive_persons': {}, 'fight_count': 0, 'new_alerts': []}
    if FIGHT_DETECTION_ENABLED and MEDIAPIPE_AVAILABLE:
        fight_detector = FightDetector(
            proximity_threshold=FIGHT_PROXIMITY_THRESHOLD,
            aggression_threshold=FIGHT_AGGRESSION_THRESHOLD,
            analysis_interval=FIGHT_ANALYSIS_INTERVAL
        )
        print("[INFO] Fight Detection: ENABLED")
    else:
        print("[INFO] Fight Detection: DISABLED")
    
    print("[INFO] All systems initialized!")

    # State variables
    night_mode = False
    heatmap_mode = False
    demo_mode = False
    demo_gen = None
    show_boxes = False           # Toggle person bounding boxes
    fight_mode = FIGHT_DETECTION_ENABLED and MEDIAPIPE_AVAILABLE
    frame_count = 0
    prev_time = time.time()
    last_valid_frame = None
    weapon_count = 0
    fight_count = 0
    fps = 30
    time_delta = 1.0 / 30.0
    process_every_n = PROCESS_EVERY_N_FRAMES  # Adaptive frame skip (mutable)
    child_alone_counts = {}
    kids_alone_count = 0
    
    # Cache for frame skipping
    cached_boxes = []
    cached_ids = []
    cached_clss = []
    cached_confs = []
    cached_person_boxes = []     # Only person bounding boxes

    sector_counts = np.zeros(GRID_SIZE)

    add_log("System initialized")

    # ============================================================
    # MAIN LOOP
    # ============================================================
    while True:
        ret, raw_frame = cap.read()
        frame_count += 1

        if not ret or raw_frame is None:
            if is_file:
                add_log("Video ended")
                break
            continue

        # FPS + Time Delta (for FPS-normalized calculations)
        curr_time = time.time()
        time_delta = curr_time - prev_time if (curr_time - prev_time) > 0 else 1.0 / 30.0
        fps = 1.0 / time_delta
        session_stats["fps_list"].append(fps)
        prev_time = curr_time
        
        # Adaptive frame skip — auto-adjust based on current FPS
        if ADAPTIVE_FRAME_SKIP and frame_count % 30 == 0:
            if fps < TARGET_FPS * 0.7:
                process_every_n = min(MAX_FRAME_SKIP, process_every_n + 1)
            elif fps > TARGET_FPS * 1.3 and process_every_n > MIN_FRAME_SKIP:
                process_every_n = max(MIN_FRAME_SKIP, process_every_n - 1)

        # Resize
        h_orig, w_orig = raw_frame.shape[:2]
        W, H = TARGET_DISPLAY_WIDTH, int(TARGET_DISPLAY_WIDTH * (h_orig / w_orig))
        frame = cv2.resize(raw_frame, (W, H))
        last_valid_frame = frame.copy()

        # Init heatmap
        if heatmap_gen is None:
            heatmap_gen = HeatmapGenerator(W, H, trail_length=20, decay_rate=0.85)

        ai_frame = frame.copy()
        if night_mode:
            ai_frame = apply_night_vision(ai_frame)

        current_dots = []
        current_positions = {}
        current_boxes = {}
        weapon_count = 0
        fight_count = 0
        sector_counts.fill(0)
        target_found_this_frame = False

        # --- DEMO MODE ---
        if demo_mode and demo_gen:
            positions = demo_gen.get_frame_data()
            current_positions = positions
            for pid, (x, y) in positions.items():
                current_dots.append((x, y))
                current_boxes[pid] = (x - 20, y - 50, x + 20, y + 10)
                cv2.circle(frame, (x, y), 18, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
                cv2.putText(frame, str(pid), (x - 6, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                sx = max(0, min(int(x / (W / GRID_SIZE[1])), GRID_SIZE[1] - 1))
                sy = max(0, min(int(y / (H / GRID_SIZE[0])), GRID_SIZE[0] - 1))
                sector_counts[sy, sx] += 1

        # --- REAL MODE ---
        else:
            run_ai = (frame_count % process_every_n == 0)

            if run_ai:
                ai_h = int(AI_INPUT_SIZE * (H / W))
                ai_small = cv2.resize(ai_frame, (AI_INPUT_SIZE, ai_h))
                results = model.track(ai_small, persist=True, verbose=False, imgsz=AI_INPUT_SIZE, conf=CONFIDENCE_THRESHOLD, tracker="bytetrack.yaml")

                if results[0].boxes.id is not None:
                    scale_x, scale_y = W / AI_INPUT_SIZE, H / ai_h
                    boxes_raw = results[0].boxes.xyxy.cpu().numpy()
                    cached_boxes = (boxes_raw * [scale_x, scale_y, scale_x, scale_y]).astype(int)
                    cached_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    cached_clss = results[0].boxes.cls.cpu().numpy().astype(int)
                    cached_confs = results[0].boxes.conf.cpu().numpy()
                else:
                    cached_boxes, cached_ids, cached_clss, cached_confs = [], [], [], []
                    cached_person_boxes = []

            if len(cached_boxes) > 0:
                tracked = person_tracker.update(cached_boxes, cached_ids, ai_frame)
                session_stats['total_persons_tracked'] = person_tracker.get_total_unique()
                
                # Build person-only boxes list (for weapon detector)
                cached_person_boxes = []
                for box, cls in zip(cached_boxes, cached_clss):
                    if int(cls) == PERSON_CLASS_ID:
                        cached_person_boxes.append(box)

                # Mouse click
                if mouse_click_pos:
                    mx, my = mouse_click_pos
                    for box, track_id, cls in zip(cached_boxes, cached_ids, cached_clss):
                        if cls == PERSON_CLASS_ID and box[0] < mx < box[2] and box[1] < my < box[3]:
                            locked_target_id = int(track_id)
                            target_memory["stats"]["status"] = "LOCKED"
                            add_log(f"TARGET LOCKED: ID {track_id}")
                            break
                    mouse_click_pos = None

                # Process persons
                for box, track_id, cls, conf in zip(cached_boxes, cached_ids, cached_clss, cached_confs):
                    track_id = int(track_id)

                    if cls == PERSON_CLASS_ID:
                        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                        current_dots.append((cx, cy))
                        current_positions[track_id] = (cx, cy)
                        current_boxes[track_id] = tuple(box)

                        sx = max(0, min(int(cx / (W / GRID_SIZE[1])), GRID_SIZE[1] - 1))
                        sy = max(0, min(int(cy / (H / GRID_SIZE[0])), GRID_SIZE[0] - 1))
                        sector_counts[sy, sx] += 1

                        speed = 0
                        if track_id in person_tracker.tracked_persons:
                            speed = round(person_tracker.tracked_persons[track_id]['speed'], 1)
                        
                        # Draw person bounding box if enabled
                        if show_boxes and track_id != locked_target_id:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 200, 0), 1)
                            cv2.putText(frame, f"#{track_id}", (box[0], box[1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

                        # Target tracking - when found
                        if track_id == locked_target_id:
                            target_found_this_frame = True
                            target_memory["last_seen"] = time.time()
                            target_memory["box"] = box
                            target_memory["stats"] = {
                                "id": track_id, 
                                "speed": speed, 
                                "pos": (cx, cy), 
                                "status": "LOCKED"
                            }

                            # Draw locked target box
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 3)
                            bw, bh = box[2] - box[0], box[3] - box[1]
                            corner_len = min(20, bw // 4, bh // 4)
                            
                            # Corner brackets
                            cv2.line(frame, (box[0], box[1]), (box[0] + corner_len, box[1]), (0, 255, 255), 3)
                            cv2.line(frame, (box[0], box[1]), (box[0], box[1] + corner_len), (0, 255, 255), 3)
                            cv2.line(frame, (box[2], box[1]), (box[2] - corner_len, box[1]), (0, 255, 255), 3)
                            cv2.line(frame, (box[2], box[1]), (box[2], box[1] + corner_len), (0, 255, 255), 3)
                            cv2.line(frame, (box[0], box[3]), (box[0] + corner_len, box[3]), (0, 255, 255), 3)
                            cv2.line(frame, (box[0], box[3]), (box[0], box[3] - corner_len), (0, 255, 255), 3)
                            cv2.line(frame, (box[2], box[3]), (box[2] - corner_len, box[3]), (0, 255, 255), 3)
                            cv2.line(frame, (box[2], box[3]), (box[2], box[3] - corner_len), (0, 255, 255), 3)
                            
                            # Label
                            cv2.rectangle(frame, (box[0], box[1] - 28), (box[0] + 130, box[1] - 3), (0, 255, 255), -1)
                            cv2.putText(frame, f"TARGET #{track_id}", (box[0] + 5, box[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # === WEAPON DETECTION (COCO + Custom Gun Model) ===
                weapon_result = weapon_detector.process_frame(
                    cached_boxes, 
                    cached_clss, 
                    cached_confs,
                    person_boxes=cached_person_boxes,  # Only person boxes (fixed bug)
                    frame=ai_frame  # Pass frame for custom gun model inference
                )
                if weapon_result['weapons']:
                    frame = weapon_detector.draw_detections(frame)
                    weapon_count = weapon_result['total_detected']
                    for weapon in weapon_result['new_alerts']:
                        session_stats["weapons_found"] += 1
                        add_log(f"WEAPON: {weapon['type']} ({weapon['danger']})")
                        alert_manager.weapon_alert(weapon['type'], frame=frame)
                
                # === FIGHT DETECTION (MediaPipe Pose) ===
                if fight_mode and fight_detector and len(current_positions) >= 2:
                    fight_result = fight_detector.process_frame(
                        ai_frame, current_boxes, current_positions
                    )
                    fight_count = fight_result['fight_count']
                    
                    if fight_result['fights']:
                        frame = fight_detector.draw_detections(frame, current_positions)
                    
                    for fight in fight_result.get('new_alerts', []):
                        session_stats['fights_detected'] += 1
                        severity = fight['severity']
                        add_log(f"FIGHT: Severity {int(severity*100)}% (#{fight['person1']} vs #{fight['person2']})")
                        alert_manager.fight_alert(severity, fight['person1'], fight['person2'], frame=frame)

                # === KIDS-ALONE DETECTION ===
                if len(current_boxes) >= 1:
                    heights = [b[3] - b[1] for b in current_boxes.values()]
                    if heights:
                        median_h = float(np.median(heights))
                        alone_radius = 120
                        kids_alone_count = 0
                        for tid, box in current_boxes.items():
                            h = box[3] - box[1]
                            is_child = h < median_h * 0.65
                            if not is_child:
                                child_alone_counts[tid] = max(0, child_alone_counts.get(tid, 0) - 1)
                                continue
                            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                            near = 0
                            for oid, (ox, oy) in current_positions.items():
                                if oid == tid:
                                    continue
                                if np.hypot(cx - ox, cy - oy) < alone_radius:
                                    near += 1
                            alone = near == 0
                            key = tid
                            if alone:
                                child_alone_counts[key] = child_alone_counts.get(key, 0) + 1
                                if child_alone_counts[key] >= 10:
                                    kids_alone_count += 1
                                    cv2.circle(frame, (cx, cy), 22, (255, 0, 255), 2)
                                    cv2.putText(frame, "CHILD", (box[0], box[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                    if child_alone_counts[key] == 10:
                                        add_log(f"CHILD ALONE: #{tid}")
                                        alert_manager.custom_alert("CHILD ALONE", f"Suspected child alone detected (ID #{tid})", alert_type="child", frame=frame)
                            else:
                                child_alone_counts[key] = max(0, child_alone_counts.get(key, 0) - 2)
        # Stats
        if len(current_dots) > session_stats["max_crowd"]:
            session_stats["max_crowd"] = len(current_dots)

        # Heatmap — always update (so trails decay even when no one visible)
        if heatmap_gen:
            heatmap_gen.update(current_positions, current_boxes)

        # Stampede (pass time_delta for FPS-normalized velocities)
        if len(current_positions) >= 2:
            stampede_result = stampede_detector.update(
                current_positions, sector_counts, (W, H), time_delta=time_delta
            )
            if stampede_result['risk_score'] > session_stats.get('max_stampede_risk', 0):
                session_stats['max_stampede_risk'] = stampede_result['risk_score']
            if stampede_result['alert_level'] >= 2:
                if len(incident_log) == 0 or "STAMPEDE" not in incident_log[0]:
                    add_log(f"STAMPEDE RISK: {stampede_result['risk_score']}%")
                alert_manager.stampede_alert(stampede_result['risk_score'], stampede_result['alert_level'], frame=frame)

        # Crowd alert + people count trend
        people_count = len(current_dots)
        people_count_history.append(people_count)
        if people_count > CROWD_DENSITY_ALERT_THRESHOLD:
            alert_manager.crowd_density_alert(people_count, CROWD_DENSITY_ALERT_THRESHOLD, frame=frame)
        
        # Calculate people count trend
        people_trend = "->"  # stable
        if len(people_count_history) > 15:
            recent_avg = np.mean(list(people_count_history)[-10:])
            older_avg = np.mean(list(people_count_history)[:10])
            if older_avg > 0:
                if recent_avg > older_avg * 1.25:
                    people_trend = "UP"
                elif recent_avg < older_avg * 0.75:
                    people_trend = "DN"
                else:
                    people_trend = "->"

        # ============================================================
        # ENHANCED TARGET TRACKING - Lost Target Handling
        # ============================================================
        if locked_target_id and not target_found_this_frame and not demo_mode:
            time_since_seen = time.time() - target_memory["last_seen"]
            b = target_memory.get("box")
            
            if time_since_seen < 5.0:  # Search for 5 seconds before giving up
                target_memory["stats"]["status"] = "SEARCHING"
                
                if b is not None:
                    # Calculate predicted position based on last known velocity
                    pred_box = list(b)
                    if locked_target_id in person_tracker.tracked_persons:
                        person = person_tracker.tracked_persons[locked_target_id]
                        vx, vy = person.get('velocity', (0, 0))
                        
                        # Predict new position
                        frames_lost = int(time_since_seen * fps) if fps > 0 else 1
                        pred_offset_x = int(vx * frames_lost * 0.3)  # Decay prediction
                        pred_offset_y = int(vy * frames_lost * 0.3)
                        
                        # Predicted box (clamped to frame)
                        pred_box = [
                            max(0, b[0] + pred_offset_x),
                            max(0, b[1] + pred_offset_y),
                            min(W, b[2] + pred_offset_x),
                            min(H, b[3] + pred_offset_y)
                        ]
                    
                    # Draw searching animation (pulsing effect)
                    pulse = int((time.time() * 4) % 2)  # Alternates 0/1
                    search_color = (0, 165, 255) if pulse else (0, 200, 255)
                    thickness = 2 if pulse else 3
                    
                    # Draw predicted position box
                    cv2.rectangle(frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), search_color, thickness)
                    
                    # Draw corner brackets
                    bw, bh = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]
                    corner_len = min(15, bw // 4, bh // 4)
                    corners = [
                        ((pred_box[0], pred_box[1]), (1, 1)),
                        ((pred_box[2], pred_box[1]), (-1, 1)),
                        ((pred_box[0], pred_box[3]), (1, -1)),
                        ((pred_box[2], pred_box[3]), (-1, -1))
                    ]
                    for (pt, (dx, dy)) in corners:
                        cv2.line(frame, pt, (pt[0] + dx * corner_len, pt[1]), search_color, 2)
                        cv2.line(frame, pt, (pt[0], pt[1] + dy * corner_len), search_color, 2)
                    
                    # Searching label with countdown
                    remaining = max(0, 5.0 - time_since_seen)
                    label = f"SEARCHING... ({remaining:.1f}s)"
                    label_w = 170
                    cv2.rectangle(frame, (pred_box[0], pred_box[1] - 28), (pred_box[0] + label_w, pred_box[1] - 3), search_color, -1)
                    cv2.putText(frame, label, (pred_box[0] + 5, pred_box[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                    
                    # Draw search radius circle
                    center_x = (pred_box[0] + pred_box[2]) // 2
                    center_y = (pred_box[1] + pred_box[3]) // 2
                    search_radius = int(40 + time_since_seen * 15)  # Expanding search radius
                    cv2.circle(frame, (center_x, center_y), search_radius, search_color, 1)
                    
                    # Inner pulsing circle
                    inner_radius = int(20 + (time.time() % 1) * 20)
                    cv2.circle(frame, (center_x, center_y), inner_radius, search_color, 1)
                    
                    # Direction arrow if we have velocity
                    if locked_target_id in person_tracker.tracked_persons:
                        person = person_tracker.tracked_persons[locked_target_id]
                        vx, vy = person.get('velocity', (0, 0))
                        if abs(vx) > 2 or abs(vy) > 2:
                            arrow_length = min(50, int(np.sqrt(vx**2 + vy**2) * 3))
                            if arrow_length > 10:
                                arrow_end = (center_x + int(vx * 3), center_y + int(vy * 3))
                                cv2.arrowedLine(frame, (center_x, center_y), arrow_end, (255, 255, 0), 2, tipLength=0.4)
                                cv2.putText(frame, "PREDICTED", (arrow_end[0] - 30, arrow_end[1] - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            
            else:
                # Lost for too long - give up
                add_log(f"TARGET {locked_target_id} LOST after 5s search")
                target_memory["stats"]["status"] = "LOST"
                locked_target_id = None
        
        # Check for re-identification
        elif locked_target_id and target_found_this_frame:
            # Target found again!
            prev_status = target_memory["stats"].get("status", "")
            if prev_status == "SEARCHING":
                add_log(f"TARGET {locked_target_id} RE-ACQUIRED!")
                target_memory["stats"]["status"] = "LOCKED"
            
            # Check if person was re-identified by tracker
            if locked_target_id in person_tracker.tracked_persons:
                person = person_tracker.tracked_persons[locked_target_id]
                if person.get('reidentified', False):
                    add_log(f"TARGET {locked_target_id} RE-IDENTIFIED!")
                    person['reidentified'] = False

        # Display frame
        display_frame = frame.copy()
        if heatmap_mode and heatmap_gen:
            display_frame = heatmap_gen.get_heatmap_overlay(frame, alpha=0.5)
            cv2.putText(display_frame, "HEATMAP ON", (W - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if night_mode:
            display_frame = apply_night_vision(display_frame)

        # Weapon banner
        if weapon_detector.active_weapons:
            display_frame = weapon_detector.draw_alert_banner(display_frame)

        # Fight alert banner
        if fight_detector and fight_detector.active_fights:
            display_frame = fight_detector.draw_alert_banner(display_frame)

        # Dots
        for d in current_dots:
            cv2.circle(display_frame, d, 5, (0, 255, 0), -1)

        # Grid
        for gy in range(GRID_SIZE[0]):
            for gx in range(GRID_SIZE[1]):
                x1, y1 = int(gx * W / GRID_SIZE[1]), int(gy * H / GRID_SIZE[0])
                x2, y2 = int((gx + 1) * W / GRID_SIZE[1]), int((gy + 1) * H / GRID_SIZE[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (40, 40, 40), 1)
                if sector_counts[gy, gx] >= DENSITY_LIMIT_PER_SECTOR:
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)

        # Stampede warning banner
        if stampede_result['alert_level'] >= 2:
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, 50), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            cv2.putText(display_frame, f"STAMPEDE WARNING: {stampede_result['risk_score']}%",
                        (W // 2 - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ========================================
        # PROFESSIONAL DASHBOARD
        # ========================================
        DASH_W = 380
        dash = np.zeros((H, DASH_W, 3), dtype=np.uint8)
        
        # Background gradient
        for i in range(H):
            shade = int(15 + (i / H) * 10)
            dash[i, :] = (shade, shade, shade + 5)

        # Header
        cv2.rectangle(dash, (0, 0), (DASH_W, 55), (25, 25, 30), -1)
        cv2.putText(dash, "CROWD INTELLIGENCE", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(dash, "Real-Time Surveillance System", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        cv2.circle(dash, (DASH_W - 25, 28), 8, (0, 255, 0), -1)
        cv2.circle(dash, (DASH_W - 25, 28), 8, (100, 100, 100), 1)

        y = 65

        # === LIVE STATISTICS ===
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 75), (30, 30, 35), -1)
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 75), (50, 50, 55), 1)
        cv2.putText(dash, "LIVE STATISTICS", (20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # People count with trend
        cv2.putText(dash, f"People: {people_count}", (20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        trend_color = (0, 200, 0) if people_trend == "->" else (0, 165, 255) if people_trend == "UP" else (255, 100, 100)
        cv2.putText(dash, people_trend, (130, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, trend_color, 1)
        cv2.putText(dash, f"(Peak: {session_stats['max_crowd']})", (175, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # FPS, Weapons, Fights
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(dash, f"FPS: {int(fps)}", (20, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1)
        
        w_color = (0, 0, 255) if weapon_count > 0 else (0, 255, 0)
        cv2.putText(dash, f"Weapons: {weapon_count}", (100, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, w_color, 1)
        
        f_color = (0, 0, 255) if fight_count > 0 else (0, 255, 0)
        cv2.putText(dash, f"Fights: {fight_count}", (220, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, f_color, 1)
        kid_color = (0, 0, 255) if kids_alone_count > 0 else (0, 255, 0)
        cv2.putText(dash, f"Kids: {kids_alone_count}", (300, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, kid_color, 1)

        y += 85

        # === STAMPEDE INTELLIGENCE ===
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 130), (30, 30, 35), -1)
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 130), (50, 50, 55), 1)
        cv2.putText(dash, "STAMPEDE INTELLIGENCE", (20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        risk = stampede_result['risk_score']
        alert_colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
        alert_names = ['SAFE', 'CAUTION', 'WARNING', 'CRITICAL']
        alert_level = stampede_result['alert_level']
        alert_color = alert_colors[alert_level]

        # Big percentage and status
        cv2.putText(dash, f"{risk}%", (20, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, alert_color, 2)
        cv2.putText(dash, alert_names[alert_level], (95, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

        # Risk bar
        bar_w = DASH_W - 50
        bar_y_pos = y + 58
        cv2.rectangle(dash, (20, bar_y_pos), (20 + bar_w, bar_y_pos + 10), (40, 40, 45), -1)
        risk_fill = int((risk / 100) * bar_w)
        if risk_fill > 0:
            cv2.rectangle(dash, (20, bar_y_pos), (20 + risk_fill, bar_y_pos + 10), alert_color, -1)
        
        # Threshold markers
        for thresh in [30, 60]:
            tx = 20 + int((thresh / 100) * bar_w)
            cv2.line(dash, (tx, bar_y_pos), (tx, bar_y_pos + 10), (80, 80, 80), 1)

        # Component vertical bars
        components = stampede_result.get('components', {})
        comp_names = ['COHR', 'ACEL', 'SPD', 'SPKE', 'EDGE', 'CFLW', 'RVRS']
        comp_keys = ['coherence', 'acceleration', 'avg_speed', 'spike', 'edge', 'counterflow', 'flow_reversal']
        
        comp_start_y = y + 78
        comp_spacing = 49
        
        for i, (name, key) in enumerate(zip(comp_names, comp_keys)):
            value = components.get(key, 0)
            cx = 25 + (i * comp_spacing)
            
            # Component name
            cv2.putText(dash, name, (cx - 5, comp_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            
            # Vertical bar background
            bar_x = cx + 12
            bar_top = comp_start_y + 5
            bar_height = 28
            bar_width = 10
            cv2.rectangle(dash, (bar_x, bar_top), (bar_x + bar_width, bar_top + bar_height), (40, 40, 45), -1)
            cv2.rectangle(dash, (bar_x, bar_top), (bar_x + bar_width, bar_top + bar_height), (60, 60, 65), 1)
            
            # Filled portion (from bottom up)
            fill_height = int(value * bar_height)
            if fill_height > 0:
                comp_color = (0, 200, 200) if value < 0.5 else (0, 255, 255) if value < 0.8 else (0, 100, 255)
                cv2.rectangle(dash, (bar_x + 1, bar_top + bar_height - fill_height), 
                             (bar_x + bar_width - 1, bar_top + bar_height - 1), comp_color, -1)
            
            # Value below bar
            cv2.putText(dash, f"{value:.1f}", (cx, comp_start_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

        y += 140

        # === ZONE DENSITY MAP ===
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 85), (30, 30, 35), -1)
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 85), (50, 50, 55), 1)
        cv2.putText(dash, "ZONE DENSITY MAP", (20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        grid_x, grid_y = 20, y + 28
        grid_w, grid_h = DASH_W - 50, 48
        cell_w, cell_h = grid_w // GRID_SIZE[1], grid_h // GRID_SIZE[0]
        max_density = max(sector_counts.max(), 1)

        for gy_i in range(GRID_SIZE[0]):
            for gx_i in range(GRID_SIZE[1]):
                cx1, cy1 = grid_x + gx_i * cell_w, grid_y + gy_i * cell_h
                cx2, cy2 = cx1 + cell_w - 2, cy1 + cell_h - 2
                density = sector_counts[gy_i, gx_i]
                ratio = density / max(max_density, DENSITY_LIMIT_PER_SECTOR)
                
                if ratio < 0.3:
                    color = (0, 100, 0)
                elif ratio < 0.6:
                    color = (0, 180, 0)
                elif ratio < 0.8:
                    color = (0, 200, 200)
                else:
                    color = (0, 0, 200)
                
                cv2.rectangle(dash, (cx1, cy1), (cx2, cy2), color, -1)
                cv2.rectangle(dash, (cx1, cy1), (cx2, cy2), (50, 50, 55), 1)
                if density > 0:
                    cv2.putText(dash, str(int(density)), (cx1 + cell_w // 3, cy1 + cell_h // 2 + 4), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        y += 95

        # === TARGET TRACKING ===
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 70), (30, 30, 35), -1)
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 70), (50, 50, 55), 1)
        cv2.putText(dash, "TARGET TRACKING", (20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if locked_target_id:
            s = target_memory.get("stats", {})
            status = s.get("status", "LOCKED")
            
            # Status indicator color
            if status == "LOCKED":
                status_color = (0, 255, 255)  # Cyan
            elif status == "SEARCHING":
                # Pulsing orange for searching
                pulse = int((time.time() * 4) % 2)
                status_color = (0, 165, 255) if pulse else (0, 200, 255)
            else:
                status_color = (0, 0, 255)  # Red for lost
            
            cv2.circle(dash, (DASH_W - 25, y + 32), 8, status_color, -1)
            
            # Info
            cv2.putText(dash, f"ID: {s.get('id', 'N/A')}", (20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(dash, f"Speed: {s.get('speed', 0):.1f}", (90, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Status badge
            cv2.rectangle(dash, (180, y + 28), (260, y + 45), status_color, -1)
            cv2.putText(dash, status, (185, y + 41), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            
            # Position
            pos = s.get('pos', (0, 0))
            cv2.putText(dash, f"Position: ({pos[0]}, {pos[1]})", (20, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            # Search indicator
            if status == "SEARCHING":
                remaining = max(0, 5.0 - (time.time() - target_memory["last_seen"]))
                cv2.putText(dash, f"Search: {remaining:.1f}s", (180, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
        else:
            cv2.circle(dash, (DASH_W - 25, y + 32), 8, (60, 60, 60), -1)
            cv2.putText(dash, "No target selected", (20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            cv2.putText(dash, "Click on a person to track", (20, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        y += 80

        # === MODE INDICATORS ===
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + 30), (30, 30, 35), -1)
        mode_x = 20
        
        for mode_name, is_on in [("DEMO", demo_mode), ("NIGHT", night_mode), ("HEAT", heatmap_mode), ("BBOX", show_boxes), ("FIGHT", fight_mode)]:
            color = (0, 255, 255) if is_on else (50, 50, 55)
            text_color = (0, 0, 0) if is_on else (80, 80, 80)
            cv2.rectangle(dash, (mode_x, y + 5), (mode_x + 50, y + 25), color, -1 if is_on else 1)
            cv2.putText(dash, mode_name, (mode_x + 5, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
            mode_x += 58

        # Source indicator
        if is_file and total_frames > 0:
            progress = int((frame_count / total_frames) * 100)
            cv2.putText(dash, f"FILE {progress}%", (mode_x + 15, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        elif is_stream:
            cv2.putText(dash, "STREAM", (mode_x + 15, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)
        else:
            cv2.putText(dash, "WEBCAM", (mode_x + 15, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

        y += 40

        # === ACTIVITY LOG ===
        log_height = H - y - 35
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + log_height), (30, 30, 35), -1)
        cv2.rectangle(dash, (10, y), (DASH_W - 10, y + log_height), (50, 50, 55), 1)
        cv2.putText(dash, "ACTIVITY LOG", (20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        log_y = y + 35
        max_logs = (log_height - 40) // 16
        for i, log in enumerate(incident_log[:max_logs]):
            color = (140, 140, 140)
            if "STAMPEDE" in log:
                color = (0, 165, 255)
            if "WEAPON" in log:
                color = (0, 0, 255)
            if "FIGHT" in log:
                color = (0, 80, 255)
            if "TARGET" in log:
                color = (0, 255, 255)
            if "RE-ACQUIRED" in log or "RE-IDENTIFIED" in log:
                color = (0, 255, 0)
            if "LOST" in log:
                color = (0, 100, 255)
            if "Demo" in log:
                color = (255, 255, 0)
            cv2.putText(dash, log[:42], (15, log_y + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

        # === FOOTER ===
        cv2.line(dash, (10, H - 28), (DASH_W - 10, H - 28), (50, 50, 55), 1)
        runtime = int(time.time() - session_stats['start_time'])
        cv2.putText(dash, f"Runtime: {runtime // 60}m {runtime % 60}s", (15, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
        cv2.putText(dash, "B=Box F=Fight", (DASH_W - 105, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

        # Bottom tip on main frame
        if locked_target_id:
            status = target_memory.get("stats", {}).get("status", "")
            if status == "SEARCHING":
                tip = "SEARCHING for target... | Press R to cancel"
            else:
                tip = "Target LOCKED | Press R to reset"
        else:
            tip = "Click on any person to start tracking"
        cv2.putText(display_frame, tip, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Combine and show
        cv2.imshow(win_name, np.hstack((display_frame, dash)))

        # === KEYBOARD CONTROLS ===
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            add_log("User quit - Generating report...")
            break
        elif key == ord('n'):
            night_mode = not night_mode
            add_log(f"Night Vision: {'ON' if night_mode else 'OFF'}")
        elif key == ord('m'):
            heatmap_mode = not heatmap_mode
            add_log(f"Heatmap: {'ON' if heatmap_mode else 'OFF'}")
        elif key == ord('r'):
            if locked_target_id:
                add_log(f"Target {locked_target_id} RESET by user")
            locked_target_id = None
            target_memory["stats"]["status"] = "NONE"
        elif key == ord('c'):
            if heatmap_gen:
                heatmap_gen.reset()
                add_log("Heatmap CLEARED")
        elif key == ord('t'):
            add_log("Testing alerts...")
            alert_manager.test_alerts()
        elif key == ord('d'):
            demo_mode = not demo_mode
            if demo_mode:
                demo_gen = DemoDataGenerator(W, H, num_people=15)
                add_log("DEMO MODE: ON - Press 1/2/3 for scenarios")
            else:
                demo_gen = None
                add_log("DEMO MODE: OFF")
        elif key == ord('1') and demo_mode and demo_gen:
            demo_gen.set_mode('normal')
            add_log("Demo: Normal crowd movement")
        elif key == ord('2') and demo_mode and demo_gen:
            demo_gen.set_mode('gathering')
            add_log("Demo: Crowd gathering")
        elif key == ord('3') and demo_mode and demo_gen:
            demo_gen.set_mode('stampede')
            add_log("Demo: STAMPEDE SIMULATION!")
        elif key == ord('h'):
            if heatmap_gen and last_valid_frame is not None:
                timestamp = int(time.time())
                save_folder = "Pictures"
                os.makedirs(save_folder, exist_ok=True)
                
                # Save 3 versions
                cv2.imwrite(os.path.join(save_folder, f"heatmap_overlay_{timestamp}.png"),
                            heatmap_gen.get_heatmap_with_background(last_valid_frame, alpha=0.6))
                cv2.imwrite(os.path.join(save_folder, f"heatmap_pure_{timestamp}.png"),
                            heatmap_gen.get_heatmap_only())
                cv2.imwrite(os.path.join(save_folder, f"heatmap_gray_{timestamp}.png"),
                            heatmap_gen.get_heatmap_overlay(last_valid_frame, alpha=0.6, grayscale_bg=True))
                add_log("Saved 3 heatmaps to Pictures/")
                print(f"[SAVED] 3 heatmaps to Pictures/ folder")
        elif key == ord('b'):
            show_boxes = not show_boxes
            add_log(f"Bounding Boxes: {'ON' if show_boxes else 'OFF'}")
        elif key == ord('f'):
            fight_mode = not fight_mode
            add_log(f"Fight Detection: {'ON' if fight_mode else 'OFF'}")

    # ============================================================
    # SESSION END - Generate Report
    # ============================================================
    print("\n" + "=" * 50)
    print("[INFO] Session ending - Generating HTML report...")
    print("=" * 50)
    
    # Encode heatmap for report
    heatmap_base64 = None
    if heatmap_gen and last_valid_frame is not None:
        try:
            heatmap_img = heatmap_gen.get_heatmap_overlay(last_valid_frame, alpha=0.6)
            _, buffer = cv2.imencode('.png', heatmap_img)
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            print("[INFO] Heatmap encoded successfully")
        except Exception as e:
            print(f"[WARN] Heatmap encoding failed: {e}")

    # Generate HTML report
    generate_html_report(
        heatmap_base64=heatmap_base64, 
        stampede_data={'max_risk': session_stats.get('max_stampede_risk', 0)},
        fight_data={'fights_detected': session_stats.get('fights_detected', 0)}
    )

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print session summary
    avg_fps = np.mean(session_stats["fps_list"]) if session_stats["fps_list"] else 0
    print(f"\n{'=' * 50}")
    print("SESSION SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Runtime:          {int(time.time() - session_stats['start_time'])} seconds")
    print(f"  Average FPS:      {avg_fps:.1f}")
    print(f"  Max Crowd:        {session_stats['max_crowd']} people")
    print(f"  Weapons Found:    {session_stats['weapons_found']}")
    print(f"  Fights Detected:  {session_stats['fights_detected']}")
    print(f"  Max Stampede:     {session_stats['max_stampede_risk']}%")
    print(f"  Persons Tracked:  {session_stats['total_persons_tracked']}")
    print(f"{'=' * 50}")
    print("[INFO] Session ended successfully!")
    print(f"[INFO] Report saved: surveillance_report.html")


if __name__ == "__main__":
    main()
