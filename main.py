# Harlin's Industrial Crowd Intelligence (PHASE 21 - THE "ZERO-ERROR" FINALE)
# 
# THE "GP-SAVER" UPDATES:
# 1. FIXED NameError: Ensured 'cap' is defined before any properties are set.
# 2. PANIC GUARD: Added checks to prevent crashes if the IP camera is offline.
# 3. EMBEDDED HEATMAP: Still baked into the HTML report for that A++ flex.
# 4. TACTICAL HUD: All systems (Suspect tracking, Weapons, Night Vision) optimized.

import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time
import os
import platform
import base64

# --- Configuration ---
MODEL_NAME = 'yolov8s.pt' 
CONFIDENCE_THRESHOLD = 0.30 
WEAPON_CONF_THRESHOLD = 0.25
PERSON_CLASS_ID = 0
KNIFE_CLASS_ID = 43 

# --- CAMERA SOURCE SELECTION ---
# Webcam: 0 | IP Cam: "http://..." | File: "video.mp4"
CAMERA_SOURCE = "http://192.168.1.62:8080/video" 

# --- Intelligence Config ---
GRID_SIZE = (4, 4)
STAMPEDE_VELOCITY_THRESHOLD = 15.0
DENSITY_LIMIT_PER_SECTOR = 4
MAX_LOG_ENTRIES = 9 
TARGET_DISPLAY_WIDTH = 960 

# --- Global Tracking State ---
locked_target_id = None
mouse_click_pos = None
target_memory = {"last_seen": 0, "box": None, "stats": {}}
incident_log = []
session_stats = {
    "start_time": time.time(), 
    "max_crowd": 0, 
    "weapons_found": 0, 
    "total_detections": 0,
    "stampede_triggers": 0,
    "fps_list": []
}

def add_log(message):
    global incident_log
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    incident_log.insert(0, entry)
    if len(incident_log) > MAX_LOG_ENTRIES: incident_log.pop()
    print(entry)

def mouse_handler(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)

def apply_night_vision_3(frame):
    gamma = 1.6
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(frame, table)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12,12))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.medianBlur(enhanced, 3)

def generate_html_report(heatmap_base64=None):
    """ Creates a professional A++ report file with embedded Heatmap """
    global session_stats, incident_log
    try:
        duration = int(time.time() - session_stats["start_time"])
        avg_fps = round(np.mean(session_stats["fps_list"]), 1) if session_stats["fps_list"] else "N/A"
        
        # Risk Score Calculation
        risk_score = min(100, (session_stats["max_crowd"] * 5) + (session_stats["weapons_found"] * 30) + (session_stats["stampede_triggers"] * 10))
        risk_level = "LOW"
        risk_color = "#00ff88"
        if risk_score > 30: risk_level = "ELEVATED"; risk_color = "#ffcc00"
        if risk_score > 60: risk_level = "CRITICAL"; risk_color = "#ff4444"

        heatmap_html = ""
        if heatmap_base64:
            heatmap_html = f"""
            <h2 style="margin-top:40px; font-size:18px; color:var(--primary);">FORENSIC VISUALIZATION</h2>
            <div class="card" style="text-align:center;">
                <img src="data:image/png;base64,{heatmap_base64}" style="max-width:100%; border-radius:10px; border: 2px solid #333;">
                <p style="font-size:12px; color:#666; margin-top:10px;">Cumulative Activity Distribution Map (Session End)</p>
            </div>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>CROWD-INTEL | Surveillance Audit</title>
            <style>
                :root {{ --primary: #00f2ff; --bg: #0a0a0c; --card: #16161a; --text: #e0e0e0; --accent: {risk_color}; }}
                body {{ font-family: 'Inter', 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 40px; line-height: 1.6; }}
                .container {{ max-width: 1000px; margin: auto; }}
                header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 40px; }}
                .badge {{ background: var(--accent); color: #000; padding: 5px 15px; border-radius: 20px; font-weight: 800; font-size: 12px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
                .card {{ background: var(--card); border: 1px solid #222; padding: 25px; border-radius: 15px; transition: transform 0.3s; }}
                .card:hover {{ transform: translateY(-5px); border-color: var(--primary); }}
                .stat-val {{ font-size: 42px; font-weight: 900; color: var(--primary); display: block; margin-top: 10px; }}
                .log-box {{ background: #000; padding: 15px; border-radius: 10px; font-family: 'Consolas', monospace; font-size: 13px; max-height: 400px; overflow-y: auto; border: 1px solid #333; }}
                .log-entry {{ padding: 8px 0; border-bottom: 1px solid #111; color: #888; }}
                .log-entry span {{ color: var(--primary); }}
                .sys-info {{ font-size: 12px; color: #555; margin-top: 50px; text-align: center; }}
                .risk-meter {{ height: 10px; background: #333; border-radius: 5px; margin-top: 15px; overflow: hidden; }}
                .risk-fill {{ height: 100%; width: {risk_score}%; background: var(--accent); }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div>
                        <h1 style="margin:0; font-size: 24px; letter-spacing: 2px;">CROWD-INTEL AUDIT</h1>
                        <p style="margin:5px 0; color: #666;">Session ID: {int(time.time())} | {time.ctime()}</p>
                    </div>
                    <div class="badge">{risk_level} RISK</div>
                </header>

                <div class="grid">
                    <div class="card">
                        <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Security Assessment</span>
                        <span class="stat-val">{risk_score}%</span>
                        <div class="risk-meter"><div class="risk-fill"></div></div>
                        <p style="font-size:12px; margin-top:10px;">Threat Score based on weapons, density, and movement.</p>
                    </div>
                    <div class="card">
                        <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Peak Crowd Density</span>
                        <span class="stat-val">{session_stats['max_crowd']}</span>
                        <p style="font-size:12px; margin-top:10px;">Maximum concurrent head-counts detected.</p>
                    </div>
                    <div class="card">
                        <span style="color:#666; text-transform: uppercase; font-size: 11px; font-weight:700;">Intelligence Data</span>
                        <span class="stat-val">{session_stats['total_detections']}</span>
                        <p style="font-size:12px; margin-top:10px;">Total unique behavioral data points logged.</p>
                    </div>
                </div>

                {heatmap_html}

                <h2 style="margin-top:40px; font-size:18px; color:var(--primary);">INCIDENT CHRONOLOGY</h2>
                <div class="log-box">
                    {''.join([f'<div class="log-entry"><span>{log.split("]")[0]}]</span> {log.split("]")[1]}</div>' for log in incident_log])}
                </div>

                <div class="sys-info">
                    <p>ENGINE: YOLOv8-SURVEILLANCE | HARDWARE: {platform.processor()} | ENV: {platform.system()} {platform.release()}</p>
                    <p>AVERAGE PERFORMANCE: {avg_fps} FPS</p>
                    <p>&copy; 2026 Developed by Sanjay (Brahh) & Harlin AI</p>
                </div>
            </div>
        </body>
        </html>
        """
        with open("final_surveillance_report.html", "w", encoding="utf-8") as f: f.write(html_content)
        print("\n" + "="*50)
        print("[SUCCESS] ULTIMATE AUDIT REPORT GENERATED (WITH HEATMAP)")
        print(f"Path: {os.path.abspath('final_surveillance_report.html')}")
        print("="*50)
    except Exception as e:
        print(f"Report generation failed: {e}")

def main():
    global locked_target_id, mouse_click_pos, target_memory, session_stats
    source = str(CAMERA_SOURCE).strip()
    if source.isdigit(): source = int(source)

    print(f"Harlin's Intelligence System starting... Source: {source}")
    model = YOLO(MODEL_NAME)
    
    # --- FIX: INITIALIZE CAP BEFORE USING IT ---
    
    #cap = cv2.VideoCapture("videos/public.mp4")
    #cap = cv2.VideoCapture(source)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"FAILED to connect to source '{source}', brahh. Check your network/path.")
        return

    # Now that we know cap exists and is opened, we can set properties
    is_stream = isinstance(source, str) and (source.startswith("http") or source.startswith("rtsp"))
    if is_stream:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    win_name = 'SURVEILLANCE_INTELLIGENCE_V21'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_handler)
    
    prev_positions = {}
    sector_counts = np.zeros(GRID_SIZE)
    all_tracked_points = []
    night_mode = False
    DASH_W = 320 
    prev_time = time.time()
    last_valid_frame = None

    while True:
        if is_stream:
            for _ in range(3): cap.grab() 
        
        ret, raw_frame = cap.retrieve() if is_stream else cap.read()
        
        if not ret: 
            if is_stream:
                time.sleep(0.5)
                continue
            else:
                break 

        # FPS Tracking
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        session_stats["fps_list"].append(fps)
        prev_time = curr_time

        # Resize & Copy
        h_orig, w_orig = raw_frame.shape[:2]
        W, H = TARGET_DISPLAY_WIDTH, int(TARGET_DISPLAY_WIDTH * (h_orig/w_orig))
        frame = cv2.resize(raw_frame, (W, H))
        last_valid_frame = frame.copy()

        ai_frame = frame.copy()
        if night_mode: ai_frame = apply_night_vision_3(ai_frame)

        results = model.track(ai_frame, persist=True, verbose=False, imgsz=640, conf=CONFIDENCE_THRESHOLD)
        
        current_dots = []
        weapon_count = 0
        sector_counts.fill(0)
        target_found_this_frame = False

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            if mouse_click_pos:
                mx, my = mouse_click_pos
                for box, track_id, cls in zip(boxes, ids, clss):
                    if cls == PERSON_CLASS_ID:
                        if box[0] < mx < box[2] and box[1] < my < box[3]:
                            locked_target_id = track_id
                            add_log(f"MANUAL LOCK: SUBJECT {track_id}")
                            break
                mouse_click_pos = None

            for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
                if cls == KNIFE_CLASS_ID and conf > WEAPON_CONF_THRESHOLD:
                    weapon_count += 1
                    session_stats["weapons_found"] += 1
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    if weapon_count == 1: add_log("CRITICAL: WEAPON DETECTED")
                    continue

                if cls == PERSON_CLASS_ID:
                    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    head_x, head_y = cx, box[1] + 20 
                    current_dots.append((head_x, head_y))
                    all_tracked_points.append((head_x, head_y))
                    session_stats["total_detections"] += 1

                    sx = max(0, min(int(head_x / (W / GRID_SIZE[1])), GRID_SIZE[1]-1))
                    sy = max(0, min(int(head_y / (H / GRID_SIZE[0])), GRID_SIZE[0]-1))
                    sector_counts[sy, sx] += 1

                    speed = 0
                    if track_id in prev_positions:
                        dist = np.sqrt((head_x-prev_positions[track_id][0])**2 + (head_y-prev_positions[track_id][1])**2)
                        speed = round(dist, 1)
                        if dist > STAMPEDE_VELOCITY_THRESHOLD:
                             session_stats["stampede_triggers"] += 1
                    prev_positions[track_id] = (head_x, head_y)

                    if track_id == locked_target_id:
                        target_found_this_frame = True
                        target_memory["last_seen"] = time.time()
                        target_memory["box"] = box
                        target_memory["stats"] = {"id": track_id, "speed": speed, "pos": (head_x, head_y)}
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 3)

        if len(current_dots) > session_stats["max_crowd"]: session_stats["max_crowd"] = len(current_dots)

        if locked_target_id and not target_found_this_frame:
            if time.time() - target_memory["last_seen"] < 5.0:
                b = target_memory["box"]
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 165, 255), 1)
                cv2.putText(frame, "LOST VISUAL", (b[0], b[1]-10), 1, 0.8, (0, 165, 255), 1)
            else:
                add_log(f"SUBJECT {locked_target_id} LOST")
                locked_target_id = None

        # DASHBOARD UI
        dash = np.zeros((H, DASH_W, 3), dtype=np.uint8)
        cv2.putText(dash, "CRITICAL INTEL HUD", (20, 35), 0, 0.6, (0, 255, 255), 1)
        cv2.line(dash, (20, 45), (300, 45), (60, 60, 60), 1)
        cv2.putText(dash, f"LIVE COUNT: {len(current_dots)}", (20, 80), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(dash, f"WEAPONS: {weapon_count}", (20, 105), 0, 0.5, (0, 0, 255) if weapon_count > 0 else (255, 255, 255), 1)
        
        cv2.rectangle(dash, (15, 130), (305, 255), (30, 30, 30), -1)
        cv2.putText(dash, "SUSPECT METADATA", (25, 150), 0, 0.5, (0, 255, 255), 1)
        if locked_target_id:
            s = target_memory["stats"]
            cv2.putText(dash, f"ID: {s.get('id')}", (30, 175), 0, 0.4, (200, 200, 200), 1)
            cv2.putText(dash, f"SPEED: {s.get('speed')} px/f", (30, 195), 0, 0.4, (0, 255, 0), 1)
            cv2.putText(dash, f"POS: {s.get('pos')}", (30, 215), 0, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(dash, "TARGETING: INACTIVE", (30, 185), 0, 0.4, (100, 100, 100), 1)

        cv2.putText(dash, "ACTIVITY FEED", (20, 285), 0, 0.5, (255, 255, 0), 1)
        for i, log in enumerate(incident_log):
            cv2.putText(dash, log, (20, 310 + (i * 20)), 0, 0.35, (160, 160, 160), 1)

        for y in range(GRID_SIZE[0]):
            for x in range(GRID_SIZE[1]):
                x1, y1 = int(x * W/GRID_SIZE[1]), int(y * H/GRID_SIZE[0])
                x2, y2 = int((x+1) * W/GRID_SIZE[1]), int((y+1) * H/GRID_SIZE[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 1)
                if sector_counts[y, x] >= DENSITY_LIMIT_PER_SECTOR:
                    ov = frame.copy()
                    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), -1)
                    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)

        for d in current_dots: cv2.circle(frame, d, 4, (0, 255, 0), -1)
        main_view = ai_frame if night_mode else frame
        cv2.imshow(win_name, np.hstack((main_view, dash)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('n'): night_mode = not night_mode
        elif key == ord('r'): locked_target_id = None
        elif key == ord('h'):
            hm_raw = np.zeros((H, W), dtype=np.float32)
            for pt in all_tracked_points: cv2.circle(hm_raw, pt, 30, 1, -1)
            hm_blur = cv2.GaussianBlur(hm_raw, (71, 71), 0)
            hm_norm = cv2.normalize(hm_blur, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
            cv2.imwrite(f"heatmap_{int(time.time())}.png", cv2.addWeighted(frame, 0.6, hm_color, 0.4, 0))

    # --- POST SESSION: GENERATE FINAL HEATMAP AS BASE64 ---
    heatmap_base64 = None
    if all_tracked_points and last_valid_frame is not None:
        try:
            h_h, h_w = last_valid_frame.shape[:2]
            hm_raw = np.zeros((h_h, h_w), dtype=np.float32)
            for pt in all_tracked_points: cv2.circle(hm_raw, pt, 30, 1, -1)
            hm_blur = cv2.GaussianBlur(hm_raw, (71, 71), 0)
            hm_norm = cv2.normalize(hm_blur, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
            final_hm_img = cv2.addWeighted(last_valid_frame, 0.6, hm_color, 0.4, 0)
            
            # Encode image as base64 string
            _, buffer = cv2.imencode('.png', final_hm_img)
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Heatmap encoding failed: {e}")

    generate_html_report(heatmap_base64)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()