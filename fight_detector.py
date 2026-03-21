# fight_detector.py - Fight/Aggression Detection using MediaPipe Pose Estimation
# For God's Eye Nexus - Crowd Intelligence System
#
# Detects fights and aggressive behavior using:
#   1. MediaPipe Pose estimation for arm/body movement analysis
#   2. Proximity analysis between people
#   3. Velocity-based aggression scoring
#   4. Striking pose detection (bent elbows, raised arms)
#
# Uses MediaPipe Tasks API (v0.10.x+) with auto-download of pose model.

import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque

# Model download URL and local path
_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
_POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] MediaPipe not installed. Fight detection disabled.")
    print("[INFO] Install with: pip install mediapipe")


def _ensure_pose_model():
    """Download the pose landmarker model if not already present."""
    if os.path.exists(_POSE_MODEL_PATH):
        return _POSE_MODEL_PATH
    print(f"[INFO] Downloading pose model to {_POSE_MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(_POSE_MODEL_URL, _POSE_MODEL_PATH)
        print("[INFO] Pose model downloaded successfully.")
        return _POSE_MODEL_PATH
    except Exception as e:
        print(f"[ERROR] Failed to download pose model: {e}")
        return None


class FightDetector:
    """
    Detects physical fights and aggressive behavior using pose estimation.
    
    How it works:
    1. Only analyzes people who are CLOSE together (saves performance)
    2. Uses MediaPipe PoseLandmarker (Tasks API) to get arm/body joints
    3. Calculates aggression score per person based on:
       - Arm velocity (rapid punching/swinging motions)
       - Arms raised above shoulders (fighting/striking stance)
       - Elbow angle (bent = striking position)
       - Body lunge speed (rushing at someone)
    4. If two close people both show aggression → FIGHT DETECTED
    """
    
    # MediaPipe PoseLandmark indices (same as Tasks API enum values)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    def __init__(self,
                 proximity_threshold=200,
                 aggression_threshold=0.4,
                 analysis_interval=3,
                 cooldown=10.0):
        """
        Args:
            proximity_threshold: Max pixel distance between people to consider as potential fight
            aggression_threshold: Score above this triggers a fight alert (0.0 to 1.0)
            analysis_interval: Run pose estimation every N frames (for performance)
            cooldown: Seconds between alerts for the same fight pair
        """
        self.proximity_threshold = proximity_threshold
        self.aggression_threshold = aggression_threshold
        self.analysis_interval = analysis_interval
        self.cooldown = cooldown
        
        self.frame_count = 0
        self.person_history = {}   # {track_id: deque of pose data}
        self.active_fights = []
        self.fight_history = []
        self.last_alert_time = {}
        self.last_aggression_scores = {}  # Cache between analysis frames
        
        # Initialize MediaPipe PoseLandmarker (Tasks API)
        self.landmarker = None
        self.enabled = False

        if MEDIAPIPE_AVAILABLE:
            model_path = _ensure_pose_model()
            if model_path:
                try:
                    BaseOptions = mp.tasks.BaseOptions
                    PoseLandmarker = mp.tasks.vision.PoseLandmarker
                    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                    VisionRunningMode = mp.tasks.vision.RunningMode

                    options = PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        running_mode=VisionRunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                    )
                    self.landmarker = PoseLandmarker.create_from_options(options)
                    self.enabled = True
                    print("[INFO] Fight Detection: MediaPipe PoseLandmarker initialized (lite model)")
                except Exception as e:
                    print(f"[WARN] Failed to create PoseLandmarker: {e}")
            else:
                print("[WARN] Pose model not available. Fight detection disabled.")
        else:
            print("[WARN] MediaPipe not available. Fight detection disabled.")
    
    def _extract_pose(self, frame, box):
        """
        Extract pose landmarks from a person's bounding box region.
        Returns dict of {landmark_id: (x, y)} in frame coordinates, or None.
        """
        if not self.enabled or self.landmarker is None:
            return None
        
        x1, y1, x2, y2 = [int(b) for b in box]
        h_frame, w_frame = frame.shape[:2]
        
        # Clamp to frame bounds with padding
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_frame, x2 + pad)
        y2 = min(h_frame, y2 + pad)
        
        if x2 <= x1 + 20 or y2 <= y1 + 20:
            return None
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
        except Exception:
            return None
        
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None
        
        pose_lms = result.pose_landmarks[0]  # First (only) detected pose
        h_roi, w_roi = roi.shape[:2]
        landmarks = {}
        
        key_points = [
            self.NOSE,
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_ELBOW, self.RIGHT_ELBOW,
            self.LEFT_WRIST, self.RIGHT_WRIST,
            self.LEFT_HIP, self.RIGHT_HIP
        ]
        
        for idx in key_points:
            if idx < len(pose_lms):
                lm = pose_lms[idx]
                vis = lm.visibility if lm.visibility is not None else (lm.presence if lm.presence is not None else 0)
                if vis > 0.4:
                    landmarks[idx] = (
                        int(lm.x * w_roi + x1),
                        int(lm.y * h_roi + y1)
                    )
        
        return landmarks if len(landmarks) >= 4 else None
    
    def _calculate_aggression_score(self, track_id):
        """
        Calculate aggression score for a person based on their pose history.
        
        Factors:
        1. Arm velocity (0.35 max) - rapid punching/swinging
        2. Arms raised (0.15 each arm) - fighting stance
        3. Elbow angle (0.10 each arm) - striking position
        4. Body lunge (0.15 max) - rushing forward
        
        Returns: float 0.0 to 1.0
        """
        if track_id not in self.person_history:
            return 0.0
        
        history = list(self.person_history[track_id])
        if len(history) < 3:
            return 0.0
        
        score = 0.0
        
        # === 1. ARM VELOCITY ===
        # Rapid arm movements indicate punching or swinging
        wrist_velocities = []
        for i in range(1, len(history)):
            dt = history[i]['time'] - history[i-1]['time']
            if dt <= 0.001:
                continue
            
            for wrist_id in [self.LEFT_WRIST, self.RIGHT_WRIST]:
                if wrist_id in history[i]['landmarks'] and wrist_id in history[i-1]['landmarks']:
                    curr = history[i]['landmarks'][wrist_id]
                    prev = history[i-1]['landmarks'][wrist_id]
                    vel = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2) / dt
                    wrist_velocities.append(vel)
        
        if wrist_velocities:
            max_vel = max(wrist_velocities)
            # Fast arm movement (pixels/sec) — normalize against threshold
            score += min(0.35, max_vel / 800.0)
        
        # === 2. ARMS RAISED ABOVE SHOULDERS ===
        current = history[-1]['landmarks']
        for wrist_id, shoulder_id in [(self.LEFT_WRIST, self.LEFT_SHOULDER),
                                       (self.RIGHT_WRIST, self.RIGHT_SHOULDER)]:
            if wrist_id in current and shoulder_id in current:
                # y-axis is inverted: smaller y = higher position
                if current[wrist_id][1] < current[shoulder_id][1] - 15:
                    score += 0.15
        
        # === 3. ELBOW ANGLE (bent = striking pose) ===
        for side_shoulder, side_elbow, side_wrist in [
            (self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST),
            (self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST)
        ]:
            if all(k in current for k in [side_shoulder, side_elbow, side_wrist]):
                s = np.array(current[side_shoulder], dtype=float)
                e = np.array(current[side_elbow], dtype=float)
                w = np.array(current[side_wrist], dtype=float)
                
                v1 = s - e
                v2 = w - e
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                if norm_product > 1e-6:
                    cos_angle = np.dot(v1, v2) / norm_product
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    
                    # Acute angle (< 90°) suggests cocked/striking position
                    if angle < 90:
                        score += 0.10
        
        # === 4. BODY LUNGE (rapid hip/torso movement) ===
        if len(history) >= 3:
            for hip_id in [self.LEFT_HIP, self.RIGHT_HIP]:
                if hip_id in history[-1]['landmarks'] and hip_id in history[-3]['landmarks']:
                    dt = history[-1]['time'] - history[-3]['time']
                    if dt > 0.01:
                        curr = history[-1]['landmarks'][hip_id]
                        prev = history[-3]['landmarks'][hip_id]
                        body_vel = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2) / dt
                        score += min(0.15, body_vel / 500.0)
        
        return min(1.0, max(0.0, score))
    
    def process_frame(self, frame, person_boxes, person_positions):
        """
        Process a frame for fight detection.
        
        Only runs pose estimation on people who are CLOSE to others (performance optimization).
        
        Args:
            frame: BGR image (numpy array)
            person_boxes: dict {track_id: (x1, y1, x2, y2)}
            person_positions: dict {track_id: (cx, cy)}
        
        Returns:
            dict with 'fights', 'aggressive_persons', 'fight_count', 'new_alerts'
        """
        self.frame_count += 1
        current_time = time.time()
        self.active_fights = []
        new_alerts = []
        
        if not self.enabled or len(person_positions) < 2:
            return {
                'fights': [],
                'aggressive_persons': self.last_aggression_scores,
                'fight_count': 0,
                'new_alerts': []
            }
        
        # Only run full analysis every N frames for performance
        run_analysis = (self.frame_count % self.analysis_interval == 0)
        
        if not run_analysis:
            return {
                'fights': self.active_fights,
                'aggressive_persons': self.last_aggression_scores,
                'fight_count': len(self.active_fights),
                'new_alerts': []
            }
        
        # --- Step 1: Find pairs of people within proximity ---
        close_pairs = []
        person_ids_list = list(person_positions.keys())
        
        for i in range(len(person_ids_list)):
            for j in range(i + 1, len(person_ids_list)):
                tid1, tid2 = person_ids_list[i], person_ids_list[j]
                p1 = person_positions[tid1]
                p2 = person_positions[tid2]
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                if dist < self.proximity_threshold:
                    close_pairs.append((tid1, tid2, dist))
        
        # --- Step 2: Only run pose estimation on people in close pairs ---
        people_to_analyze = set()
        for tid1, tid2, _ in close_pairs:
            people_to_analyze.add(tid1)
            people_to_analyze.add(tid2)
        
        # --- Step 3: Extract poses and calculate aggression ---
        aggression_scores = {}
        
        for track_id in people_to_analyze:
            if track_id not in person_boxes:
                continue
            
            box = person_boxes[track_id]
            landmarks = self._extract_pose(frame, box)
            
            if landmarks:
                if track_id not in self.person_history:
                    self.person_history[track_id] = deque(maxlen=15)
                
                self.person_history[track_id].append({
                    'landmarks': landmarks,
                    'time': current_time
                })
                
                aggression = self._calculate_aggression_score(track_id)
                aggression_scores[track_id] = aggression
        
        self.last_aggression_scores = aggression_scores
        
        # --- Step 4: Detect fights (close pair + mutual aggression) ---
        for tid1, tid2, dist in close_pairs:
            agg1 = aggression_scores.get(tid1, 0)
            agg2 = aggression_scores.get(tid2, 0)
            
            combined = (agg1 + agg2) / 2
            max_agg = max(agg1, agg2)
            
            # Fight detected if:
            # - Combined aggression exceeds threshold, OR
            # - One person is highly aggressive (> 0.6) while close to another
            is_fight = (combined > self.aggression_threshold) or (max_agg > 0.6 and combined > 0.25)
            
            if is_fight:
                p1 = person_positions[tid1]
                p2 = person_positions[tid2]
                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                
                fight_key = f"{min(tid1, tid2)}_{max(tid1, tid2)}"
                
                fight_data = {
                    'person1': tid1,
                    'person2': tid2,
                    'severity': combined,
                    'max_aggression': max_agg,
                    'distance': dist,
                    'center': center,
                    'aggression': {tid1: agg1, tid2: agg2}
                }
                
                self.active_fights.append(fight_data)
                
                # Alert with cooldown
                last_time = self.last_alert_time.get(fight_key, 0)
                if current_time - last_time > self.cooldown:
                    self.last_alert_time[fight_key] = current_time
                    self.fight_history.append({
                        **fight_data,
                        'timestamp': current_time
                    })
                    new_alerts.append(fight_data)
        
        # Clean up history for people no longer visible
        for tid in list(self.person_history.keys()):
            if tid not in person_positions:
                if len(self.person_history[tid]) > 0:
                    last_time_seen = self.person_history[tid][-1]['time']
                    if current_time - last_time_seen > 5.0:
                        del self.person_history[tid]
        
        return {
            'fights': self.active_fights,
            'aggressive_persons': aggression_scores,
            'fight_count': len(self.active_fights),
            'new_alerts': new_alerts
        }
    
    def draw_detections(self, frame, person_positions):
        """Draw fight detection visualizations on frame."""
        for fight in self.active_fights:
            tid1, tid2 = fight['person1'], fight['person2']
            severity = fight['severity']
            center = fight['center']
            
            # Color based on severity
            if severity > 0.7:
                color = (0, 0, 255)     # Red — active fight
                label = "FIGHT!"
            elif severity > 0.5:
                color = (0, 100, 255)   # Orange — aggression
                label = "AGGRESSION"
            else:
                color = (0, 200, 255)   # Yellow — altercation
                label = "ALTERCATION"
            
            # Draw line between fighters
            if tid1 in person_positions and tid2 in person_positions:
                p1 = person_positions[tid1]
                p2 = person_positions[tid2]
                cv2.line(frame, p1, p2, color, 3)
            
            # Alert circle around fight area
            radius = int(fight['distance'] / 2 + 30)
            cv2.circle(frame, center, radius, color, 2)
            
            # Pulsing inner circle
            pulse = int((time.time() * 3) % 2)
            inner_r = max(5, radius - 10 - pulse * 5)
            cv2.circle(frame, center, inner_r, color, 1)
            
            # Label
            label_text = f"{label} {int(severity * 100)}%"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_x = center[0] - text_size[0] // 2
            label_y = center[1] - radius - 15
            
            cv2.rectangle(frame,
                          (label_x - 5, label_y - 18),
                          (label_x + text_size[0] + 5, label_y + 5),
                          color, -1)
            cv2.putText(frame, label_text,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_alert_banner(self, frame):
        """Draw fight alert banner at top of frame."""
        if not self.active_fights:
            return frame
        
        h, w = frame.shape[:2]
        max_severity = max(f['severity'] for f in self.active_fights)
        
        if max_severity > 0.4:
            overlay = frame.copy()
            banner_color = (0, 0, 200) if max_severity > 0.6 else (0, 100, 200)
            cv2.rectangle(overlay, (0, 0), (w, 50), banner_color, -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            text = f"FIGHT DETECTED - Severity: {int(max_severity * 100)}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, text, (w // 2 - text_size[0] // 2, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def get_summary(self):
        """Get fight detection summary for reports."""
        return {
            'total_fights': len(self.fight_history),
            'max_severity': round(max((f['severity'] for f in self.fight_history), default=0), 2),
            'active_fights': len(self.active_fights),
            'enabled': self.enabled
        }
    
    def reset(self):
        """Reset all tracking data."""
        self.person_history.clear()
        self.active_fights.clear()
        self.fight_history.clear()
        self.last_alert_time.clear()
        self.last_aggression_scores.clear()
        self.frame_count = 0


    def __del__(self):
        """Cleanup PoseLandmarker resources."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("Fight Detection Module")
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    
    detector = FightDetector()
    print(f"Detector enabled: {detector.enabled}")
    
    if detector.enabled:
        print("Fight detection is ready!")
    else:
        print("Install mediapipe: pip install mediapipe")
