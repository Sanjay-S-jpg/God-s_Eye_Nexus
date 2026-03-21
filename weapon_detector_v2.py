# weapon_detector_v2.py - Advanced Weapon Detection System v3.0
# 
# Features:
# - Multiple weapon types (knife, scissors, bat, etc.)
# - Custom YOLO model support for gun/firearm detection
# - Small object detection enhancement
# - Context awareness (person holding weapon = higher threat)
# - Threat zone visualization
# - Reduced false positives
#
# COCO Class IDs (YOLOv8):
#   34=baseball bat, 39=bottle, 43=knife, 76=scissors

import cv2
import numpy as np
import time
import os
from collections import defaultdict


class WeaponDetectorV2:
    """
    Advanced weapon detection with context awareness.
    
    Detects:
    - Knife (high danger)
    - Scissors (medium danger)
    - Baseball bat (medium danger)
    - Bottle (low danger - can be weapon)
    
    Context Analysis:
    - Checks if weapon is near a person
    - Analyzes if weapon is being held
    - Calculates threat level
    """
    
    # YOLO COCO class IDs for potential weapons (CORRECTED for YOLOv8)
    WEAPON_CLASSES = {
        43: {"name": "Knife", "danger": "HIGH", "color": (0, 0, 255), "min_conf": 0.25},
        76: {"name": "Scissors", "danger": "MEDIUM", "color": (0, 165, 255), "min_conf": 0.30},
        34: {"name": "Baseball Bat", "danger": "MEDIUM", "color": (0, 165, 255), "min_conf": 0.35},
        39: {"name": "Bottle", "danger": "LOW", "color": (0, 255, 255), "min_conf": 0.40},
    }
    
    # Classes to IGNORE (common false positives) — CORRECTED IDs
    IGNORE_CLASSES = {
        38: "Tennis Racket",
        36: "Skateboard",
        37: "Surfboard",
        46: "Banana",  # Often detected as knife
    }
    
    PERSON_CLASS_ID = 0
    
    def __init__(self, 
                 alert_cooldown=5.0,
                 proximity_threshold=150,
                 min_weapon_size=15,
                 max_weapon_size=500,
                 require_person_nearby=False,
                 weapon_model_path=None):
        """
        Args:
            alert_cooldown: Seconds between alerts for same weapon
            proximity_threshold: Pixels - weapon must be this close to person for high threat
            min_weapon_size: Minimum pixel size to detect (filters noise)
            max_weapon_size: Maximum pixel size (filters large false positives)
            require_person_nearby: If True, only alert when weapon is near a person
            weapon_model_path: Path to custom YOLO model for gun/firearm detection (optional)
        """
        self.alert_cooldown = alert_cooldown
        self.proximity_threshold = proximity_threshold
        self.min_weapon_size = min_weapon_size
        self.max_weapon_size = max_weapon_size
        self.require_person_nearby = require_person_nearby
        
        # Tracking
        self.last_alert_time = {}
        self.active_weapons = []
        self.weapon_history = []
        self.frame_count = 0
        self.seen_counts = {}
        
        # Person positions (updated each frame)
        self.person_positions = []
        self.person_boxes = []
        
        # Custom weapon model for gun detection
        self.gun_model = None
        self.gun_model_classes = {}
        if weapon_model_path and os.path.exists(weapon_model_path):
            try:
                from ultralytics import YOLO
                self.gun_model = YOLO(weapon_model_path)
                self.gun_model_classes = self.gun_model.names
                print(f"[INFO] Custom weapon model loaded: {weapon_model_path}")
                print(f"[INFO] Gun model classes: {self.gun_model_classes}")
            except Exception as e:
                print(f"[WARN] Failed to load custom weapon model: {e}")
        elif weapon_model_path:
            print(f"[INFO] Custom weapon model not found: {weapon_model_path}")
            print(f"[INFO] Gun detection disabled. Place a YOLO weapon model at: {weapon_model_path}")
    
    def _get_box_center(self, box):
        """Get center point of bounding box"""
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    
    def _get_box_size(self, box):
        """Get width and height of bounding box"""
        return (box[2] - box[0], box[3] - box[1])
    
    def _distance(self, p1, p2):
        """Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _is_near_person(self, weapon_center):
        """Check if weapon is near any detected person"""
        for person_pos in self.person_positions:
            if self._distance(weapon_center, person_pos) < self.proximity_threshold:
                return True
        return False
    
    def _find_nearest_person(self, weapon_center):
        """Find the nearest person to a weapon"""
        min_dist = float('inf')
        nearest_idx = -1
        
        for i, person_pos in enumerate(self.person_positions):
            dist = self._distance(weapon_center, person_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        if nearest_idx >= 0 and min_dist < self.proximity_threshold * 2:
            return nearest_idx, min_dist
        return None, None
    
    def _validate_weapon(self, box, cls, conf):
        """
        Validate if detection is actually a weapon.
        Returns: (is_valid, reason)
        """
        if cls not in self.WEAPON_CLASSES:
            return False, "Not a weapon class"
        
        weapon_info = self.WEAPON_CLASSES[cls]
        
        # Check confidence threshold
        if conf < weapon_info['min_conf']:
            return False, f"Low confidence"
        
        # Check size
        width, height = self._get_box_size(box)
        size = max(width, height)
        
        if size < self.min_weapon_size:
            return False, f"Too small"
        
        if size > self.max_weapon_size:
            return False, f"Too large"
        
        # Check aspect ratio (weapons are usually elongated)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        
        # Knives and bats should be elongated
        if weapon_info['name'] in ['Knife', 'Baseball Bat']:
            if aspect_ratio < 1.3:  # Relaxed from 1.5
                return False, f"Wrong shape"
        
        # Check if near person (optional)
        if self.require_person_nearby:
            center = self._get_box_center(box)
            if not self._is_near_person(center):
                return False, "No person nearby"
        
        return True, "Valid weapon"
    
    def _calculate_threat_level(self, weapon_data):
        """
        Calculate threat level based on multiple factors.
        Returns: float (0.0 to 1.0)
        """
        base_threat = {
            'CRITICAL': 0.85,
            'HIGH': 0.7,
            'MEDIUM': 0.4,
            'LOW': 0.2
        }.get(weapon_data['danger'], 0.3)
        
        threat = base_threat
        
        # Increase threat if near person
        if weapon_data.get('near_person'):
            threat += 0.2
        
        # Increase threat based on confidence
        conf_bonus = (weapon_data['confidence'] - 0.25) * 0.3
        threat += max(0, conf_bonus)
        
        # Increase threat if weapon is being held (upper body area)
        if weapon_data.get('held_position'):
            threat += 0.1
        
        return min(1.0, max(0.0, threat))
    
    def _detect_with_custom_model(self, frame):
        """
        Run custom weapon detection model (for guns/firearms).
        Returns list of weapon_data dicts.
        """
        if self.gun_model is None or frame is None:
            return []
        
        custom_weapons = []
        try:
            results = self.gun_model(frame, verbose=False, conf=0.30, imgsz=640)
            
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                for box, cls_t, conf_t in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy()
                ):
                    class_name = self.gun_model_classes.get(int(cls_t), "Weapon")
                    
                    # Determine danger level from class name
                    name_lower = class_name.lower()
                    if any(w in name_lower for w in ["gun", "pistol", "rifle", "firearm", "revolver"]):
                        danger = "CRITICAL"
                        color = (0, 0, 255)
                    elif any(w in name_lower for w in ["knife", "blade", "sword", "machete"]):
                        danger = "HIGH"
                        color = (0, 0, 255)
                    else:
                        danger = "HIGH"
                        color = (0, 100, 255)
                    
                    box_int = tuple([int(b) for b in box])
                    center = self._get_box_center(box_int)
                    width, height = self._get_box_size(box_int)
                    near_person = self._is_near_person(center)
                    
                    weapon_data = {
                        'type': class_name.title(),
                        'danger': danger,
                        'color': color,
                        'box': box_int,
                        'center': center,
                        'size': (width, height),
                        'confidence': float(conf_t),
                        'class_id': int(cls_t),
                        'near_person': near_person,
                        'held_position': False,
                        'distance_to_person': None,
                        'frame': self.frame_count,
                        'source': 'custom_model'
                    }
                    weapon_data['threat_level'] = self._calculate_threat_level(weapon_data)
                    custom_weapons.append(weapon_data)
        except Exception as e:
            print(f"[WARN] Custom weapon model error: {e}")
        
        return custom_weapons
    
    def process_frame(self, boxes, classes, confidences, person_boxes=None, frame=None):
        """
        Process a frame and detect weapons with context.
        
        Args:
            boxes: All detected bounding boxes
            classes: Class IDs for each box
            confidences: Confidence scores
            person_boxes: Bounding boxes of detected PERSONS ONLY
            frame: Original frame for custom gun model inference (optional)
            
        Returns:
            dict with weapon detections and alerts
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Update person positions — use person_boxes if provided, otherwise filter from all boxes
        self.person_positions = []
        self.person_boxes = []
        
        if person_boxes is not None and len(person_boxes) > 0:
            for box in person_boxes:
                center = self._get_box_center(box)
                self.person_positions.append(center)
                self.person_boxes.append(box)
        elif len(boxes) > 0:
            for box, cls in zip(boxes, classes):
                if int(cls) == self.PERSON_CLASS_ID:
                    center = self._get_box_center(box)
                    self.person_positions.append(center)
                    self.person_boxes.append(box)
        
        # Detect weapons
        self.active_weapons = []
        new_alerts = []
        
        if len(boxes) == 0:
            return {
                'weapons': [],
                'new_alerts': [],
                'total_detected': 0,
                'high_danger_count': 0,
                'threats_near_people': 0
            }
        
        for box, cls, conf in zip(boxes, classes, confidences):
            cls = int(cls)
            box = [int(b) for b in box]
            
            # Skip if not a weapon class
            if cls not in self.WEAPON_CLASSES:
                continue
            
            # Skip ignored classes
            if cls in self.IGNORE_CLASSES:
                continue
            
            # Validate weapon
            is_valid, reason = self._validate_weapon(box, cls, conf)
            if not is_valid:
                continue
            
            weapon_info = self.WEAPON_CLASSES[cls]
            center = self._get_box_center(box)
            width, height = self._get_box_size(box)
            
            # Check proximity to person
            near_person = self._is_near_person(center)
            nearest_person_idx, distance_to_person = self._find_nearest_person(center)
            
            # Determine if weapon is in "held" position (upper body area)
            held_position = False
            if nearest_person_idx is not None and len(self.person_boxes) > nearest_person_idx:
                person_box = self.person_boxes[nearest_person_idx]
                person_height = person_box[3] - person_box[1]
                waist_line = person_box[1] + person_height * 0.6
                if center[1] < waist_line:
                    held_position = True
            
            # Build weapon data
            key = (cls, center[0] // 20, center[1] // 20)
            self.seen_counts[key] = self.seen_counts.get(key, 0) + 1
            weapon_data = {
                'type': weapon_info['name'],
                'danger': weapon_info['danger'],
                'color': weapon_info['color'],
                'box': tuple(box),
                'center': center,
                'size': (width, height),
                'confidence': float(conf),
                'class_id': cls,
                'near_person': near_person,
                'held_position': held_position,
                'distance_to_person': distance_to_person,
                'frame': self.frame_count
            }
            
            # Calculate threat level
            weapon_data['threat_level'] = self._calculate_threat_level(weapon_data)
            
            low_conf = weapon_data['confidence'] < 0.35
            persistent = self.seen_counts.get(key, 0) >= (2 if low_conf else 1)
            if not persistent:
                continue
            self.active_weapons.append(weapon_data)
            
            # Check cooldown for alerts
            weapon_key = f"{weapon_info['name']}_{center[0]//100}_{center[1]//100}"
            last_time = self.last_alert_time.get(weapon_key, 0)
            
            if current_time - last_time > self.alert_cooldown:
                # Only alert for meaningful threats
                if weapon_data['threat_level'] > 0.25:
                    self.last_alert_time[weapon_key] = current_time
                    new_alerts.append(weapon_data)
                    self.weapon_history.append({
                        **weapon_data,
                        'timestamp': current_time
                    })
        
        # --- Run custom gun model if available ---
        if self.gun_model is not None and frame is not None:
            custom_weapons = self._detect_with_custom_model(frame)
            for weapon_data in custom_weapons:
                self.active_weapons.append(weapon_data)
                # Alert for custom model detections
                weapon_key = f"gun_{weapon_data['center'][0]//100}_{weapon_data['center'][1]//100}"
                last_time = self.last_alert_time.get(weapon_key, 0)
                if current_time - last_time > self.alert_cooldown:
                    if weapon_data['threat_level'] > 0.2:
                        self.last_alert_time[weapon_key] = current_time
                        new_alerts.append(weapon_data)
                        self.weapon_history.append({
                            **weapon_data,
                            'timestamp': current_time
                        })
        
        return {
            'weapons': self.active_weapons,
            'new_alerts': new_alerts,
            'total_detected': len(self.active_weapons),
            'high_danger_count': sum(1 for w in self.active_weapons if w['danger'] == 'HIGH'),
            'threats_near_people': sum(1 for w in self.active_weapons if w['near_person'])
        }
    
    def draw_detections(self, frame):
        """
        Draw weapon detections on frame with threat visualization.
        """
        for weapon in self.active_weapons:
            box = weapon['box']
            color = weapon['color']
            danger = weapon['danger']
            threat = weapon['threat_level']
            name = weapon['type']
            near_person = weapon['near_person']
            
            x1, y1, x2, y2 = box
            
            # Thickness based on threat level
            thickness = 2 + int(threat * 2)
            
            # Draw main box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw corner accents for high danger
            if danger == 'HIGH' or threat > 0.6:
                corner_len = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
                corners = [
                    ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                    ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))
                ]
                for (pt, (dx, dy)) in corners:
                    cv2.line(frame, pt, (pt[0] + dx * corner_len, pt[1]), color, thickness + 1)
                    cv2.line(frame, pt, (pt[0], pt[1] + dy * corner_len), color, thickness + 1)
            
            # Label with threat percentage
            label = f"{name} {int(threat * 100)}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + label_size[0] + 10, y1 - 2), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw line to nearest person if weapon is near them
            if near_person and weapon.get('distance_to_person'):
                center = weapon['center']
                for person_pos in self.person_positions:
                    dist = self._distance(center, person_pos)
                    if dist < self.proximity_threshold:
                        # Draw line to person
                        cv2.line(frame, center, person_pos, (0, 0, 255), 2)
                        # Draw danger circle around person
                        cv2.circle(frame, person_pos, int(dist), (0, 0, 255), 1)
                        break
            
            # Threat indicator below box
            if threat > 0.5:
                threat_text = "DANGER!" if threat > 0.7 else "THREAT!"
                cv2.putText(frame, threat_text, (x1, y2 + 18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_alert_banner(self, frame):
        """
        Draw alert banner if dangerous weapons detected.
        """
        if not self.active_weapons:
            return frame
        
        h, w = frame.shape[:2]
        high_threats = [wp for wp in self.active_weapons if wp['threat_level'] > 0.5]
        
        if high_threats:
            # Red alert banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            weapon_names = ", ".join(set(wp['type'] for wp in high_threats))
            cv2.putText(frame, "WEAPON ALERT", (w // 2 - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Detected: {weapon_names}", (w // 2 - 100, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif self.active_weapons:
            # Orange caution banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            weapon_names = ", ".join(set(wp['type'] for wp in self.active_weapons))
            cv2.putText(frame, f"Caution: {weapon_names} detected", (w // 2 - 120, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_summary(self):
        """Get detection summary"""
        if not self.weapon_history:
            return {
                'total_detections': 0,
                'unique_types': [],
                'high_threat_events': 0,
                'avg_threat_level': 0
            }
        
        return {
            'total_detections': len(self.weapon_history),
            'unique_types': list(set(w['type'] for w in self.weapon_history)),
            'high_threat_events': sum(1 for w in self.weapon_history if w['threat_level'] > 0.7),
            'avg_threat_level': np.mean([w['threat_level'] for w in self.weapon_history])
        }
    
    def reset(self):
        """Reset all tracking"""
        self.last_alert_time.clear()
        self.active_weapons.clear()
        self.weapon_history.clear()
        self.frame_count = 0
        self.person_positions.clear()
        self.person_boxes.clear()
