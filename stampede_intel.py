# stampede_intel.py - Intelligent Stampede Detection Module
# Created for Sanjay's Final Year Project

import numpy as np
from collections import deque
import random
import cv2
import time


class StampedeDetector:
    """
    Multi-factor stampede detection system.
    
    Instead of just counting people (which gives false alarms),
    this analyzes BEHAVIOR patterns that indicate real panic.
    """
    
    def __init__(self, history_length=30):
        self.velocity_history = deque(maxlen=history_length)
        self.count_history = deque(maxlen=history_length)
        self.alert_level = 0  # 0=SAFE, 1=CAUTION, 2=WARNING, 3=CRITICAL
        self.prev_positions = {}
        
    def update(self, current_positions, sector_counts, frame_dims):
        """
        Call this every frame with current tracking data.
        
        Args:
            current_positions: dict of {track_id: (x, y)}
            sector_counts: numpy array of density per grid cell
            frame_dims: tuple of (width, height)
        
        Returns:
            dict with risk_score, alert_level, and component breakdown
        """
        # Calculate velocities from position changes
        velocities = self._calculate_velocities(current_positions)
        
        # Store for acceleration detection
        speeds = {tid: np.linalg.norm(v) for tid, v in velocities.items()}
        self.velocity_history.append(speeds)
        self.count_history.append(len(current_positions))
        
        # Calculate all component scores (each 0.0 to 1.0)
        coherence = self._calculate_coherence(velocities)
        acceleration = self._calculate_acceleration()
        avg_speed = self._calculate_average_speed(speeds)
        spike = self._calculate_crowd_spike()
        edge = self._calculate_edge_pressure(current_positions, frame_dims)
        
        # Weighted combination - coherence is most important!
        weights = {
            'coherence': 0.30,      # Are people moving SAME direction?
            'acceleration': 0.25,   # Did speed suddenly increase?
            'avg_speed': 0.20,      # How fast is everyone moving?
            'spike': 0.15,          # Did crowd size suddenly change?
            'edge': 0.10            # Are people crushed at edges?
        }
        
        raw_score = (
            coherence * weights['coherence'] +
            acceleration * weights['acceleration'] +
            avg_speed * weights['avg_speed'] +
            spike * weights['spike'] +
            edge * weights['edge']
        )
        
        # Scale to 0-100
        risk_score = int(raw_score * 100)
        
        # Update alert level with hysteresis (prevents flickering)
        self._update_alert_level(risk_score)
        
        # Save current positions for next frame
        self.prev_positions = current_positions.copy()
        
        return {
            'risk_score': risk_score,
            'alert_level': self.alert_level,
            'alert_name': ['SAFE', 'CAUTION', 'WARNING', 'CRITICAL'][self.alert_level],
            'components': {
                'coherence': round(coherence, 2),
                'acceleration': round(acceleration, 2),
                'avg_speed': round(avg_speed, 2),
                'spike': round(spike, 2),
                'edge': round(edge, 2)
            }
        }
    
    def _calculate_velocities(self, current_positions):
        """Calculate velocity vector for each tracked person"""
        velocities = {}
        for track_id, (cx, cy) in current_positions.items():
            if track_id in self.prev_positions:
                px, py = self.prev_positions[track_id]
                velocities[track_id] = (cx - px, cy - py)
        return velocities
    
    def _calculate_coherence(self, velocities):
        """
        How aligned is everyone's movement direction?
        
        High coherence = everyone running same way = PANIC
        Low coherence = random directions = normal crowd
        """
        if len(velocities) < 3:
            return 0.0
        
        vectors = np.array(list(velocities.values()))
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        # Only consider people who are actually moving
        moving_mask = magnitudes > 3.0
        if moving_mask.sum() < 3:
            return 0.0
        
        moving_vectors = vectors[moving_mask]
        moving_mags = magnitudes[moving_mask]
        
        # Normalize to unit vectors (direction only)
        unit_vectors = moving_vectors / moving_mags[:, np.newaxis]
        
        # Mean direction - if everyone aligned, this has magnitude ~1
        mean_direction = np.mean(unit_vectors, axis=0)
        coherence = np.linalg.norm(mean_direction)
        
        return float(coherence)
    
    def _calculate_acceleration(self):
        """Detect sudden collective speed increase"""
        if len(self.velocity_history) < 10:
            return 0.0
        
        history = list(self.velocity_history)
        
        # Average speed in first half vs second half
        first_half = history[:len(history)//2]
        second_half = history[len(history)//2:]
        
        def avg_speed(frames):
            all_speeds = []
            for frame in frames:
                if frame:
                    all_speeds.extend(frame.values())
            return np.mean(all_speeds) if all_speeds else 0
        
        early_speed = avg_speed(first_half)
        recent_speed = avg_speed(second_half)
        
        if early_speed < 1:
            early_speed = 1  # Prevent division issues
        
        # How much faster is recent movement?
        acceleration_ratio = (recent_speed - early_speed) / early_speed
        
        # Clamp to 0-1
        return min(1.0, max(0.0, acceleration_ratio / 2.0))
    
    def _calculate_average_speed(self, speeds):
        """Normalized average movement speed"""
        if not speeds:
            return 0.0
        
        avg = np.mean(list(speeds.values()))
        
        # Normalize: 0 speed = 0, 20+ pixels/frame = 1.0
        return min(1.0, avg / 20.0)
    
    def _calculate_crowd_spike(self):
        """Detect sudden increase in crowd size"""
        if len(self.count_history) < 10:
            return 0.0
        
        history = list(self.count_history)
        baseline = np.mean(history[:-5]) if len(history) > 5 else history[0]
        recent = np.mean(history[-5:])
        
        if baseline < 1:
            baseline = 1
        
        spike_ratio = (recent - baseline) / baseline
        return min(1.0, max(0.0, spike_ratio))
    
    def _calculate_edge_pressure(self, positions, frame_dims):
        """Detect people clustering at frame edges (crushing)"""
        if not positions:
            return 0.0
        
        W, H = frame_dims
        margin = min(W, H) * 0.1  # 10% margin
        
        edge_count = 0
        for (x, y) in positions.values():
            at_edge = (x < margin or x > W - margin or 
                      y < margin or y > H - margin)
            if at_edge:
                edge_count += 1
        
        return edge_count / len(positions)
    
    def _update_alert_level(self, risk_score):
        """Update alert level with hysteresis to prevent flickering"""
        if risk_score > 65:
            self.alert_level = min(3, self.alert_level + 1)
        elif risk_score > 40:
            if self.alert_level < 2:
                self.alert_level = max(1, self.alert_level)
        elif risk_score < 20:
            self.alert_level = max(0, self.alert_level - 1)


# ============================================================
# DEMO MODE - For presentation without real camera/GPU
# ============================================================

class DemoDataGenerator:
    """
    Generates fake but realistic crowd movement for demos.
    Use this when you don't have GPU/camera during presentation.
    """
    
    def __init__(self, frame_width, frame_height, num_people=12):
        self.W = frame_width
        self.H = frame_height
        self.num_people = num_people
        self.mode = 'normal'
        self.frame_count = 0
        self.transition_frames = 0
        self._init_people()
    
    def _init_people(self):
        """Initialize random people positions"""
        self.people = {}
        for i in range(self.num_people):
            self.people[i] = {
                'x': random.randint(100, self.W - 100),
                'y': random.randint(100, self.H - 100),
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-2, 2),
                'target_vx': 0,
                'target_vy': 0
            }
    
    def set_mode(self, mode):
        """
        Switch crowd behavior:
        - 'normal': Random walking
        - 'gathering': People cluster together  
        - 'stampede': Everyone runs same direction (PANIC!)
        """
        self.mode = mode
        self.transition_frames = 30
        
        if mode == 'stampede':
            # Pick a random escape direction
            angle = random.uniform(0, 2 * 3.14159)
            for p in self.people.values():
                speed = random.uniform(12, 18)
                p['target_vx'] = np.cos(angle) * speed + random.uniform(-2, 2)
                p['target_vy'] = np.sin(angle) * speed + random.uniform(-2, 2)
                
        elif mode == 'gathering':
            # Everyone moves toward center
            cx, cy = self.W // 2, self.H // 2
            for p in self.people.values():
                dx = cx - p['x']
                dy = cy - p['y']
                dist = max(1, np.sqrt(dx*dx + dy*dy))
                p['target_vx'] = (dx / dist) * 3
                p['target_vy'] = (dy / dist) * 3
                
        elif mode == 'normal':
            for p in self.people.values():
                p['target_vx'] = random.uniform(-2, 2)
                p['target_vy'] = random.uniform(-2, 2)
    
    def get_frame_data(self):
        """Returns positions dict like real tracking would"""
        self.frame_count += 1
        
        # Smooth transition to target velocities
        if self.transition_frames > 0:
            self.transition_frames -= 1
            blend = 0.1
        else:
            blend = 0.02
        
        positions = {}
        for pid, p in self.people.items():
            # Blend toward target velocity
            p['vx'] += (p['target_vx'] - p['vx']) * blend
            p['vy'] += (p['target_vy'] - p['vy']) * blend
            
            # Add some randomness
            p['vx'] += random.uniform(-0.5, 0.5)
            p['vy'] += random.uniform(-0.5, 0.5)
            
            # Update position
            p['x'] += p['vx']
            p['y'] += p['vy']
            
            # Bounce off edges
            if p['x'] < 50:
                p['x'] = 50
                p['vx'] = abs(p['vx']) * 0.5
            if p['x'] > self.W - 50:
                p['x'] = self.W - 50
                p['vx'] = -abs(p['vx']) * 0.5
            if p['y'] < 50:
                p['y'] = 50
                p['vy'] = abs(p['vy']) * 0.5
            if p['y'] > self.H - 50:
                p['y'] = self.H - 50
                p['vy'] = -abs(p['vy']) * 0.5
            
            positions[pid] = (int(p['x']), int(p['y']))
        
        return positions
    
    def draw_people(self, frame):
        """Draw circles for each simulated person"""
        for pid, (x, y) in self.get_frame_data().items():
            color = (0, 255, 0)  # Green
            cv2.circle(frame, (x, y), 15, color, -1)
            cv2.circle(frame, (x, y), 15, (255, 255, 255), 2)
            cv2.putText(frame, str(pid), (x-5, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return frame


# ============================================================
# OPTIMIZED REAL-TIME HEATMAP SYSTEM
# ============================================================



# ============================================================
# PER-PERSON HEATMAP SYSTEM (Like thermal imaging)
# ============================================================

# ============================================================
# VERTICAL TRAIL HEATMAP (Like thermal shadow)
# ============================================================

class HeatmapGenerator:
    """
    Creates vertical heat trails behind each person.
    Looks like thermal imaging with trails/shadows.
    """
    
    def __init__(self, width, height, trail_length=25, decay_rate=0.88):
        """
        Args:
            width: Frame width
            height: Frame height  
            trail_length: How many past positions to keep
            decay_rate: Trail fade speed
        """
        self.width = width
        self.height = height
        self.trail_length = trail_length
        self.decay_rate = decay_rate
        
        # Store trail per person: {track_id: [(x, y, box_height, heat), ...]}
        self.person_trails = {}
        self.person_boxes = {}  # Store last known box size
        
    def update(self, person_positions, boxes=None):
        """
        Update with current positions and bounding boxes.
        """
        current_ids = set(person_positions.keys())
        
        for track_id, (cx, cy) in person_positions.items():
            cx, cy = int(cx), int(cy)
            
            # Get box dimensions if available
            box_height = 100  # default
            box_width = 40
            if boxes and track_id in boxes:
                x1, y1, x2, y2 = boxes[track_id]
                box_height = int(y2 - y1)
                box_width = int(x2 - x1)
                self.person_boxes[track_id] = (box_width, box_height)
            elif track_id in self.person_boxes:
                box_width, box_height = self.person_boxes[track_id]
            
            if track_id not in self.person_trails:
                self.person_trails[track_id] = []
            
            trail = self.person_trails[track_id]
            
            # Add current position with full heat
            trail.append({
                'x': cx,
                'y': cy,
                'w': box_width,
                'h': box_height,
                'heat': 1.0
            })
            
            # Keep only recent trail
            if len(trail) > self.trail_length:
                trail.pop(0)
            
            # Decay older points
            for i in range(len(trail) - 1):
                trail[i]['heat'] *= self.decay_rate
        
        # Fade out people who left
        for track_id in list(self.person_trails.keys()):
            if track_id not in current_ids:
                trail = self.person_trails[track_id]
                for point in trail:
                    point['heat'] *= 0.8
                # Remove if faded
                if len(trail) == 0 or trail[-1]['heat'] < 0.05:
                    del self.person_trails[track_id]
    
    def get_heatmap_overlay(self, frame, alpha=0.55, grayscale_bg=True):
        """
        Generate vertical trail heatmap overlay.
        
        Args:
            frame: Original BGR frame
            alpha: Heat transparency
            grayscale_bg: If True, makes background grayscale for better visibility
        """
        # Optionally convert background to grayscale for contrast
        if grayscale_bg:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            base_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Slight color tint
            base_frame = (base_frame * 0.7).astype(np.uint8)
        else:
            base_frame = frame.copy()
        
        # Create heat layer
        heat_layer = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Draw each person's vertical heat trail
        for track_id, trail in self.person_trails.items():
            for i, point in enumerate(trail):
                x, y = point['x'], point['y']
                w, h = point['w'], point['h']
                heat = point['heat']
                
                # Draw vertical ellipse (taller than wide) for body heat
                # Heat extends from head to feet
                center_x = x
                center_y = y + h // 4  # Shift down a bit
                
                # Size based on position in trail (newer = bigger)
                size_factor = 0.5 + (i / max(1, len(trail))) * 0.5
                ellipse_w = int(w * 0.6 * size_factor)
                ellipse_h = int(h * 0.8 * size_factor)
                
                if ellipse_w > 0 and ellipse_h > 0:
                    # Draw filled ellipse
                    cv2.ellipse(
                        heat_layer,
                        (center_x, center_y),
                        (ellipse_w // 2, ellipse_h // 2),
                        0, 0, 360,
                        heat * 255,
                        -1
                    )
        
        # Apply blur for smooth effect
        if heat_layer.max() > 0:
            heat_layer = cv2.GaussianBlur(heat_layer, (31, 31), 0)
        
        # Normalize
        max_val = heat_layer.max()
        if max_val > 0:
            normalized = np.clip(heat_layer / max_val * 255, 0, 255).astype(np.uint8)
        else:
            return frame if not grayscale_bg else base_frame
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Mask - only show where heat exists
        mask = normalized > 10
        
        # Blend
        output = base_frame.copy()
        for c in range(3):
            output[:,:,c] = np.where(
                mask,
                np.clip(base_frame[:,:,c] * (1 - alpha) + heatmap_colored[:,:,c] * alpha, 0, 255).astype(np.uint8),
                base_frame[:,:,c]
            )
        
        return output
    
    def get_heatmap_only(self):
        """Get just the heatmap (no background) - for saving"""
        heat_layer = np.zeros((self.height, self.width), dtype=np.float32)
        
        for track_id, trail in self.person_trails.items():
            for i, point in enumerate(trail):
                x, y = point['x'], point['y']
                w, h = point['w'], point['h']
                heat = point['heat']
                
                center_x = x
                center_y = y + h // 4
                
                size_factor = 0.5 + (i / max(1, len(trail))) * 0.5
                ellipse_w = int(w * 0.6 * size_factor)
                ellipse_h = int(h * 0.8 * size_factor)
                
                if ellipse_w > 0 and ellipse_h > 0:
                    cv2.ellipse(
                        heat_layer,
                        (center_x, center_y),
                        (ellipse_w // 2, ellipse_h // 2),
                        0, 0, 360,
                        heat * 255,
                        -1
                    )
        
        if heat_layer.max() > 0:
            heat_layer = cv2.GaussianBlur(heat_layer, (31, 31), 0)
            normalized = np.clip(heat_layer / heat_layer.max() * 255, 0, 255).astype(np.uint8)
        else:
            normalized = np.zeros((self.height, self.width), dtype=np.uint8)
        
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    def get_heatmap_with_background(self, frame, alpha=0.6):
        """Get heatmap overlaid on original frame (color, not grayscale) - for saving"""
        heat_layer = np.zeros((self.height, self.width), dtype=np.float32)
        
        for track_id, trail in self.person_trails.items():
            for i, point in enumerate(trail):
                x, y = point['x'], point['y']
                w, h = point['w'], point['h']
                heat = point['heat']
                
                center_x = x
                center_y = y + h // 4
                
                size_factor = 0.5 + (i / max(1, len(trail))) * 0.5
                ellipse_w = int(w * 0.6 * size_factor)
                ellipse_h = int(h * 0.8 * size_factor)
                
                if ellipse_w > 0 and ellipse_h > 0:
                    cv2.ellipse(
                        heat_layer,
                        (center_x, center_y),
                        (ellipse_w // 2, ellipse_h // 2),
                        0, 0, 360,
                        heat * 255,
                        -1
                    )
        
        if heat_layer.max() > 0:
            heat_layer = cv2.GaussianBlur(heat_layer, (31, 31), 0)
            normalized = np.clip(heat_layer / heat_layer.max() * 255, 0, 255).astype(np.uint8)
        else:
            return frame.copy()
        
        heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        mask = normalized > 10
        
        output = frame.copy()
        for c in range(3):
            output[:,:,c] = np.where(
                mask,
                np.clip(frame[:,:,c] * (1 - alpha) + heatmap_colored[:,:,c] * alpha, 0, 255).astype(np.uint8),
                frame[:,:,c]
            )
        
        return output
    
    def reset(self):
        """Clear all trails"""
        self.person_trails.clear()
        self.person_boxes.clear()
# ============================================================
# IMPROVED PERSON TRACKER - Handles overlaps better
# ============================================================

# ============================================================
# ADVANCED PERSON TRACKER - Handles overlaps & occlusions
# ============================================================

class PersonTracker:
    """
    Advanced person tracking with:
    - Position prediction during occlusion
    - Appearance matching (color histogram)
    - Re-identification after overlap
    """
    
    def __init__(self, max_lost_frames=60, match_threshold=100):
        """
        Args:
            max_lost_frames: How many frames to remember a lost person
            match_threshold: Max distance to re-match a lost person
        """
        self.max_lost_frames = max_lost_frames
        self.match_threshold = match_threshold
        self.tracked_persons = {}
        self.next_custom_id = 10000  # For re-identified persons
        
    def update(self, boxes, ids, frame=None):
        """
        Update tracking with new detections.
        Uses appearance + position for better tracking.
        """
        current_time = time.time()
        current_ids = set()
        matched_tracks = set()
        
        for box, track_id in zip(boxes, ids):
            track_id = int(track_id)
            current_ids.add(track_id)
            
            cx = int((box[0] + box[2]) // 2)
            cy = int((box[1] + box[3]) // 2)
            box = [int(b) for b in box]
            
            # Extract appearance if frame available
            appearance = None
            if frame is not None:
                appearance = self._extract_appearance(frame, box)
            
            if track_id in self.tracked_persons:
                # Update existing person
                self._update_person(track_id, cx, cy, box, appearance, current_time)
                matched_tracks.add(track_id)
                
            else:
                # Check if this is a re-appeared lost person
                matched_lost_id = self._find_matching_lost_person(cx, cy, appearance)
                
                if matched_lost_id is not None:
                    # Re-identified! Update the old track
                    self._update_person(matched_lost_id, cx, cy, box, appearance, current_time)
                    self.tracked_persons[matched_lost_id]['reidentified'] = True
                    matched_tracks.add(matched_lost_id)
                else:
                    # Truly new person
                    self._create_person(track_id, cx, cy, box, appearance, current_time)
                    matched_tracks.add(track_id)
        
        # Handle lost persons
        for track_id in list(self.tracked_persons.keys()):
            if track_id not in matched_tracks:
                person = self.tracked_persons[track_id]
                person['lost_frames'] += 1
                person['status'] = 'lost'
                
                # Predict position based on velocity
                if person['velocity'] != (0, 0):
                    vx, vy = person['velocity']
                    px, py = person['predicted_pos']
                    decay = 0.85 ** person['lost_frames']
                    person['predicted_pos'] = (
                        int(px + vx * decay),
                        int(py + vy * decay)
                    )
                
                # Remove if lost too long
                if person['lost_frames'] > self.max_lost_frames:
                    del self.tracked_persons[track_id]
        
        return self.tracked_persons
    
    def _extract_appearance(self, frame, box):
        """Extract color histogram as appearance descriptor"""
        try:
            x1, y1, x2, y2 = box
            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract person region (middle portion - mainly body/clothes)
            roi_y1 = y1 + (y2 - y1) // 4  # Skip head
            roi_y2 = y1 + 3 * (y2 - y1) // 4  # Skip feet
            roi = frame[roi_y1:roi_y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Calculate color histogram (HSV is better for color matching)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
            
        except Exception:
            return None
    
    def _compare_appearance(self, hist1, hist2):
        """Compare two appearance histograms. Returns similarity 0-1."""
        if hist1 is None or hist2 is None:
            return 0.0
        try:
            # Correlation method - higher is more similar
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0.0, similarity)
        except Exception:
            return 0.0
    
    def _find_matching_lost_person(self, cx, cy, appearance):
        """Try to match a detection with a recently lost person"""
        best_match_id = None
        best_score = 0.0
        
        for track_id, person in self.tracked_persons.items():
            if person['lost_frames'] == 0:
                continue  # Only check lost persons
            
            if person['lost_frames'] > self.max_lost_frames // 2:
                continue  # Don't match very old lost tracks
            
            # Get predicted position
            px, py = person['predicted_pos']
            
            # Calculate distance
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            
            if dist > self.match_threshold:
                continue  # Too far
            
            # Calculate appearance similarity
            appearance_score = self._compare_appearance(appearance, person.get('appearance'))
            
            # Combined score (position + appearance)
            # Closer = better, more similar appearance = better
            position_score = 1.0 - (dist / self.match_threshold)
            combined_score = position_score * 0.4 + appearance_score * 0.6
            
            if combined_score > best_score and combined_score > 0.3:
                best_score = combined_score
                best_match_id = track_id
        
        return best_match_id
    
    def _create_person(self, track_id, cx, cy, box, appearance, current_time):
        """Create a new tracked person"""
        self.tracked_persons[track_id] = {
            'id': track_id,
            'first_seen': current_time,
            'last_seen': current_time,
            'last_pos': (cx, cy),
            'predicted_pos': (cx, cy),
            'box': box,
            'lost_frames': 0,
            'speed': 0,
            'velocity': (0, 0),
            'path': [(cx, cy)],
            'status': 'active',
            'appearance': appearance,
            'reidentified': False,
            'total_time_tracked': 0
        }
    
    def _update_person(self, track_id, cx, cy, box, appearance, current_time):
        """Update an existing tracked person"""
        person = self.tracked_persons[track_id]
        
        # Calculate velocity
        if person['last_pos'] is not None:
            dx = cx - person['last_pos'][0]
            dy = cy - person['last_pos'][1]
            speed = np.sqrt(dx**2 + dy**2)
            person['velocity'] = (dx, dy)
        else:
            speed = 0
            person['velocity'] = (0, 0)
        
        # Update tracking time
        if person['status'] == 'lost':
            # Just re-appeared
            person['reidentified'] = True
        
        person['last_pos'] = (cx, cy)
        person['predicted_pos'] = (cx, cy)
        person['box'] = box
        person['last_seen'] = current_time
        person['lost_frames'] = 0
        person['speed'] = speed
        person['status'] = 'active'
        person['total_time_tracked'] = current_time - person['first_seen']
        
        # Update appearance (moving average)
        if appearance is not None:
            if person['appearance'] is not None:
                # Blend old and new appearance
                person['appearance'] = person['appearance'] * 0.7 + appearance * 0.3
            else:
                person['appearance'] = appearance
        
        # Update path
        person['path'].append((cx, cy))
        if len(person['path']) > 150:
            person['path'].pop(0)
    
    def get_active_persons(self):
        """Get currently visible persons"""
        return {
            tid: p for tid, p in self.tracked_persons.items() 
            if p['lost_frames'] == 0
        }
    
    def get_lost_persons(self):
        """Get recently lost persons"""
        return {
            tid: p for tid, p in self.tracked_persons.items() 
            if 0 < p['lost_frames'] <= self.max_lost_frames
        }
    
    def get_person_info(self, track_id):
        """Get detailed info for a specific person"""
        if track_id in self.tracked_persons:
            p = self.tracked_persons[track_id]
            return {
                'id': track_id,
                'status': p['status'],
                'speed': round(p['speed'], 1),
                'pos': p['last_pos'] if p['lost_frames'] == 0 else p['predicted_pos'],
                'lost_frames': p['lost_frames'],
                'time_tracked': round(p['total_time_tracked'], 1),
                'reidentified': p.get('reidentified', False),
                'box': p['box']
            }
        return None
    
    def is_target_lost(self, track_id):
        """Check if target is currently lost"""
        if track_id in self.tracked_persons:
            return self.tracked_persons[track_id]['lost_frames'] > 0
        return True
    
    def get_predicted_position(self, track_id):
        """Get predicted position for a lost target"""
        if track_id in self.tracked_persons:
            return self.tracked_persons[track_id]['predicted_pos']
        return None
    
