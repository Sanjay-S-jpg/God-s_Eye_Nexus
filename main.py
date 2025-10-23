# Harlin's Project, Brahh.
# Phase 3: Making the AI actually useful.
#
# What's new:
# 1. Proximity Detection: We're implementing your "get-close" rule.
#    - If two people are closer than PROXIMITY_THRESHOLD for more than PROXIMITY_TIME_SECONDS, we raise an alert.
# 2. Basic Heatmap: We're tracking the center point of each person to create a density map.
#    - This shows where people are congregating.
# 3. State Management: We need dictionaries to keep track of who is close to whom and for how long.
#
# This is where your simple rules start to look like an intelligent system.

import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time

# --- Configuration ---
MODEL_NAME = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0

# --- Feature Configuration ---
# Proximity Alert Config
PROXIMITY_THRESHOLD = 150  # Max distance in pixels to be considered "close". TUNE THIS.
PROXIMITY_TIME_SECONDS = 3.0 # How long they need to be close to trigger an alert.

# Heatmap Config
TRAIL_LENGTH = 30 # How many past points to store for each person's trail.

def main():
    """ The main function, now with analysis layers. """
    print("Harlin's Crowd Analyzer starting up, brahh... Let's find some trouble.")
    
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- State Management Dictionaries ---
    # For proximity alerts: { (id1, id2): start_time, ... }
    proximity_timers = {} 
    # For alerts on screen: { (id1, id2): alert_triggered, ... }
    proximity_alerts = defaultdict(bool)
    # For heatmap trails: { id: [ (x,y), (x,y), ... ], ... }
    track_history = defaultdict(list)
    
    print("Webcam feed acquired. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a semi-transparent overlay for the heatmap
        heatmap_overlay = frame.copy()
        alpha = 0.4 # Transparency factor

        # Track persons in the frame
        results = model.track(frame, persist=True, verbose=False, classes=[PERSON_CLASS_ID])

        person_boxes = []
        person_ids = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                person_boxes.append(box)
                person_ids.append(track_id)
                
                # --- Heatmap Logic ---
                x1, y1, x2, y2 = box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                history = track_history[track_id]
                history.append((center_x, center_y))
                if len(history) > TRAIL_LENGTH:
                    history.pop(0)

                # Draw the trail on the overlay
                for point in history:
                    cv2.circle(heatmap_overlay, point, 5, (0, 0, 255), -1)

        # --- Proximity Logic ---
        current_time = time.time()
        active_pairs = set()

        # Iterate through all unique pairs of people
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                id1, box1 = person_ids[i], person_boxes[i]
                id2, box2 = person_ids[j], person_boxes[j]
                
                # Sort IDs to make a consistent key for the dictionary
                pair_key = tuple(sorted((id1, id2)))
                active_pairs.add(pair_key)

                center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
                center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
                
                distance = np.linalg.norm(center1 - center2)

                if distance < PROXIMITY_THRESHOLD:
                    if pair_key not in proximity_timers:
                        proximity_timers[pair_key] = current_time
                    
                    elif current_time - proximity_timers[pair_key] > PROXIMITY_TIME_SECONDS:
                        proximity_alerts[pair_key] = True
                        # Draw a red line between them on the main frame
                        cv2.line(frame, tuple(center1.astype(int)), tuple(center2.astype(int)), (0, 0, 255), 2)
                else:
                    if pair_key in proximity_timers:
                        del proximity_timers[pair_key]
                    proximity_alerts[pair_key] = False

        # --- Cleanup old pairs that are no longer tracked ---
        for pair in list(proximity_timers.keys()):
            if pair not in active_pairs:
                del proximity_timers[pair]
                proximity_alerts[pair] = False

        # Blend the heatmap overlay with the original frame
        frame = cv2.addWeighted(heatmap_overlay, alpha, frame, 1 - alpha, 0)

        # --- Draw UI Elements ---
        # Draw bounding boxes and IDs
        if results[0].boxes.id is not None:
            annotated_frame = results[0].plot(img=frame)
        else:
            annotated_frame = frame

        # Display person count
        info_text = f'People Tracked: {len(person_ids)}'
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display Alert if any proximity alert is active
        if any(proximity_alerts.values()):
            alert_text = "ALERT: Suspicious Close Contact!"
            cv2.putText(annotated_frame, alert_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Harlin\'s Crowd Safety Monitor - Phase 3', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done. You've got alerts, brahh. Now go tune the settings.")


if __name__ == "__main__":
    main()
