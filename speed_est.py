import cv2
import pandas as pd
import math
from ultralytics import YOLO
from tracker import *

# Load YOLO model
model = YOLO('yolov8s.pt')
class_list = model.names
tracker = Tracker()

# Line Y-coordinates
line_a = 198
line_b = 268

# Track info
entry_time, exit_time = {}, {}
entry_pos, exit_pos = {}, {}
direction_dict = {}
count_ab, count_ba = 0, 0
count = 0
writer_initialized = False
out = None

# Open video
cap = cv2.VideoCapture('E:/Projects(DS)/vehicle speed detection/highway.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing full video.")
        break

    count += 1
    frame = cv2.resize(frame, (1020, 500))  # Standardize resolution

    # === Initialize writer after knowing final frame size ===
    if not writer_initialized:
        h, w, _ = frame.shape
        out = cv2.VideoWriter('annotated_output.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (w, h))
        writer_initialized = True

    # Run YOLOv8
    results = model(frame, conf=0.3, verbose=False)
    boxes = results[0].boxes.data

    if boxes is None or len(boxes) == 0:
        out.write(frame)
        continue

    boxes = boxes.detach().cpu().numpy()
    df = pd.DataFrame(boxes).astype("float")
    detections = []

    for index, row in df.iterrows():
        x1, y1, x2, y2, _, cls_id = map(int, row[:6])
        cls_name = class_list[cls_id]
        if cls_name == 'car':
            detections.append([x1, y1, x2, y2])

    bbox_ids = tracker.update(detections)

    for bbox in bbox_ids:
        x3, y3, x4, y4, obj_id = bbox
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)

        # Entry
        if line_a - 10 <= cy <= line_a + 10 and obj_id not in entry_time:
            entry_time[obj_id] = count
            entry_pos[obj_id] = (cx, cy)
            direction_dict[obj_id] = 'A'

        elif line_b - 10 <= cy <= line_b + 10 and obj_id not in entry_time:
            entry_time[obj_id] = count
            entry_pos[obj_id] = (cx, cy)
            direction_dict[obj_id] = 'B'

        # Exit
        if (line_b - 10 <= cy <= line_b + 10 and obj_id in entry_time and obj_id not in exit_time and
            direction_dict[obj_id] == 'A'):
            exit_time[obj_id] = count
            exit_pos[obj_id] = (cx, cy)
            count_ab += 1

        elif (line_a - 10 <= cy <= line_a + 10 and obj_id in entry_time and obj_id not in exit_time and
              direction_dict[obj_id] == 'B'):
            exit_time[obj_id] = count
            exit_pos[obj_id] = (cx, cy)
            count_ba += 1

        # Speed
        if obj_id in entry_time and obj_id in exit_time:
            dx = exit_pos[obj_id][0] - entry_pos[obj_id][0]
            dy = exit_pos[obj_id][1] - entry_pos[obj_id][1]
            pixel_dist = math.hypot(dx, dy)
            distance_m = pixel_dist / 8
            time_s = (exit_time[obj_id] - entry_time[obj_id]) / fps

            if time_s > 0:
                speed_kmph = (distance_m / time_s) * 3.6
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                cv2.putText(frame, f"{int(speed_kmph)} km/h", (x3, y3 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"ID:{obj_id}", (x3, y3 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Draw lines and counts
    cv2.line(frame, (172, line_a), (774, line_a), (255, 0, 0), 2)
    cv2.line(frame, (8, line_b), (927, line_b), (0, 255, 0), 2)
    cv2.putText(frame, "Line A", (172, line_a - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, "Line B", (8, line_b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"A → B: {count_ab}", (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"B → A: {count_ba}", (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Vehicle Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
