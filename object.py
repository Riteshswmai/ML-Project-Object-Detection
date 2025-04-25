import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import time

# Enable OpenCV and CUDA optimizations
cv2.setUseOptimized(True)
torch.backends.cudnn.benchmark = True

# Load YOLOv8s model on GPU
model = YOLO("yolov8s.pt")
model.fuse()

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=15)

# Start video stream (change to path for file)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not available")
    exit()

# Output video setup
frame_width, frame_height = 640, 480  # Adjust as necessary
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

# Track object IDs
prev_ids = set()
track_memory = {}

# Variables for FPS tracking
fps_values = []
frame_count = 100  # Number of frames to average over

# Missing Object Detection Settings
missing_threshold = 15  # Number of frames to consider object as missing
missing_objects = {}  # Stores the last time the object was seen

# Object Reappearance Logic
reappeared_objects = {}  # Stores reappeared objects and when they reappeared

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start timer for FPS calculation
    frame_start_time = time.time()

    # Resize frame for speed (optional)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Run YOLOv8 inference
    results = model.predict(frame, device='cuda', conf=0.4, verbose=False)[0]

    # Collect detections for DeepSORT
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltwh()
        label = track.get_det_class()

        current_ids.add(track_id)

        # Check if the object was previously missing and now reappeared
        if track_id in missing_objects and track_id not in reappeared_objects:
            # Highlight in red (reappearing object)
            cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} ID:{track_id} (Reappeared)", (int(l), int(t - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            reappeared_objects[track_id] = time.time()  # Mark as reappeared

        # Draw regular boxes in green for detected objects
        if track_id not in missing_objects:  # Only draw green if not missing
            cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (int(l), int(t - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # New IDs
    new_ids = current_ids - prev_ids
    for new_id in new_ids:
        print(f"➕ New Object Detected: ID {new_id}")

    # Missing IDs
    lost_ids = prev_ids - current_ids
    for lost_id in lost_ids:
        track_memory[lost_id] = track_memory.get(lost_id, 0) + 1
        if track_memory[lost_id] == missing_threshold:
            missing_objects[lost_id] = time.time()  # Mark as missing
            print(f"❌ Missing Object Detected: ID {lost_id}")

    # Reset seen IDs
    for track_id in current_ids:
        track_memory[track_id] = 0

    prev_ids = current_ids.copy()

    # End timer and calculate FPS
    frame_end_time = time.time()
    fps = 1 / (frame_end_time - frame_start_time)
    fps_values.append(fps)

    # If frame count reaches 100, calculate the average FPS
    if len(fps_values) >= frame_count:
        avg_fps = sum(fps_values) / len(fps_values)
        print(f"Average FPS for the last {frame_count} frames: {avg_fps:.2f}")

        # Reset for the next set of frames
        fps_values = []

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the output video
    output_video.write(frame)

    # Show the frame
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

# Print missing object information
print("\nMissing Object Information:")
for obj_id, timestamp in missing_objects.items():
    print(f"Object ID {obj_id} was last seen at {timestamp}")

# Print reappeared object information
print("\nReappeared Object Information:")
for obj_id, timestamp in reappeared_objects.items():
    print(f"Object ID {obj_id} reappeared at {timestamp}")
