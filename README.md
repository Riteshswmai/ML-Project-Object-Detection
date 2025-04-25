# Real-Time Object Detection and Tracking using YOLOv8 and DeepSORT

## Introduction

This system is designed for real-time object detection and tracking using a combination of two powerful algorithms: **YOLOv8** and **DeepSORT**. The primary purpose is to identify and track objects in video streams (either from a live camera or a video file) for applications like surveillance, automated inventory management, or object tracking in various environments.

## System Components

### 1. **YOLOv8 (You Only Look Once - Version 8)**

YOLOv8 is a state-of-the-art deep learning model for real-time object detection. It can identify and locate objects within an image or video frame by predicting bounding boxes and classifying objects.

### 2. **DeepSORT (Deep Learning-based SORT)**

DeepSORT is an advanced tracking algorithm that builds on the Simple Online and Realtime Tracking (SORT) algorithm by incorporating deep learning techniques. It tracks multiple objects across frames, assigning a unique ID to each object and updating their positions over time.

## System Workflow

1. **Input**: A video stream (either from a webcam or a video file).
2. **Object Detection**: YOLOv8 processes each frame of the video to detect objects. It outputs bounding boxes, class labels, and confidence scores for the detected objects.
3. **Object Tracking**: DeepSORT receives these bounding boxes and associates them with tracked objects from previous frames. It assigns unique IDs to each object and tracks their movement over time.
4. **Output**: The system displays the video with the tracked objects, drawing bounding boxes around them and labeling each one with its class and unique ID. Missing objects are detected, and reappearing objects are highlighted in **red**.

## FPS Achieved

During the tests, the system was able to achieve an average **FPS** of **X.XX FPS**. This was measured over a batch of 100 frames.

![FPS Average](https://github.com/Riteshswmai/ML-Project-Object-Detection/blob/main/fps_avg.png)


## Hardware Configuration

The following hardware configuration was used during testing:

- **CPU**: Ryzen 7 5000 series
- **GPU**: NVIDIA GeForce RTX 3050 (4GB VRAM)
- **RAM**: 16 GB DDR4

## Video Link

You can view the output video of the system in action below:

![Output Video](https://github.com/Riteshswmai/ML-Project-Object-Detection/blob/main/output_video.mp4)


## Additional Techniques, Optimizations, or Architectural Decisions

### 1. **Optimization with OpenCV and CUDA**

- **OpenCV Optimizations**: Enabled OpenCV optimizations using `cv2.setUseOptimized(True)` to improve frame processing speed.
- **CUDA Optimization**: Utilized `torch.backends.cudnn.benchmark = True` to optimize the performance on GPU for real-time inference.

### 2. **Model and Tracker Initialization**

- **YOLOv8**: The lightweight YOLOv8 model (`yolov8s.pt`) was used to detect objects in real-time. This model was chosen to balance performance and accuracy.
- **DeepSORT**: DeepSORT was used for object tracking. The tracker was configured with a maximum age of 15 frames, allowing it to track objects for 15 frames before considering them lost. This parameter was chosen to balance responsiveness and stability.

### 3. **Tracking Memory System**

- **Tracking Memory**: Implemented a tracking memory system to detect when an object is missing for more than a specified number of frames. Objects that disappear for too long are marked as "missing" and can be detected again when they reappear in the frame. This system helps improve the robustness of the tracker.

