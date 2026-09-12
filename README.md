# pose-estimation
An AI-powered pose estimation application built with Python and Streamlit using the Ultralytics YOLO Pose model. The application can estimate human poses from photos, videos, and live webcam streams.

# Features
- Estimate human poses from uploaded images
- Process uploaded videos frame by frame
- Perform real-time pose estimation using a webcam
- Draw pose keypoints and skeletons on detected people
- Preview the original and processed media
- Downloadable video processing is not currently included
- Simple interactive interface with multiple modes
# Modes
## Photo
Upload a JPG, JPEG, or PNG image and estimate human poses. The detected poses are visualized with keypoints and skeleton connections directly on the image.

## Video
Upload an MP4, AVI, or MOV video and process it frame by frame. The application runs YOLO Pose inference on each frame and generates a new video containing the detected pose annotations.

## Webcam
Use a live webcam stream for real-time pose estimation. The application uses WebRTC to receive webcam frames, runs YOLO Pose inference on each frame, and displays the annotated result in real time.

# Tech Stack
Python, Streamlit, Ultralytics YOLO, OpenCV, Pillow, NumPy, PyAV, streamlit-webrtc
# How It Works
Photo:
Upload Image → YOLO Pose Model → Pose Detection → Keypoints & Skeleton → Detection Result

Video:
Upload Video → Read Video Frames → YOLO Pose Model → Pose Detection → Annotate Frames → Output Video

Webcam:
Webcam Stream → WebRTC → Video Frame → YOLO Pose Model → Keypoints & Skeleton → Annotated Live Frame

The application uses the pre-trained yolo26n-pose.pt model from Ultralytics to detect human poses. For each detected person, the model estimates body keypoints and visualizes the pose using keypoints and skeleton connections.
# Model
YOLO26 Nano Pose
# Live Demo
https://pose-estimation-yolo.streamlit.app/
