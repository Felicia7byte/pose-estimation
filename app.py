import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import av

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
)

st.title("AI Pose Estimation")

@st.cache_resource
def load_model():
    return YOLO("yolo26n-pose.pt")

model = load_model()

menu = st.sidebar.radio(
    "Pilih Mode",
    ["📷 Photo", "🎥 Video", "📹 Webcam"]
)

# PHOTO
if menu == "📷 Photo":
    st.header("📷 Pose Estimation - Photo")

    uploaded_file = st.file_uploader(
        "Upload photo",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        st.image(image, use_container_width=True)

        if st.button("Estimate Pose", type="primary"):

            results = model(image)

            result_image = cv2.cvtColor(
                results[0].plot(),
                cv2.COLOR_BGR2RGB
            )

            st.image(
                result_image,
                use_container_width=True
            )

# VIDEO
elif menu == "🎥 Video":

    st.header("🎥 Pose Estimation - Video")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        # Video temporarily saved
        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.video(temp_video.name)

        if st.button("Estimate Pose", type="primary"):

            st.info("Processing video...")

            cap = cv2.VideoCapture(temp_video.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model(frame)

                result_frame = results[0].plot()

                out.write(result_frame)

            cap.release()
            out.release()

            st.success("Pose estimation completed!")

            st.video(output_path)

# WEBCAM
elif menu == "📹 Webcam":
    st.header("📹 Pose Estimation - Live Webcam")

    st.write(
        "Point the camera at your body. "
        "YOLO will detect your pose in real time."
    )

    # VIDEO PROCESSOR
    class PoseProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model


        def recv(self, frame):
            # Frame from webcam
            img = frame.to_ndarray(format="bgr24")

            # YOLO Pose
            results = self.model(
                img,
                verbose=False
            )

            # Plot skeleton
            annotated_frame = results[0].plot()

            # Return the frame
            return av.VideoFrame.from_ndarray(
                annotated_frame,
                format="bgr24"
            )

    # WEBRTC
    RTC_CONFIGURATION = {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }

    webrtc_streamer(
        key="pose-webcam",
        video_processor_factory=PoseProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )
