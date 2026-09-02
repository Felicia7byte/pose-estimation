from ultralytics import YOLO
import cv2

model = YOLO("yolo26n-pose.pt")

results = model(source=0, stream=True)

cv2.namedWindow("YOLO Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Pose", 960, 540)

for result in results:
    frame = result.plot()

    cv2.imshow("YOLO Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.getWindowProperty(
        "YOLO Pose",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break

cv2.destroyAllWindows()
