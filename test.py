from ultralytics import YOLO
import cv2

model = YOLO("yolo26n-pose.pt")

results = model("face.jpg")

result = results[0].plot()

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()