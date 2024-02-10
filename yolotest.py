import cv2
import numpy as np
from ultralytics import YOLO
import ultralytics
from collections import defaultdict

if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    ultralytics.checks()
    model = YOLO("tflite1/detect.tflite")
    model.predict(source="0",show=True,imgsz=640)
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
