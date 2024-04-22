import cv2
import numpy as np

video_capture = cv2.VideoCapture(2)
video_capture2 = cv2.VideoCapture(0)


while True:
    result, video_frame = video_capture.read()  # read frames from the video
    result2, video_frame2 = video_capture2.read() 
    if result is False:
        break  # terminate the loop if the frame is not read successfully
    if video_frame is None or video_frame.size == 0:
        print("Error: The frame is empty")
        continue
    cv2.imshow(
        "USB Camera Test", video_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()