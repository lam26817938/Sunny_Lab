import numpy as np
import cv2 as cv
import os, time
from tensorflow.lite.python.interpreter import Interpreter

# Load the model and label map
CWD_PATH = os.getcwd()
MODEL_NAME = 'custom_model_lite'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Camera settings
imW, imH = 1280, 720
camera_fov = 78

# Load label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Open the camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, imW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, imH)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Prepare the frame for inference
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_resized = cv.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve the detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence scores

    for i in range(len(scores)):
        if scores[i] > 0.15:
            ymin, xmin, ymax, xmax = boxes[i]
            dxmin = int(max(0, xmin * imW))
            dxmax = int(min(imW, xmax * imW))
            dymin = int(max(0, ymin * imH))
            dymax = int(min(imH, ymax * imH))

            # Calculate the center of the detected object
            center_x = (dxmin + dxmax) / 2
            object_width = dxmax - dxmin

            # Calculate the horizontal angle
            offset_from_center = center_x - (imW / 2)
            angle = round((offset_from_center / (imW / 2)) * (camera_fov / 2), 1)

            # Estimate the distance based on the object's size
            distance = round(1000 / object_width, 1)  # Adjust scaling factor as needed

            # Display the results on the frame
            object_name = labels[int(classes[i])]
            confidence = int(scores[i] * 100)
            
            label = f"{object_name}: {confidence}%, Distance: {distance} cm, Angle: {angle}"
            cv.rectangle(frame, (dxmin, dymin), (dxmax, dymax), (0, 255, 0), 2)
            cv.putText(frame, label, (dxmin, dymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow("Camera Feed", frame)

    # Exit loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close windows
cap.release()
cv.destroyAllWindows()