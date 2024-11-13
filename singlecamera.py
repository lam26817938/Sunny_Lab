import numpy as np
import cv2 as cv
import os, time
import serial
from tensorflow.lite.python.interpreter import Interpreter

using_arduino=True
if using_arduino:
    arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)

# Load the model and label map
CWD_PATH = os.getcwd()
MODEL_NAME = 'custom_model_lite'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

calibration_data = np.load('CameraCalibration-main/stereo_calibration_data_720.npz')
cameraMatrix = calibration_data['cameraMatrixR']
f_x = cameraMatrix[0, 0]
f_y = cameraMatrix[1, 1] 


def calculate_iou(box1, box2):
    # Box format: (ymin, xmin, ymax, xmax)
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    
    # Calculate intersection area
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # Calculate union area
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = area1 + area2 - inter_area

    # IoU calculation
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    selected_indices = []
    
    while indices:
        current_index = indices.pop(0)
        selected_indices.append(current_index)
        
        remove_indices = []
        for i in indices:
            iou = calculate_iou(boxes[current_index], boxes[i])
            if iou > iou_threshold:
                remove_indices.append(i)
        
        indices = [i for i in indices if i not in remove_indices]
    
    return selected_indices

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
actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {actual_width}x{actual_height}")

last_print_time = time.time()
recording = False
out = None

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

    current_time = time.time()
    detected_boxes = []
    detected_scores = []
    detected_classes = []

    # Collect all the detected boxes, scores, and classes
    for i in range(len(scores)):
        if scores[i] > 0.15:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            detected_boxes.append([ymin, xmin, ymax, xmax])
            detected_scores.append(scores[i])
            detected_classes.append(classes[i])

    # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    selected_indices = non_max_suppression(detected_boxes, detected_scores, iou_threshold=0.1)

    detected_points = []
    
    # Process only the selected boxes after NMS
    for i in selected_indices:
        ymin, xmin, ymax, xmax = detected_boxes[i]

        dxmin = int(max(0, xmin * imW))
        dxmax = int(min(imW, xmax * imW))
        dymin = int(max(0, ymin * imH))
        dymax = int(min(imH, ymax * imH))
        
        center_x = (dxmin + dxmax) / 2
        object_width = dxmax - dxmin
        
        if object_width > 0:  
            offset_from_center = center_x - (imW / 2)

            angle = round((offset_from_center / (imW / 2)) * (camera_fov / 2), 1)
            
            y_position = dymax

            # phi = (y_position / imH) * 44
            # phi = np.radians(phi)

            # distance = 7 / (2 * np.tan(np.radians(30)+phi))
            distance = (f_y * 7) / dymax
            


            object_name = labels[int(detected_classes[i])]
            confidence = int(detected_scores[i] * 100)
            
            detected_points.append({
                "confidence": confidence,
                "distance": distance,
                "angle": angle
            })
            
            label = f"{object_name}: {confidence}%, Distance: {round(distance, 1)} cm, Angle: {angle}"
            cv.rectangle(frame, (dxmin, dymin), (dxmax, dymax), (0, 255, 0), 2)
            cv.putText(frame, label, (dxmin, dymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if current_time - last_print_time >= 1.0:
        if detected_points:
            print("Detected points:")
            m=0
            for point in detected_points:
                if using_arduino:
                    if point['confidence']>m:
                        m = point['confidence']
                        message = f"C:{point['confidence']}%, Distance: {point['distance']} cm, Angle: {point['angle']}\n"
                    arduino.write(message.encode())
                print(f"Confidence: {point['confidence']}%, Distance: {point['distance']} cm, Angle: {point['angle']} degrees")
            if m:
                arduino.write(message.encode())
            print("No objects detected.")
        last_print_time = current_time

    cv.imshow("Camera Feed", frame)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord('v'):
        if not recording:
            # Start recording
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out = cv.VideoWriter(f'output_{timestamp}.mp4', 
                                 cv.VideoWriter_fourcc(*'mp4v'),  # MP4 codec
                                 20.0, (imW, imH))
            print("Recording started...")
            recording = True
        else:
            # Stop recording
            out.release()
            print("Recording stopped.")
            recording = False

    # Write the frame to the video file if recording
    if recording and out is not None:
        out.write(frame)

    # Exit loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close windows
cap.release()
cv.destroyAllWindows()