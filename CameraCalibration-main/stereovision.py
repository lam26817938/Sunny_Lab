import numpy as np
import cv2 as cv
import os
from tensorflow.lite.python.interpreter import Interpreter
from matplotlib import pyplot as plt
import pprint




def find_depth(circle_right, circle_left, frame_right, frame_left, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = circle_right[0]
    x_left = circle_left[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]
    

    return abs(zDepth)




# Open the XML file with stereo rectification maps
cv_file = cv.FileStorage("CameraCalibration-main/StereoMap.xml", cv.FILE_STORAGE_READ)

# Read the stereo rectification maps
stereoMapL_x = cv_file.getNode("StereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("StereoMapL_y").mat()
stereoMapR_x = cv_file.getNode("StereoMapR_x").mat()
stereoMapR_y = cv_file.getNode("StereoMapR_y").mat()

cv_file.release()
CWD_PATH = os.getcwd()
MODEL_NAME='custom_model_lite2'
GRAPH_NAME='detect.tflite'
LABELMAP_NAME='labelmap.txt'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

imW, imH = 640, 480

# Camera parameters (these need to be adjusted based on your setup)
focal_length =  8 # Your camera's focal length
baseline = 5 # The physical distance between cameras
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

c_x,c_y=320,240

# Open both cameras
cap_left = cv.VideoCapture(0, cv.CAP_DSHOW)  # Adjust the index to match your left camera
cap_right = cv.VideoCapture(1, cv.CAP_DSHOW)  # Adjust the index to match your right camera

while cap_left.isOpened() and cap_right.isOpened():
    success_left, frame_left = cap_left.read()
    success_right, frame_right = cap_right.read()

    # Undistort and rectify images using the stereo rectification maps
    frame_left_rectified = cv.remap(frame_left, stereoMapL_x, stereoMapL_y, cv.INTER_LINEAR)
    frame_right_rectified = cv.remap(frame_right, stereoMapR_x, stereoMapR_y, cv.INTER_LINEAR)
    
    circle_left=None
    circle_right=None
    
    for side, frame_rectified in [('left', frame_left_rectified), ('right', frame_right_rectified)]:
        frame_rgb = cv.cvtColor(frame_rectified, cv.COLOR_BGR2RGB)
        frame_resized = cv.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 解析检测结果
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence

        # 假设左右图像都检测到了对象，你需要根据实际情况进行调整
        for i in range(len(scores)):
            if scores[i] > 0.2:
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(max(0, xmin * imW))
                xmax = int(min(imW, xmax * imW))
                ymin = int(max(0, ymin * imH))
                ymax = int(min(imH, ymax * imH))
                
                if side=='left':
                    circle_left=(xmin + xmax / 2, ymin + ymax / 2)
                else:
                    circle_right=(xmin + xmax / 2, ymin + ymax / 2)
                    
                cv.rectangle(frame_rectified, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                object_name = labels[int(classes[i])]
                confidence = int(scores[i] * 100)
                label = f"{object_name}: {confidence}%"
                cv.putText(frame_rectified, label, (xmin, ymin-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    if circle_left and circle_right:
        depth = find_depth(circle_left, circle_right, frame_right, frame_left, baseline, focal_length, alpha)

        u, v = circle_left
        
        pixel_size = 2 * depth * np.tan(np.radians(alpha / 2)) / width
        x = (u - c_x) * pixel_size
        y = (v - c_y) * pixel_size
        z = round(depth, 1)
        coords_text = f"Coords: ({x:.1f}, {y:.1f}, {z:.1f})"

        cv.putText(frame_left_rectified, coords_text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(frame_right_rectified, coords_text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
     #   print("Depth: ", str(round(depth,1)))


    cv.imshow("Frame Left", frame_left_rectified)
    cv.imshow("Frame Right", frame_right_rectified)
    
    


    # Wait for the 'q' key to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture objects and close the windows
cap_left.release()
cap_right.release()
cv.destroyAllWindows()