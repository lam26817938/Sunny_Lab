import numpy as np
import cv2 as cv
import os, time
from tensorflow.lite.python.interpreter import Interpreter
import pickle



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
# cv_file = cv.FileStorage("CameraCalibration-main/StereoMap_480p.xml", cv.FILE_STORAGE_READ)

# # Read the stereo rectification maps
# mapLx = cv_file.getNode("StereoMapL_x").mat()
# mapLy = cv_file.getNode("StereoMapL_y").mat()
# mapRx = cv_file.getNode("StereoMapR_x").mat()
# mapRy = cv_file.getNode("StereoMapR_y").mat()

# with open('stereo_calibration_data.pkl', 'rb') as f:
#     calibration_data = pickle.load(f)

calibration_data = np.load('CameraCalibration-main/stereo_calibration_data_720.npz')

cameraMatrixL = calibration_data['cameraMatrixL']
distCoeffsL = calibration_data['distCoeffsL']
cameraMatrixR = calibration_data['cameraMatrixR']
distCoeffsR = calibration_data['distCoeffsR']
R = calibration_data['R']
T = calibration_data['T']
R1 = calibration_data['R1']
R2 = calibration_data['R2']
P1 = calibration_data['P1']
P2 = calibration_data['P2']
Q = calibration_data['Q']



CWD_PATH = os.getcwd()
MODEL_NAME='custom_model_lite7'
GRAPH_NAME='detect.tflite'
LABELMAP_NAME='labelmap.txt'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

imW, imH = 1280, 720

mapLx, mapLy = cv.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, (imW, imH), cv.CV_32FC1)
mapRx, mapRy = cv.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, (imW, imH), cv.CV_32FC1)

# Camera parameters (these need to be adjusted based on your setup)
focal_length =  8 # Your camera's focal length
baseline = 5 # The physical distance between cameras
alpha = 69        #Camera field of view in the horisontal plane [degrees]

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

num=0
last_print_time=0
# Open both cameras
#cap_left = cv.VideoCapture(2, cv.CAP_DSHOW)  # Adjust the index to match your left camera
#cap_right = cv.VideoCapture(0, cv.CAP_DSHOW)  # Adjust the index to match your right camera
cap_left = cv.VideoCapture(0)  # Adjust the index to match your left camera
cap_right = cv.VideoCapture(1)  # Adjust the index to match your right camera
cap_left.set(cv.CAP_PROP_FRAME_WIDTH, imW)
cap_left.set(cv.CAP_PROP_FRAME_HEIGHT, imH)
cap_right.set(cv.CAP_PROP_FRAME_WIDTH, imW)
cap_right.set(cv.CAP_PROP_FRAME_HEIGHT, imH)
while cap_left.isOpened() and cap_right.isOpened():
    success_left, frame_left = cap_left.read()
    success_right, frame_right = cap_right.read()
    # Undistort and rectify images using the stereo rectification maps
    frame_left_rectified = cv.remap(frame_left, mapLx, mapLy, cv.INTER_LINEAR)
    frame_right_rectified = cv.remap(frame_right, mapRx, mapRy, cv.INTER_LINEAR)
    
    # x_offset = (imW - imH) // 2
    # frame_left_cropped = frame_left_rectified[0:imH, x_offset:x_offset + imH]
    # frame_right_cropped = frame_right_rectified[0:imH, x_offset:x_offset + imH]
    

    circles_left = [] 
    circles_right = []
    scores_left = []
    scores_right = []
    
    for side, frame_rectified in [('left', frame_left_rectified), ('right', frame_right_rectified)]:
        frame_rgb = cv.cvtColor(frame_rectified, cv.COLOR_BGR2RGB)
        frame_resized = cv.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

       
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence
        
        for i in range(len(scores)):
            if scores[i] > 0.1:
                ymin, xmin, ymax, xmax = boxes[i]
                dxmin = int(max(0, xmin * imW))
                dxmax = int(min(imW, xmax * imW))
                dymin = int(max(0, ymin * imH))
                dymax = int(min(imH, ymax * imH))
                
                if side == 'left':
                    circles_left.append(((dxmin + dxmax) / 2, (dymin + dymax) / 2))
                    scores_left.append(scores[i])
                else:
                    circles_right.append(((dxmin + dxmax) / 2, (dymin + dymax) / 2))
                    scores_right.append(scores[i])
                    
                cv.rectangle(frame_rectified, (dxmin, dymin), (dxmax, dymax), (0, 255, 0), 2)

                object_name = labels[int(classes[i])]
                confidence = int(scores[i] * 100)
                label = f"{object_name}: {confidence}%"
                cv.putText(frame_rectified, label, (dxmin, dymin-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        point_counter = 1 
        current_time = time.time()
        if current_time - last_print_time >= 1.5:
            timeprint=True
            
        if circles_left and circles_right:
            first_match_displayed = False  # A flag to ensure only the first match is displayed
            for i in range(len(circles_left)-1, -1, -1): 
                left_circle = circles_left[i]
                for j in range(len(circles_right)-1, -1, -1): 
                    right_circle = circles_right[j]
                    
                    confidence_left = scores_left[i] * 100
                    confidence_right = scores_right[j] * 100
                    if abs(confidence_left - confidence_right) < 5:
                        depth = find_depth(left_circle, right_circle, frame_right_rectified, frame_left_rectified, baseline, focal_length, alpha)

                        u, v = left_circle
                        pixel_size = 2 * depth * np.tan(np.radians(alpha / 2)) / width
                        x = (u - (imW/2)) * pixel_size
                        y = -(v - (imH/2)) * pixel_size + 16
                        z = round(depth, 1)
                        coords_text = f"Coords: ({x:.1f}, {y:.1f}, {z:.1f})"

                        # Display the first match on the image
                        if not first_match_displayed:
                            cv.putText(frame_left_rectified, coords_text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv.putText(frame_right_rectified, coords_text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            first_match_displayed = True  # Set the flag to true after displaying the first match

                        if timeprint:
                            print(f"Point {point_counter}: Coords: ({x:.1f}, {y:.1f}, {z:.1f}), Confidence Left: {confidence_left:.1f}%, Confidence Right: {confidence_right:.1f}%")
                            last_print_time = current_time  
                            point_counter += 1

                        
                        circles_left.pop(i)
                        circles_right.pop(j)
                        break  
            timeprint = False


    cv.imshow("Frame Left", frame_left_rectified)
    cv.imshow("Frame Right", frame_right_rectified)
    
    k = cv.waitKey(5)
    if k == ord('s'): # wait for 's' key to save and exit
        cv.imwrite('CameraCalibration-main/test/LC/img' + str(num) + '.png', frame_left_rectified)
        cv.imwrite('CameraCalibration-main/test/RC/img' + str(num) + '.png', frame_right_rectified)
        print("image saved!")
        num += 1

    # Wait for the 'q' key to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture objects and close the windows
cap_left.release()
cap_right.release()
cv.destroyAllWindows()