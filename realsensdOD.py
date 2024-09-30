import numpy as np
import cv2 as cv
import os
import time
from tensorflow.lite.python.interpreter import Interpreter
import pyrealsense2 as rs

# Function to find the depth using disparity
def find_depth(circle_right, circle_left, frame_right, frame_left, baseline, f_pixel, alpha):
    x_right = circle_right[0]
    x_left = circle_left[0]

    # Calculate disparity
    disparity = x_left - x_right

    # Calculate depth z
    if disparity != 0:
        zDepth = (baseline * f_pixel) / disparity  # Depth in [cm]
    else:
        zDepth = 0  # Avoid division by zero

    return abs(zDepth)

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the pipeline
profile = pipeline.start(config)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Get intrinsics for calculating depth
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx = intrinsics.fx
fy = intrinsics.fy
cx = intrinsics.ppx
cy = intrinsics.ppy
f_pixel = fx

# Camera parameters
baseline = 5  # in cm (adjust according to your setup)
alpha = 69  # Camera field of view in the horizontal plane [degrees]

# Load TFLite model and label map
CWD_PATH = os.getcwd()
MODEL_NAME = 'custom_model_lite7'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

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

num = 0
last_print_time = 0

try:
    while True:
        # Get frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run your detection model on the color image
        frame_rgb = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
        frame_resized = cv.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence

        circles_left = []
        circles_right = []
        scores_left = []
        scores_right = []

        for i in range(len(scores)):
            if scores[i] > 0.1:
                ymin, xmin, ymax, xmax = boxes[i]
                dxmin = int(max(0, xmin * 1280))
                dxmax = int(min(1280, xmax * 1280))
                dymin = int(max(0, ymin * 720))
                dymax = int(min(720, ymax * 720))

                center_x = (dxmin + dxmax) / 2
                center_y = (dymin + dymax) / 2

                # Store left and right circles separately
                circles_left.append((center_x, center_y))
                scores_left.append(scores[i])

                cv.rectangle(color_image, (dxmin, dymin), (dxmax, dymax), (0, 255, 0), 2)
                object_name = labels[int(classes[i])]
                confidence = int(scores[i] * 100)
                label = f"{object_name}: {confidence}%"
                cv.putText(color_image, label, (dxmin, dymin - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_time = time.time()
        if current_time - last_print_time >= 1.5:
            timeprint = True

        if circles_left:
            first_match_displayed = False
            for i in range(len(circles_left)):
                left_circle = circles_left[i]
                confidence_left = scores_left[i] * 100
                # Get depth from RealSense for this point
                depth_value = depth_frame.get_distance(int(left_circle[0]), int(left_circle[1])) * 100  # Convert meters to cm
                
                u, v = left_circle
                pixel_size = 2 * depth_value * np.tan(np.radians(alpha / 2)) / 1280
                x = (u - (1280 / 2)) * pixel_size
                y = -(v - (720 / 2)) * pixel_size
                z = round(depth_value, 1)
                coords_text = f"Coords: ({x:.1f}, {y:.1f}, {z:.1f})"

                if not first_match_displayed:
                    cv.putText(color_image, coords_text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    first_match_displayed = True

                if timeprint:
                    print(f"Point {i + 1}: Coords: ({x:.1f}, {y:.1f}, {z:.1f}), Confidence Left: {confidence_left:.1f}%")
                    last_print_time = current_time

        timeprint = False
        cv.imshow("RealSense Frame", color_image)

        # Press 's' to save an image
        if cv.waitKey(1) & 0xFF == ord('s'):
            cv.imwrite(f'output/img{num}.png', color_image)
            print("image saved!")
            num += 1

        # Press 'q' to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv.destroyAllWindows()