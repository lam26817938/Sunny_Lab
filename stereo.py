import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

from streamclass import VideoStream2cam

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', default='custom_model_lite2')
parser.add_argument('--graph', default='detect.tflite')
parser.add_argument('--labels', default='labelmap.txt')
parser.add_argument('--threshold', default=0.2)
parser.add_argument('--resolution', default='640x480')
parser.add_argument('--edgetpu', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU:
    if GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

videostream = VideoStream2cam(resolution=(imW,imH), framerate=30).start()
time.sleep(1)


while True:

    frame0, frame1 = videostream.read()


    frame_rgb_0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    frame_rgb_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)


    frame_resized_0 = cv2.resize(frame_rgb_0, (width, height))
    frame_resized_1 = cv2.resize(frame_rgb_1, (width, height))


    input_data_0 = np.expand_dims(frame_resized_0, axis=0)

    if floating_model:
        input_data_0 = (np.float32(input_data_0) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data_0)
    interpreter.invoke()


    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence


    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:

            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(max(1, xmin * imW))
            xmax = int(min(imW, xmax * imW))
            ymin = int(max(1, ymin * imH))
            ymax = int(min(imH, ymax * imH))

            cv2.rectangle(frame0, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

            object_name = labels[int(classes[i])] 
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame0, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame0, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            print(f"Detected {object_name} with confidence {scores[i]*100:.2f}%")



    cv2.imshow('Object Detector - Camera 0', frame0)
    cv2.imshow('Camera 1', frame1)


    if cv2.waitKey(1) == ord('q'):
        break


videostream.stop()
cv2.destroyAllWindows()