import ultralytics
from ultralytics import YOLO
import torch


if __name__ == '__main__':
    
    model = YOLO("models/yolov8n-seg.pt")
    
    # Use the model
    model.train(data="D:\\Cornell\\Lab\\datasets\\Segment\\Bootle_Segment.v2i.yolov8-obb\\data.yaml", epochs=25, imgsz=640)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="tflite")  # export the model to ONNX format
    