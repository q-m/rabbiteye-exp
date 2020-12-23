#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd

from config import *
from evaluate import iou, yolo_non_max_suppression
from imageai.Detection.Custom import CustomObjectDetection

def load_model():
    dtc = CustomObjectDetection()
    dtc.setModelTypeAsYOLOv3()
    dtc.setModelPath(MODEL_PATH)
    dtc.setJsonPath(JSON_PATH)
    dtc.loadModel()

    return dtc

def predict(model, filename):
    # find detections with ImageAI
    returned_image, detections = model.detectObjectsFromImage(
            input_image=filename,
            output_type='array',
            minimum_percentage_probability=70,
            display_percentage_probability=True,
            display_object_name=True,
            nms_treshold=0.5)

    # custom post-processing of detections
    probs = np.array([d['percentage_probability'] for d in detections])
    boxes = np.array([d['box_points'] for d in detections])
    classes = np.array([d['name'] for d in detections])

    if len(probs) > 0:
        probs, boxes, classes = yolo_non_max_suppression(scores=probs, boxes=boxes, classes=classes)

    # gather post-processed detections in single array
    detections = []
    for i in range(len(probs)):
        detections.append({
            'name': classes[i],
            'percentage_probability': probs[i],
            'box_points': boxes[i]
        })

    return detections

if __name__ == '__main__':
    model = load_model()
    for f in sys.argv[1:]:
        detections = predict(model, f)
        print('--', f)
        for d in detections:
            print(d)

