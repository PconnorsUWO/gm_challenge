#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
import helpers  # Ensure this module is available in your PYTHONPATH

# Load your model
model = YOLO("src/models/model_name.pt")

def process_directory(input_dir, results_path):
    # Load images along with their file paths
    image_tuples = helpers.load_and_scale_images_from_directory(input_dir)
    results_list = []
    
    # Extract just the image objects for prediction
    images = [img for (_, img) in image_tuples]
    
    results = model.predict(source=images, classes=[0, 1, 2, 3, 5, 7], conf=0.20)
    
    for (file_path, _), result in zip(image_tuples, results):
        boxes = result.boxes
        boundaries_arr = boxes.xyxy  # tensor-like structure with coordinates
        classes_arr = boxes.cls
        confidences_arr = boxes.conf

        for boundary, cls, conf in zip(boundaries_arr.tolist(), classes_arr.tolist(), confidences_arr.tolist()):
            x1, y1, x2, y2 = map(int, boundary)
            file_name = os.path.basename(file_path)
            
            # Adjust classes as needed
            if cls == 3:
                cls = 1
            elif cls in [5, 7]:
                cls = 2
            results_list.append((file_name, int(cls), conf, x1, y1, x2, y2))
    
    with open(results_path, 'w') as out_file:
        for file_name, cls, conf, x1, y1, x2, y2 in results_list:
            out_file.write(f"{file_name} {cls + 1} {round(float(conf),2)} {x1} {y1} {x2} {y2}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline for YOLO detection')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to the folder containing input images')
    parser.add_argument('--results_folder', type=str, required=True,
                        help='Path to the output results file')
    args = parser.parse_args()
    
    process_directory(args.image_folder, args.results_folder)
