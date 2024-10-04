import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import os

class model():
    def __init__(self, model_path = "models\plate_new.pt"):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        # Load the YOLOv8 model
        self.detection_model = YOLO(model_path)  # Replace with your trained YOLO model

        # Initialize PaddleOCR for English language
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def detect(self, iou_threshold = 0.3, conf_threshold = 0.5):
        return self.detection_model(frame, conf=conf_threshold, iou=iou_threshold)
    
    def ocr(self, cropped_image):
        return self.ocr.ocr(cropped_image)

