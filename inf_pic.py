import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the YOLOv8 model
model = YOLO('/Users/nguyendang/license-plate-recognition/models/best.pt') 

# Initialize PaddleOCR for English language
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load the image file
image_path = '/Users/nguyendang/licensce-plate/car_long/6759.jpg'
frame = cv2.imread(image_path)

# Set IoU threshold and confidence threshold for NMS
iou_threshold = 0.3  # IoU threshold (you can adjust this)
conf_threshold = 0.5  # Confidence threshold (you can adjust this)

# Process the image frame
def process_frame(frame):
    # Run YOLOv8 inference with custom IoU and confidence threshold for NMS
    results = model(frame, conf=conf_threshold, iou=iou_threshold)
    detected_objects = []
    
    # Loop over the detected objects
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = box.cls[0]

            # Crop the detected object (license plate area)
            cropped_image = frame[y1:y2, x1:x2]

            # Use PaddleOCR to detect text in the cropped image
            ocr_result = ocr.ocr(cropped_image)

            detected_objects.append((x1, y1, x2, y2, ocr_result))
    
    return detected_objects

# Process the image frame and retrieve detected objects
detected_objects = process_frame(frame)

# Loop through the detected objects and draw bounding boxes and text
for (x1, y1, x2, y2, ocr_result) in detected_objects:
    try:
        detected_text = ''.join([text[0] for bounding_box, text in ocr_result[0]])
        # Draw the bounding box around the license plate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add the detected text on the frame with smaller font size and clearer font
        cv2.putText(frame, str(detected_text), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    except:
        # Draw the bounding box around the license plate even if OCR fails
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the processed image with bounding boxes and recognized text
cv2.imshow('YOLOv8 License Plate Detection with OCR', frame)

# Save the processed image
output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, frame)

# Press 'q' to quit the image display window
cv2.waitKey(0)
cv2.destroyAllWindows()