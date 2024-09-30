import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the YOLOv8 model
model = YOLO('//Users/nguyendang/licensce-plate/plate_yolov8n_320_2024.pt')  # Replace with your trained YOLO model

# Initialize PaddleOCR for English language
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Open the video file
video_path = '/Users/nguyendang/licensce-plate/test_video/test_6.mp4'
cap = cv2.VideoCapture(video_path)

# Set IoU threshold and confidence threshold for NMS
iou_threshold = 0.3  # IoU threshold (you can adjust this)
conf_threshold = 0.5  # Confidence threshold (you can adjust this)

# Initialize variables for FPS calculation
prev_time = 0

# Get the video frame width, height, and frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
output_path = 'output_video4.mp4'  # Output file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 video
out = cv2.VideoWriter(output_path, fourcc, 15, (frame_width, frame_height))

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

# Create a thread pool for processing frames
with ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start timer for FPS calculation
        current_time = cv2.getTickCount()

        # Submit the frame for YOLO detection and OCR processing in a separate thread
        future = executor.submit(process_frame, frame)
        
        # Process the result from the thread
        detected_objects = future.result()

        for (x1, y1, x2, y2, ocr_result) in detected_objects:
            try:
                detected_text = ''.join([text[0] for bounding_box, text in ocr_result[0]])
                # Draw the bounding box around the license plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add the detected text on the frame
                cv2.putText(frame, str(detected_text), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except:
                # Draw the bounding box around the license plate even if OCR fails
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate FPS
        time_elapsed = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        fps = 1 / time_elapsed if time_elapsed > 0 else 0

        # Display FPS on the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame with bounding boxes, recognized text, and FPS
        cv2.imshow('YOLOv8 License Plate Detection with OCR', frame)

        # Press 'q' to quit the video display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
