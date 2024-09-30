import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from database import SessionLocal, LicensePlate
import datetime
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the YOLOv8 model
model = YOLO('/Users/nguyendang/licensce-plate/plate_yolov8n_320_2024.pt')  # Replace with your trained YOLO model

# Initialize PaddleOCR for English language
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # lang='en' là tốt nhất cho biển số xe có chữ số Latin

# Open the video file
video_path = '/Users/nguyendang/licensce-plate/test_video/test_4.mp4'
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
output_path = '/Users/nguyendang/licensce-plate/results/output_video.mp4'  # Output file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 video
out = cv2.VideoWriter(output_path, fourcc, 15, (frame_width, frame_height))

# Định nghĩa để lưu biển số đã ghi gần đây cùng với thời gian lưu
last_detected = {}

def clean_plate_number(plate_text):
    plate_text = plate_text.replace('.', '').replace('-', '')
    
    # Kiểm tra định dạng biển số hợp lệ: 2 chữ số + 1 chữ cái + 5 chữ số (VD: 30G49729, 34K23664)
    if re.match(r'^\d{2}[A-Z]\d{5}$', plate_text):
        return plate_text
    return None

def process_frame(frame, db_session):
    # Run YOLOv8 inference with custom IoU and confidence threshold for NMS
    results = model(frame, conf=conf_threshold, iou=iou_threshold)
    detected_objects = []
    
    # Đảm bảo thư mục tồn tại trước khi lưu ảnh
    detected_images_dir = 'detected_plates/'
    if not os.path.exists(detected_images_dir):
        os.makedirs(detected_images_dir)

    # Loop over the detected objects
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = box.conf[0]
            label = int(box.cls[0])
            # Only proceed if confidence is above the threshold
            if confidence < conf_threshold:
                continue

            # Crop the detected object (license plate area)
            cropped_image = frame[y1:y2, x1:x2]
            
            # Use PaddleOCR to detect text in the cropped image
            ocr_result = ocr.ocr(cropped_image)
            
            # Check if ocr_result is not None and has at least one valid entry
            if ocr_result is not None and len(ocr_result) > 0 and isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                detected_text = ''.join([line[1][0] for line in ocr_result[0]])
                
                # Chuẩn hóa biển số: bỏ dấu . và -, và kiểm tra định dạng
                cleaned_plate = clean_plate_number(detected_text)
                
                if cleaned_plate:
                    current_time = datetime.datetime.now()

                    # Kiểm tra xem biển số này đã được ghi trong vòng 30 giây trước đó chưa
                    if cleaned_plate in last_detected:
                        time_diff = (current_time - last_detected[cleaned_plate]).total_seconds()
                        if time_diff < 30:
                            print(f"Biển số {cleaned_plate} đã được ghi trong vòng 30 giây, bỏ qua.")
                            # Tuy nhiên vẫn cần vẽ bounding box cho biển số đó
                            detected_objects.append((x1, y1, x2, y2, cleaned_plate, cropped_image))
                            continue
                    
                    # Nếu vượt qua các kiểm tra, lưu biển số vào cơ sở dữ liệu
                    detected_objects.append((x1, y1, x2, y2, cleaned_plate, cropped_image))
                    
                    # Cập nhật thời gian lưu biển số
                    last_detected[cleaned_plate] = current_time
                    
                    # Lưu biển số xe và thời gian vào cơ sở dữ liệu
                    plate = LicensePlate(
                        plate_number=cleaned_plate,
                        timestamp=current_time
                    )
                    db_session.add(plate)
                    db_session.commit()

                    # Lưu hình ảnh đã detect
                    image_save_path = f"{detected_images_dir}{cleaned_plate}_{current_time.strftime('%Y%m%d%H%M%S')}.jpg"
                    cv2.imwrite(image_save_path, cropped_image)
                    
                    print(f"Biển số xe nhận diện: {cleaned_plate}")
                else:
                    print(f"Biển số '{detected_text}' không hợp lệ sau khi chuẩn hóa.")
            else:
                print("Không nhận diện được văn bản từ OCR.")

    return detected_objects

# Chỉnh sửa đoạn hiển thị bounding box cho dù đã lưu biển số trước đó
with ThreadPoolExecutor(max_workers=2) as executor:
    # Tạo session database
    db_session = SessionLocal()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start timer for FPS calculation
        current_time = cv2.getTickCount()

        # Submit the frame for YOLO detection and OCR processing in a separate thread
        future = executor.submit(process_frame, frame, db_session)
        
        # Process the result from the thread
        detected_objects = future.result()

        for (x1, y1, x2, y2, cleaned_plate, cropped_image) in detected_objects:
            # Always draw the bounding box around the license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add the detected text on the frame
            cv2.putText(frame, cleaned_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

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

    # Đóng session database khi hoàn tất
    db_session.close()

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

