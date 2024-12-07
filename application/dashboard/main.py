from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import cv2
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import pandas as pd
import io
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure OpenCV does not encounter duplicate library errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the YOLO model
model = YOLO('models/plate_new.pt')  

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# In-memory data store (replace with a database in production)
data_lock = threading.Lock()
data = {
    "request_type": "",
    "start_date": "",
    "work_area": "",
    "department": "",
    "visiting_unit": "",
    "purpose": "",
    "reference_document": "",
    "id_number": "",
    "guest_name": "",
    "phone_number": "",
    "company_address": "",
    "representative_guest": "",
    "car_name": "",
    "license_plate": "",
    "license_plate_valid": False,
    "driver_name": "",
    "image1_url": "/video_feed1",
    "image2_url": "/video_feed2",
    "image3_url": "/video_feed3",
    "image4_url": "/video_feed4",
    "time_1": "",
    "time_2": "",
    "time_3": "",
    "time_4": "",
}

# Admin data store
admin_data_lock = threading.Lock()
admin_data = []  # List of customers registered from Excel

history_data_lock = threading.Lock()
history_data = []

camera_last_detection_time = {
    1: None,  # Camera 1: Entry
    2: None,  # Camera 2: Entry
    3: None,  # Camera 3: Exit
    4: None   # Camera 4: Exit
}

# Cooldown time (in seconds) to avoid multiple records for the same event
DETECTION_COOLDOWN = 5

# Normalize license plate
def normalize_license_plate(text):
    """
    Remove '-', '.' and spaces from the plate.
    Example: '99-H7 7060' -> '99H77060'
    """
    # Use regex to remove non-alphanumeric characters
    normalized = re.sub(r'[^A-Za-z0-9]', '', text)
    return normalized.upper()

# Validate license plate
def is_valid_license_plate(plate):
    return len(plate) >= 6

# Update or add a new customer entry in admin_data
def update_admin_data(customer):
    """
    Update customer info in admin_data if the license plate already exists.
    If not, add a new customer entry.
    """
    with admin_data_lock:
        for idx, existing_customer in enumerate(admin_data):
            if existing_customer['license_plate'].upper() == customer['license_plate'].upper():
                admin_data[idx] = customer
                logger.info(f"Updated customer info with license plate: {customer['license_plate']}")
                return
        # If not found, add new
        admin_data.append(customer)
        logger.info(f"Added new customer with license plate: {customer['license_plate']}")

# Process frames to detect license plates and OCR
def process_frame(frame, camera_id):
    results = model(frame, conf=0.5, iou=0.3)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = box.cls[0]

            cropped_image = frame[y1:y2, x1:x2]

            # Use PaddleOCR to recognize text in the cropped image
            ocr_result = ocr.ocr(cropped_image, rec=True, cls=True)
            detected_objects.append((x1, y1, x2, y2, ocr_result))

    # If a license plate is detected, update data and history_data
    for (x1, y1, x2, y2, ocr_result) in detected_objects:
        try:
            # Check if OCR recognized any text
            if ocr_result and len(ocr_result) > 0:
                # Extract text from OCR result
                detected_texts = [line[1][0] for line in ocr_result[0]]
                raw_text = ' '.join(detected_texts).strip()
                normalized_text = normalize_license_plate(raw_text)
                logger.info(f"Raw OCR Text: '{raw_text}' | Normalized: '{normalized_text}'")

                if is_valid_license_plate(normalized_text):
                    current_time = time.time()  # Current timestamp in seconds

                    with data_lock:
                        # Check cooldown
                        last_detection = camera_last_detection_time.get(camera_id)
                        if last_detection and (current_time - last_detection) < DETECTION_COOLDOWN:
                            logger.info(f"Cooldown active for camera {camera_id}. Skipping detection.")
                            continue  # Skip if still in cooldown

                        # Update license plate and its validity
                        data["license_plate"] = normalized_text
                        data["license_plate_valid"] = True

                        # Update detection time in data according to camera
                        time_field = f"time_{camera_id}"
                        formatted_time = time.strftime("%H:%M:%S %d-%m-%Y", time.localtime())
                        data[time_field] = formatted_time
                        logger.info(f"Updated {time_field} to: {formatted_time}")

                        # Determine entry/exit status based on camera_id
                        if camera_id in [1, 2]:
                            in_out_status = "Entry"
                        elif camera_id in [3, 4]:
                            in_out_status = "Exit"
                        else:
                            in_out_status = "Undefined"

                        # Create history record
                        history_record = {
                            "guest_name": data.get("guest_name", ""),
                            "license_plate": normalized_text,
                            "car_name": data.get("car_name", ""),
                            "work_area": data.get("work_area", ""),
                            "visiting_unit": data.get("visiting_unit", ""),
                            "purpose": data.get("purpose", ""),
                            "date": time.strftime("%d-%m-%Y", time.localtime()),
                            "time": time.strftime("%H:%M:%S", time.localtime()),
                            "in_out_status": in_out_status
                        }

                        with history_data_lock:
                            history_data.append(history_record)
                            logger.info(f"Added history record: {history_record}")

                        # Update the last detection time for this camera
                        camera_last_detection_time[camera_id] = current_time

                        # Check if the license plate exists in admin_data
                        with admin_data_lock:
                            matched_customer = next((customer for customer in admin_data if customer['license_plate'].upper() == normalized_text), None)

                        if matched_customer:
                            # Auto-fill info from admin_data
                            data.update({
                                "request_type": matched_customer.get("request_type", ""),
                                "start_date": matched_customer.get("start_date", ""),
                                "work_area": matched_customer.get("work_area", ""),
                                "department": matched_customer.get("department", ""),
                                "visiting_unit": matched_customer.get("visiting_unit", ""),
                                "purpose": matched_customer.get("purpose", ""),
                                "reference_document": matched_customer.get("reference_document", ""),
                                "id_number": matched_customer.get("id_number", ""),
                                "guest_name": matched_customer.get("guest_name", ""),
                                "phone_number": matched_customer.get("phone_number", ""),
                                "company_address": matched_customer.get("company_address", ""),
                                "representative_guest": matched_customer.get("representative_guest", ""),
                                "car_name": matched_customer.get("car_name", ""),
                                "driver_name": matched_customer.get("driver_name", ""),
                            })
                            logger.info(f"Matched customer: {matched_customer['guest_name']}")
                        else:
                            # Unregistered license plate, clear other fields
                            data.update({
                                "request_type": "",
                                "start_date": "",
                                "work_area": "",
                                "department": "",
                                "visiting_unit": "",
                                "purpose": "",
                                "reference_document": "",
                                "id_number": "",
                                "guest_name": "",
                                "phone_number": "",
                                "company_address": "",
                                "representative_guest": "",
                                "car_name": "",
                                "driver_name": "",
                            })
                            data["license_plate_valid"] = False
                            logger.warning(f"Detected license plate '{normalized_text}' is not registered.")
                else:
                    with data_lock:
                        data["license_plate_valid"] = False
                        # When license plate is invalid and not in admin_data, clear other fields
                        data.update({
                            "request_type": "",
                            "start_date": "",
                            "work_area": "",
                            "department": "",
                            "visiting_unit": "",
                            "purpose": "",
                            "reference_document": "",
                            "id_number": "",
                            "guest_name": "",
                            "phone_number": "",
                            "company_address": "",
                            "representative_guest": "",
                            "car_name": "",
                            "driver_name": "",
                        })
                        logger.warning(f"Detected license plate '{normalized_text}' is invalid (less than 6 characters).")
        except Exception as e:
            logger.error(f"Error processing OCR result: {e}")

    return detected_objects

# Stream frames generator function
def gen_frames(video_source, camera_id):
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            success, frame = cap.read()
            if not success:
                logger.warning(f"Failed to read frame from {video_source}. Retrying in 1 second.")
                time.sleep(1)
                continue

            # Submit frame for YOLO + OCR processing
            future = executor.submit(process_frame, frame, camera_id)
            detected_objects = future.result()

            # Draw bounding boxes and recognized text on the frame
            for (x1, y1, x2, y2, ocr_result) in detected_objects:
                try:
                    detected_texts = [line[1][0] for line in ocr_result[0]]
                    raw_text = ' '.join(detected_texts).strip()
                    normalized_text = normalize_license_plate(raw_text)
                    if normalized_text:
                        if is_valid_license_plate(normalized_text):
                            color = (0, 255, 0) 
                            text_color = (0, 0, 255)
                            display_text = normalized_text
                        else:
                            color = (0, 0, 255)  
                            text_color = (255, 255, 255)
                            display_text = "Invalid License Plate"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                except Exception as e:
                    logger.error(f"Error drawing OCR text on frame: {e}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes for video streams
@app.get('/video_feed1')
async def video_feed1():
    # Camera 1: Entry
    return StreamingResponse(gen_frames('rtsp://admin:namtiep2005@192.168.1.25:554/Streaming/channels/101', camera_id=1), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video_feed2')
async def video_feed2():
    # Camera 2: Entry
    return StreamingResponse(gen_frames('rtsp://pathtech:pathtech1@192.168.1.35:554/stream1', camera_id=2), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video_feed3')
async def video_feed3():
    # Camera 3: Exit
    return StreamingResponse(gen_frames(0, camera_id=3), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video_feed4')
async def video_feed4():
    # Camera 4: Exit
    return StreamingResponse(gen_frames('rtsp://rtsp_stream_url_4', camera_id=4), media_type='multipart/x-mixed-replace; boundary=frame')

# Routes for admin.html and home
@app.get("/", response_class=HTMLResponse)
@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request, "message": None})

# Route for guest.html
@app.get("/guest", response_class=HTMLResponse)
async def read_guest(request: Request):
    with data_lock:
        current_data = data.copy()
    return templates.TemplateResponse("guest.html", {
        "request": request,
        "is_editable": False,
        "button_text": "EDIT",
        "action": "edit",
        "data": current_data,
        "message": None
    })

# Route for history.html
@app.get("/history", response_class=HTMLResponse)
async def read_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})

# API to get current guest data
@app.get("/api/guest", response_class=JSONResponse)
async def get_guest_data():
    with data_lock:
        return data.copy()

# API to get admin data
@app.get("/api/admin", response_class=JSONResponse)
async def get_admin_data():
    with admin_data_lock:
        return admin_data.copy()
    
@app.get("/api/history", response_class=JSONResponse)
async def get_history():
    with history_data_lock:
        return history_data.copy()

# Route to upload Excel file for admin
@app.post("/admin/upload", response_class=HTMLResponse)
async def upload_admin_data(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "message": "Only Excel files (.xlsx, .xls) are supported."
        })

    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        # Check required columns
        required_columns = [
            "request_type", "start_date", "work_area",
            "department", "visiting_unit", "purpose",
            "reference_document", "id_number", "guest_name",
            "phone_number", "company_address", "representative_guest",
            "car_name", "license_plate", "driver_name"
        ]

        for col in required_columns:
            if col not in df.columns:
                return templates.TemplateResponse("admin.html", {
                    "request": request,
                    "message": f"Missing column: {col}"
                })

        # Convert DataFrame to list of dict
        admin_entries = df.to_dict(orient='records')

        # Normalize license plates
        for entry in admin_entries:
            original_plate = entry['license_plate']
            normalized_plate = normalize_license_plate(str(original_plate))
            entry['license_plate'] = normalized_plate

        with admin_data_lock:
            admin_data.clear()
            admin_data.extend(admin_entries)

        logger.info(f"Uploaded {len(admin_entries)} admin entries from Excel.")

        return templates.TemplateResponse("admin.html", {
            "request": request,
            "message": f"Successfully uploaded {len(admin_entries)} customers."
        })

    except Exception as e:
        logger.error(f"Error uploading admin data: {e}")
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "message": f"Error uploading file: {e}"
        })

# Route to download admin data as Excel
@app.get("/admin/download")
async def download_admin_data():
    with admin_data_lock:
        if not admin_data:
            return JSONResponse(content={"message": "No data to download."}, status_code=400)
        df = pd.DataFrame(admin_data)
        # Prepare the Excel file in memory
        stream = io.BytesIO()
        df.to_excel(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=admin_data.xlsx"}
        )

# Handle POST request for guest.html
@app.post("/guest", response_class=JSONResponse)
async def submit_guest(
    request: Request,
    action: str = Form(...),
    request_type: str = Form(None),
    start_date: str = Form(None),
    work_area: str = Form(None),
    department: str = Form(None),
    visiting_unit: str = Form(None),
    purpose: str = Form(None),
    reference_document: str = Form(None),
    id_number: str = Form(None),
    guest_name: str = Form(None),
    phone_number: str = Form(None),
    company_address: str = Form(None),
    representative_guest: str = Form(None),
    car_name: str = Form(None),
    license_plate: str = Form(None),
    driver_name: str = Form(None)
):
    if action == "edit":
        # Switch to edit mode
        with data_lock:
            current_data = data.copy()
        return JSONResponse({
            "status": "edit",
            "data": current_data,
            "message": None
        })
    elif action == "confirm":
        # Update data with new values
        form_data = await request.form()
        with data_lock:
            data.update({
                "request_type": form_data.get("request_type", data["request_type"]),
                "start_date": form_data.get("start_date", data["start_date"]),
                "work_area": form_data.get("work_area", data["work_area"]),
                "department": form_data.get("department", data["department"]),
                "visiting_unit": form_data.get("visiting_unit", data["visiting_unit"]),
                "purpose": form_data.get("purpose", data["purpose"]),
                "reference_document": form_data.get("reference_document", data["reference_document"]),
                "id_number": form_data.get("id_number", data["id_number"]),
                "guest_name": form_data.get("guest_name", data["guest_name"]),
                "phone_number": form_data.get("phone_number", data["phone_number"]),
                "company_address": form_data.get("company_address", data["company_address"]),
                "representative_guest": form_data.get("representative_guest", data["representative_guest"]),
                "car_name": form_data.get("car_name", data["car_name"]),
                "license_plate": form_data.get("license_plate", data["license_plate"]),
                "driver_name": form_data.get("driver_name", data["driver_name"]),
            })
            # Validate license plate after update
            normalized_plate = normalize_license_plate(data["license_plate"])
            if is_valid_license_plate(normalized_plate):
                data["license_plate"] = normalized_plate
                data["license_plate_valid"] = True
                logger.info(f"Confirmed license_plate: {data['license_plate']}")

                # Check if the license plate is in admin_data
                with admin_data_lock:
                    matched_customer = next((customer for customer in admin_data if customer['license_plate'].upper() == normalized_plate), None)

                if matched_customer:
                    # Update admin_data with form info
                    updated_customer = {
                        "request_type": data.get("request_type", ""),
                        "start_date": data.get("start_date", ""),
                        "work_area": data.get("work_area", ""),
                        "department": data.get("department", ""),
                        "visiting_unit": data.get("visiting_unit", ""),
                        "purpose": data.get("purpose", ""),
                        "reference_document": data.get("reference_document", ""),
                        "id_number": data.get("id_number", ""),
                        "guest_name": data.get("guest_name", ""),
                        "phone_number": data.get("phone_number", ""),
                        "company_address": data.get("company_address", ""),
                        "representative_guest": data.get("representative_guest", ""),
                        "car_name": data.get("car_name", ""),
                        "license_plate": data.get("license_plate", ""),
                        "driver_name": data.get("driver_name", ""),
                    }
                    update_admin_data(updated_customer)
                else:
                    # Unregistered license plate, add new to admin_data
                    new_customer = {
                        "request_type": data.get("request_type", ""),
                        "start_date": data.get("start_date", ""),
                        "work_area": data.get("work_area", ""),
                        "department": data.get("department", ""),
                        "visiting_unit": data.get("visiting_unit", ""),
                        "purpose": data.get("purpose", ""),
                        "reference_document": data.get("reference_document", ""),
                        "id_number": data.get("id_number", ""),
                        "guest_name": data.get("guest_name", ""),
                        "phone_number": data.get("phone_number", ""),
                        "company_address": data.get("company_address", ""),
                        "representative_guest": data.get("representative_guest", ""),
                        "car_name": data.get("car_name", ""),
                        "license_plate": data.get("license_plate", ""),
                        "driver_name": data.get("driver_name", ""),
                    }
                    update_admin_data(new_customer)
            else:
                data["license_plate_valid"] = False
                # If license plate is invalid, clear other fields
                data.update({
                    "request_type": "",
                    "start_date": "",
                    "work_area": "",
                    "department": "",
                    "visiting_unit": "",
                    "purpose": "",
                    "reference_document": "",
                    "id_number": "",
                    "guest_name": "",
                    "phone_number": "",
                    "company_address": "",
                    "representative_guest": "",
                    "car_name": "",
                    "driver_name": "",
                })
                logger.warning(f"Confirmed license plate '{normalized_plate}' is invalid (less than 6 characters).")

        with data_lock:
            current_data = data.copy()
        if current_data["license_plate_valid"]:
            message = "Information saved successfully!"
        else:
            message = "Invalid information! Please check again."

        return JSONResponse({
            "status": "success" if current_data["license_plate_valid"] else "error",
            "message": message,
            "data": current_data
        })
