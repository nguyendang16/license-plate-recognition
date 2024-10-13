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

# Cấu hình logging để theo dõi ứng dụng
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đảm bảo rằng OpenCV không gặp lỗi liên quan đến thư viện
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the YOLO model
# Hãy chắc chắn rằng đường dẫn đến mô hình YOLO là chính xác
model = YOLO('models/plate_new.pt')  # Thay thế bằng đường dẫn đúng đến mô hình YOLO của bạn

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# In-memory data store (thay thế bằng cơ sở dữ liệu trong môi trường production)
data_lock = threading.Lock()
data = {
    "request_type": "",
    "ngay_bat_dau": "",
    "khu_vuc_lam_viec": "",
    "phong_ban": "",
    "don_vi_den": "",
    "muc_dich": "",
    "van_ban_tham_chieu": "",
    "so_giay_to": "",
    "ten_khach": "",
    "so_dien_thoai": "",
    "dia_chi_cong_ty": "",
    "khach_dai_dien": "",
    "ten_xe": "",
    "bien_so": "",
    "bien_so_valid": False,  # Thêm trường để xác định tính hợp lệ của biển số
    "ten_lai_xe": "",
    "image1_url": "/video_feed1",
    "image2_url": "/video_feed2",
    "image3_url": "/video_feed3",
    "image4_url": "/video_feed4",
    "thoi_gian_1": "",
    "thoi_gian_2": "",
    "thoi_gian_3": "",
    "thoi_gian_4": "",
}

# Admin data store
admin_data_lock = threading.Lock()
admin_data = []  # Danh sách các khách hàng đã đăng ký từ Excel

history_data_lock = threading.Lock()
history_data = []

# Hàm normal hóa biển số
def normalize_license_plate(text):
    """
    Loại bỏ các dấu '-', '.', và khoảng cách từ biển số.
    Ví dụ: '99-H7 7060' -> '99H77060'
    """
    # Sử dụng regex để loại bỏ các ký tự không phải chữ cái hoặc số
    normalized = re.sub(r'[^A-Za-z0-9]', '', text)
    return normalized.upper()

# Hàm xác thực biển số
def is_valid_license_plate(plate):
    return len(plate) >= 6

# Hàm cập nhật hoặc thêm mới dữ liệu vào admin_data
def update_admin_data(customer):
    """
    Cập nhật thông tin khách hàng trong admin_data nếu biển số đã tồn tại.
    Nếu không, thêm mới khách hàng vào admin_data.
    """
    with admin_data_lock:
        for idx, existing_customer in enumerate(admin_data):
            if existing_customer['bien_so'].upper() == customer['bien_so'].upper():
                admin_data[idx] = customer
                logger.info(f"Cập nhật thông tin khách hàng với biển số: {customer['bien_so']}")
                return
        # Nếu không tìm thấy, thêm mới
        admin_data.append(customer)
        logger.info(f"Thêm mới khách hàng với biển số: {customer['bien_so']}")

# Hàm xử lý khung hình để phát hiện biển số và OCR
# Thay đổi định nghĩa hàm process_frame
def process_frame(frame, camera_id):
    results = model(frame, conf=0.5, iou=0.3)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = box.cls[0]

            # Crop khu vực biển số
            cropped_image = frame[y1:y2, x1:x2]

            # Sử dụng PaddleOCR để nhận diện văn bản trong hình ảnh cắt
            ocr_result = ocr.ocr(cropped_image, rec=True, cls=True)
            detected_objects.append((x1, y1, x2, y2, ocr_result))

    # Nếu phát hiện biển số, cập nhật vào data và history_data
    for (x1, y1, x2, y2, ocr_result) in detected_objects:
        try:
            # Kiểm tra xem OCR có nhận diện được văn bản không
            if ocr_result and len(ocr_result) > 0:
                # Lấy văn bản từ kết quả OCR
                detected_texts = [line[1][0] for line in ocr_result[0]]
                raw_text = ' '.join(detected_texts).strip()
                normalized_text = normalize_license_plate(raw_text)
                logger.info(f"Raw OCR Text: '{raw_text}' | Normalized: '{normalized_text}'")

                if is_valid_license_plate(normalized_text):
                    with data_lock:
                        data["bien_so"] = normalized_text
                        data["bien_so_valid"] = True
                        # Cập nhật thời gian nhận diện biển số
                        current_time = time.strftime("%H:%M:%S %d-%m-%Y", time.localtime())
                        if data["thoi_gian_1"] == "":
                            data["thoi_gian_1"] = current_time
                        elif data["thoi_gian_2"] == "":
                            data["thoi_gian_2"] = current_time
                        elif data["thoi_gian_3"] == "":
                            data["thoi_gian_3"] = current_time
                        elif data["thoi_gian_4"] == "":
                            data["thoi_gian_4"] = current_time
                        logger.info(f"Updated bien_so to: {data['bien_so']}")

                        # Xác định trạng thái ra vào dựa trên camera_id
                        if camera_id in [1, 2]:
                            trang_thai_ra_vao = "Vào"
                        elif camera_id in [3, 4]:
                            trang_thai_ra_vao = "Ra"
                        else:
                            trang_thai_ra_vao = "Không xác định"

                        # Tạo bản ghi lịch sử
                        history_record = {
                            "ten_khach": data.get("ten_khach", ""),
                            "bien_so": normalized_text,
                            "ten_xe": data.get("ten_xe", ""),
                            "khu_vuc_lam_viec": data.get("khu_vuc_lam_viec", ""),
                            "don_vi_den": data.get("don_vi_den", ""),
                            "muc_dich": data.get("muc_dich", ""),
                            "ngay": time.strftime("%d-%m-%Y", time.localtime()),
                            "thoi_gian": time.strftime("%H:%M:%S", time.localtime()),
                            "trang_thai_ra_vao": trang_thai_ra_vao
                        }

                        with history_data_lock:
                            history_data.append(history_record)
                            logger.info(f"Added history record: {history_record}")

                        # Kiểm tra xem biển số có trong admin_data không
                        with admin_data_lock:
                            matched_customer = next((customer for customer in admin_data if customer['bien_so'].upper() == normalized_text), None)

                        if matched_customer:
                            # Auto-fill thông tin từ admin_data
                            data.update({
                                "request_type": matched_customer.get("request_type", ""),
                                "ngay_bat_dau": matched_customer.get("ngay_bat_dau", ""),
                                "khu_vuc_lam_viec": matched_customer.get("khu_vuc_lam_viec", ""),
                                "phong_ban": matched_customer.get("phong_ban", ""),
                                "don_vi_den": matched_customer.get("don_vi_den", ""),
                                "muc_dich": matched_customer.get("muc_dich", ""),
                                "van_ban_tham_chieu": matched_customer.get("van_ban_tham_chieu", ""),
                                "so_giay_to": matched_customer.get("so_giay_to", ""),
                                "ten_khach": matched_customer.get("ten_khach", ""),
                                "so_dien_thoai": matched_customer.get("so_dien_thoai", ""),
                                "dia_chi_cong_ty": matched_customer.get("dia_chi_cong_ty", ""),
                                "khach_dai_dien": matched_customer.get("khach_dai_dien", ""),
                                "ten_xe": matched_customer.get("ten_xe", ""),
                                "ten_lai_xe": matched_customer.get("ten_lai_xe", ""),
                            })
                            logger.info(f"Matched customer: {matched_customer['ten_khach']}")
                        else:
                            # Biển số chưa đăng ký, chỉ cập nhật bien_so và đặt các trường khác về trống
                            data.update({
                                "request_type": "",
                                "ngay_bat_dau": "",
                                "khu_vuc_lam_viec": "",
                                "phong_ban": "",
                                "don_vi_den": "",
                                "muc_dich": "",
                                "van_ban_tham_chieu": "",
                                "so_giay_to": "",
                                "ten_khach": "",
                                "so_dien_thoai": "",
                                "dia_chi_cong_ty": "",
                                "khach_dai_dien": "",
                                "ten_xe": "",
                                "ten_lai_xe": "",
                            })
                            data["bien_so_valid"] = False
                            logger.warning(f"Detected license plate '{normalized_text}' is not registered.")
                else:
                    with data_lock:
                        data["bien_so_valid"] = False
                        # Khi biển số không hợp lệ và không tồn tại trong admin_data, chỉ cập nhật bien_so và đặt các trường khác về trống
                        data.update({
                            "request_type": "",
                            "ngay_bat_dau": "",
                            "khu_vuc_lam_viec": "",
                            "phong_ban": "",
                            "don_vi_den": "",
                            "muc_dich": "",
                            "van_ban_tham_chieu": "",
                            "so_giay_to": "",
                            "ten_khach": "",
                            "so_dien_thoai": "",
                            "dia_chi_cong_ty": "",
                            "khach_dai_dien": "",
                            "ten_xe": "",
                            "ten_lai_xe": "",
                        })
                        logger.warning(f"Detected license plate '{normalized_text}' is invalid (less than 6 characters).")
        except Exception as e:
            logger.error(f"Error processing OCR result: {e}")

    return detected_objects

# Thay đổi định nghĩa hàm gen_frames
def gen_frames(video_source, camera_id):
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Đặt chiều rộng
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Đặt chiều cao
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            success, frame = cap.read()
            if not success:
                logger.warning(f"Failed to read frame from {video_source}. Retrying in 1 second.")
                time.sleep(1)
                continue

            # Submit frame để xử lý YOLO + OCR
            future = executor.submit(process_frame, frame, camera_id)
            detected_objects = future.result()

            # Vẽ khung và văn bản nhận diện trên frame
            for (x1, y1, x2, y2, ocr_result) in detected_objects:
                try:
                    detected_texts = [line[1][0] for line in ocr_result[0]]
                    raw_text = ' '.join(detected_texts).strip()
                    normalized_text = normalize_license_plate(raw_text)
                    if normalized_text:
                        if is_valid_license_plate(normalized_text):
                            color = (0, 255, 0)  # Xanh lá cho biển số hợp lệ
                            text_color = (0, 0, 255)
                            display_text = normalized_text
                        else:
                            color = (0, 0, 255)  # Đỏ cho biển số không hợp lệ
                            text_color = (255, 255, 255)
                            display_text = "Biển số không hợp lệ"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                except Exception as e:
                    logger.error(f"Error drawing OCR text on frame: {e}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Các route cho các luồng video
@app.get('/video_feed1')
async def video_feed1():
    # Camera 1: Vào
    return StreamingResponse(gen_frames('rtsp://admin:password@192.168.1.25:554/Streaming/channels/101', camera_id=1), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video_feed2')
async def video_feed2():
    # Camera 2: Vào
    return StreamingResponse(gen_frames('rtsp://pathtech:pathtech1@192.168.1.35:554/stream1', camera_id=2), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video_feed3')
async def video_feed3():
    # Camera 3: Ra
    return StreamingResponse(gen_frames(0, camera_id=3), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video_feed4')
async def video_feed4():
    # Camera 4: Ra
    return StreamingResponse(gen_frames('rtsp://rtsp_stream_url_4', camera_id=4), media_type='multipart/x-mixed-replace; boundary=frame')

# Route cho admin.html và trang chủ
@app.get("/", response_class=HTMLResponse)
@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request, "message": None})

# Route cho guest.html
@app.get("/guest", response_class=HTMLResponse)
async def read_guest(request: Request):
    with data_lock:
        current_data = data.copy()
    return templates.TemplateResponse("guest.html", {
        "request": request,
        "is_editable": False,  # Ban đầu không ở chế độ chỉnh sửa
        "button_text": "CHỈNH SỬA",
        "action": "edit",
        "data": current_data,
        "message": None
    })

# Route cho history.html
@app.get("/history", response_class=HTMLResponse)
async def read_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})

# API để lấy dữ liệu guest hiện tại
@app.get("/api/guest", response_class=JSONResponse)
async def get_guest_data():
    with data_lock:
        return data.copy()

# API để lấy dữ liệu admin
@app.get("/api/admin", response_class=JSONResponse)
async def get_admin_data():
    with admin_data_lock:
        return admin_data.copy()
    
@app.get("/api/history", response_class=JSONResponse)
async def get_history():
    with history_data_lock:
        return history_data.copy()

# Route để tải lên file Excel cho admin
@app.post("/admin/upload", response_class=HTMLResponse)
async def upload_admin_data(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "message": "Chỉ hỗ trợ file Excel (.xlsx, .xls)."
        })

    try:
        contents = await file.read()
        # Đọc file Excel sử dụng pandas
        df = pd.read_excel(io.BytesIO(contents))

        # Kiểm tra các cột cần thiết
        required_columns = [
            "request_type", "ngay_bat_dau", "khu_vuc_lam_viec",
            "phong_ban", "don_vi_den", "muc_dich",
            "van_ban_tham_chieu", "so_giay_to", "ten_khach",
            "so_dien_thoai", "dia_chi_cong_ty", "khach_dai_dien",
            "ten_xe", "bien_so", "ten_lai_xe"
        ]

        for col in required_columns:
            if col not in df.columns:
                return templates.TemplateResponse("admin.html", {
                    "request": request,
                    "message": f"Thiếu cột: {col}"
                })

        # Chuyển đổi DataFrame thành danh sách các dict
        admin_entries = df.to_dict(orient='records')

        # Chuẩn hóa biển số (bỏ dấu - . và khoảng cách, uppercase)
        for entry in admin_entries:
            original_plate = entry['bien_so']
            normalized_plate = normalize_license_plate(str(original_plate))
            entry['bien_so'] = normalized_plate

        with admin_data_lock:
            admin_data.clear()
            admin_data.extend(admin_entries)

        logger.info(f"Uploaded {len(admin_entries)} admin entries from Excel.")

        return templates.TemplateResponse("admin.html", {
            "request": request,
            "message": f"Tải lên thành công {len(admin_entries)} khách hàng."
        })

    except Exception as e:
        logger.error(f"Error uploading admin data: {e}")
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "message": f"Lỗi khi tải lên file: {e}"
        })

# Route để tải xuống admin data dưới dạng Excel
@app.get("/admin/download")
async def download_admin_data():
    with admin_data_lock:
        if not admin_data:
            return JSONResponse(content={"message": "Không có dữ liệu để tải xuống."}, status_code=400)
        df = pd.DataFrame(admin_data)
        # Chuẩn bị file Excel trong bộ nhớ
        stream = io.BytesIO()
        df.to_excel(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=admin_data.xlsx"}
        )

# Xử lý POST request cho guest.html
@app.post("/guest", response_class=JSONResponse)
async def submit_guest(
    request: Request,
    action: str = Form(...),
    request_type: str = Form(None),
    ngay_bat_dau: str = Form(None),
    khu_vuc_lam_viec: str = Form(None),
    phong_ban: str = Form(None),
    don_vi_den: str = Form(None),
    muc_dich: str = Form(None),
    van_ban_tham_chieu: str = Form(None),
    so_giay_to: str = Form(None),
    ten_khach: str = Form(None),
    so_dien_thoai: str = Form(None),
    dia_chi_cong_ty: str = Form(None),
    khach_dai_dien: str = Form(None),
    ten_xe: str = Form(None),
    bien_so: str = Form(None),
    ten_lai_xe: str = Form(None)
):
    if action == "edit":
        # Chuyển sang chế độ chỉnh sửa
        with data_lock:
            current_data = data.copy()
        return JSONResponse({
            "status": "edit",
            "data": current_data,
            "message": None
        })
    elif action == "confirm":
        # Cập nhật dữ liệu với các giá trị mới
        form_data = await request.form()
        with data_lock:
            data.update({
                "request_type": form_data.get("request_type", data["request_type"]),
                "ngay_bat_dau": form_data.get("ngay_bat_dau", data["ngay_bat_dau"]),
                "khu_vuc_lam_viec": form_data.get("khu_vuc_lam_viec", data["khu_vuc_lam_viec"]),
                "phong_ban": form_data.get("phong_ban", data["phong_ban"]),
                "don_vi_den": form_data.get("don_vi_den", data["don_vi_den"]),
                "muc_dich": form_data.get("muc_dich", data["muc_dich"]),
                "van_ban_tham_chieu": form_data.get("van_ban_tham_chieu", data["van_ban_tham_chieu"]),
                "so_giay_to": form_data.get("so_giay_to", data["so_giay_to"]),
                "ten_khach": form_data.get("ten_khach", data["ten_khach"]),
                "so_dien_thoai": form_data.get("so_dien_thoai", data["so_dien_thoai"]),
                "dia_chi_cong_ty": form_data.get("dia_chi_cong_ty", data["dia_chi_cong_ty"]),
                "khach_dai_dien": form_data.get("khach_dai_dien", data["khach_dai_dien"]),
                "ten_xe": form_data.get("ten_xe", data["ten_xe"]),
                "bien_so": form_data.get("bien_so", data["bien_so"]),
                "ten_lai_xe": form_data.get("ten_lai_xe", data["ten_lai_xe"]),
            })
            # Xác thực biển số sau khi cập nhật
            normalized_bien_so = normalize_license_plate(data["bien_so"])
            if is_valid_license_plate(normalized_bien_so):
                data["bien_so"] = normalized_bien_so
                data["bien_so_valid"] = True
                logger.info(f"Confirmed bien_so: {data['bien_so']}")
                
                # Kiểm tra xem biển số có trong admin_data không
                with admin_data_lock:
                    matched_customer = next((customer for customer in admin_data if customer['bien_so'].upper() == normalized_bien_so), None)

                if matched_customer:
                    # Cập nhật thông tin từ form vào admin_data
                    updated_customer = {
                        "request_type": data.get("request_type", ""),
                        "ngay_bat_dau": data.get("ngay_bat_dau", ""),
                        "khu_vuc_lam_viec": data.get("khu_vuc_lam_viec", ""),
                        "phong_ban": data.get("phong_ban", ""),
                        "don_vi_den": data.get("don_vi_den", ""),
                        "muc_dich": data.get("muc_dich", ""),
                        "van_ban_tham_chieu": data.get("van_ban_tham_chieu", ""),
                        "so_giay_to": data.get("so_giay_to", ""),
                        "ten_khach": data.get("ten_khach", ""),
                        "so_dien_thoai": data.get("so_dien_thoai", ""),
                        "dia_chi_cong_ty": data.get("dia_chi_cong_ty", ""),
                        "khach_dai_dien": data.get("khach_dai_dien", ""),
                        "ten_xe": data.get("ten_xe", ""),
                        "bien_so": data.get("bien_so", ""),
                        "ten_lai_xe": data.get("ten_lai_xe", ""),
                    }
                    update_admin_data(updated_customer)
                else:
                    # Biển số chưa đăng ký, thêm mới vào admin_data
                    new_customer = {
                        "request_type": data.get("request_type", ""),
                        "ngay_bat_dau": data.get("ngay_bat_dau", ""),
                        "khu_vuc_lam_viec": data.get("khu_vuc_lam_viec", ""),
                        "phong_ban": data.get("phong_ban", ""),
                        "don_vi_den": data.get("don_vi_den", ""),
                        "muc_dich": data.get("muc_dich", ""),
                        "van_ban_tham_chieu": data.get("van_ban_tham_chieu", ""),
                        "so_giay_to": data.get("so_giay_to", ""),
                        "ten_khach": data.get("ten_khach", ""),
                        "so_dien_thoai": data.get("so_dien_thoai", ""),
                        "dia_chi_cong_ty": data.get("dia_chi_cong_ty", ""),
                        "khach_dai_dien": data.get("khach_dai_dien", ""),
                        "ten_xe": data.get("ten_xe", ""),
                        "bien_so": data.get("bien_so", ""),
                        "ten_lai_xe": data.get("ten_lai_xe", ""),
                    }
                    update_admin_data(new_customer)
            else:
                data["bien_so_valid"] = False
                # Khi biển số không hợp lệ và không tồn tại trong admin_data, chỉ cập nhật bien_so và đặt các trường khác về trống
                data.update({
                    "request_type": "",
                    "ngay_bat_dau": "",
                    "khu_vuc_lam_viec": "",
                    "phong_ban": "",
                    "don_vi_den": "",
                    "muc_dich": "",
                    "van_ban_tham_chieu": "",
                    "so_giay_to": "",
                    "ten_khach": "",
                    "so_dien_thoai": "",
                    "dia_chi_cong_ty": "",
                    "khach_dai_dien": "",
                    "ten_xe": "",
                    "ten_lai_xe": "",
                })
                logger.warning(f"Confirmed bien_so '{normalized_bien_so}' is invalid (less than 6 characters).")
        
        # Trả về thông báo thành công hoặc cảnh báo
        with data_lock:
            current_data = data.copy()
        if current_data["bien_so_valid"]:
            message = "Đã lưu thông tin thành công!"
        else:
            message = "Thông tin không hợp lệ! Vui lòng kiểm tra lại."
        
        return JSONResponse({
            "status": "success" if current_data["bien_so_valid"] else "error",
            "message": message,
            "data": current_data
        })