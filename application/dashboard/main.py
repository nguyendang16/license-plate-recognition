from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import cv2
import time

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/application/dashboard/static", StaticFiles(directory="static"), name="static")

# In-memory data store (replace with a database in production)
data = {
    "request_type": "once",
    "ngay_bat_dau": "2024-10-03",
    "khu_vuc_lam_viec": "Phòng 101",
    "phong_ban": "Kỹ thuật",
    "don_vi_den": "Công ty ABC",
    "muc_dich": "Họp mặt",
    "van_ban_tham_chieu": "",
    "so_giay_to": "123456789",
    "ten_khach": "Nguyễn Văn A",
    "so_dien_thoai": "0901234567",
    "dia_chi_cong_ty": "123 Đường XYZ",
    "khach_dai_dien": "Trần Thị B",
    "ten_xe": "Toyota",
    "bien_so": "30A-12345",
    "ten_lai_xe": "Lê Văn C",
    "image1_url": "/video_feed1",
    "image2_url": "/video_feed2",
    "image3_url": "/video_feed3",
    "image4_url": "/video_feed4",
    "thoi_gian_1": "11:00:00 03-10-2024",
    "thoi_gian_2": "12:00:00 03-10-2024",
    "thoi_gian_3": "13:00:00 03-10-2024",
    "thoi_gian_4": "14:00:00 03-10-2024",
}

# Hàm generator chung cho các luồng RTSP
def gen_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(1)
            continue
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route cho luồng video thứ nhất
@app.get('/video_feed1')
async def video_feed1():
    return StreamingResponse(gen_frames('rtsp://admin:namtiep2005@192.168.1.25:554/Streaming/channels/101'), media_type='multipart/x-mixed-replace; boundary=frame')

# Route cho luồng video thứ hai
@app.get('/video_feed2')
async def video_feed2():
    return StreamingResponse(gen_frames('rtsp://pathtech:pathtech1@192.168.1.35:554/stream1'), media_type='multipart/x-mixed-replace; boundary=frame')

# Route cho luồng video thứ ba
@app.get('/video_feed3')
async def video_feed3():
    return StreamingResponse(gen_frames('rtsp://rtsp_stream_url_3'), media_type='multipart/x-mixed-replace; boundary=frame')

# Route cho luồng video thứ tư
@app.get('/video_feed4')
async def video_feed4():
    return StreamingResponse(gen_frames('rtsp://rtsp_stream_url_4'), media_type='multipart/x-mixed-replace; boundary=frame')

# Route cho dashboard.html
@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Route cho guest.html
@app.get("/guest", response_class=HTMLResponse)
async def read_guest(request: Request):
    return templates.TemplateResponse("guest.html", {
        "request": request,
        "is_editable": False,
        "button_text": "CHỈNH SỬA",
        "action": "edit",
        "data": data,
        "message": None
    })

# Route cho admin.html
@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {
        "request": request,
    })

# Xử lý POST request cho guest.html
@app.post("/guest", response_class=HTMLResponse)
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
        return templates.TemplateResponse("guest.html", {
            "request": request,
            "is_editable": True,
            "button_text": "XÁC NHẬN",
            "action": "confirm",
            "data": data,
            "message": None
        })
    elif action == "confirm":
        # Cập nhật dữ liệu với các giá trị mới
        form_data = await request.form()
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
        # Return read-only
        return templates.TemplateResponse("guest.html", {
            "request": request,
            "is_editable": False,
            "button_text": "CHỈNH SỬA",
            "action": "edit",
            "data": data,
            "message": "Đã lưu thông tin thành công!"
        })
