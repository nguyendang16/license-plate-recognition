import pandas as pd

# Tạo dữ liệu mẫu
data = {
    "request_type": ["once", "repeat", "once", "repeat", "once"],
    "ngay_bat_dau": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-18"],
    "khu_vuc_lam_viec": ["Khu A", "Khu B", "Khu C", "Khu D", "Khu E"],
    "phong_ban": ["Phòng Kinh Doanh", "Phòng IT", "Phòng Marketing", "Phòng Nhân Sự", "Phòng Tài Chính"],
    "don_vi_den": ["Công ty XYZ", "Công ty ABC", "Công ty LMN", "Công ty OPQ", "Công ty RST"],
    "muc_dich": ["Thăm quan văn phòng", "Bảo trì hệ thống", "Họp mặt định kỳ", "Đào tạo nhân viên", "Kiểm toán tài chính"],
    "van_ban_tham_chieu": ["VB12345", "VB67890", "VB54321", "VB98765", "VB11223"],
    "so_giay_to": ["0123456789", "0987654321", "0234567890", "0345678901", "0456789012"],
    "ten_khach": ["Nguyễn Văn A", "Trần Thị D", "Lê Thị G", "Phạm Thị J", "Vũ Thị M"],
    "so_dien_thoai": ["0912345678", "0987654321", "0901234567", "0912345678", "0923456789"],
    "dia_chi_cong_ty": [
        "123 Đường ABC, Hà Nội",
        "456 Đường DEF, TP.HCM",
        "789 Đường GHI, Đà Nẵng",
        "321 Đường JKL, Hải Phòng",
        "654 Đường MNO, Cần Thơ"
    ],
    "khach_dai_dien": ["Trần Thị B", "Nguyễn Văn E", "Lê Văn H", "Nguyễn Văn K", "Trần Văn N"],
    "ten_xe": ["Toyota Camry", "Honda Civic", "Ford Focus", "Kia Rio", "BMW 3 Series"],
    "bien_so": ["99H77060", "88K12345", "77M54321", "66N67890", "55P11223"],
    "ten_lai_xe": ["Lê Văn C", "Phạm Văn F", "Trương Văn I", "Đỗ Văn L", "Hoàng Văn O"]
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Lưu vào file Excel
df.to_excel("admin_data_sample.xlsx", index=False)

print("File 'admin_data_sample.xlsx' đã được tạo thành công!")
