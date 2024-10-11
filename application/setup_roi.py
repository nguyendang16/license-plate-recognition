import cv2
import tkinter as tk
from tkinter import messagebox
import yaml
import threading

# Hàm load file config.yaml
def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load config file: {e}")
        return None

# Hàm lưu config vào file
def save_config(config, file_path):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(config, file)
        # Thông báo cập nhật thành công
        messagebox.showinfo("Success", "Cập nhật tọa độ thành công!!!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save config file: {e}")

# Hàm vẽ ROI bằng OpenCV và lưu kết quả
def select_roi(stream_path, roi_key, config, file_path):
    cap = cv2.VideoCapture(stream_path)
    
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open stream: {stream_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", f"Cannot read stream: {stream_path}")
        return
    
    # Hiển thị ROI hiện tại nếu có
    if roi_key in config and config[roi_key]:
        x, y, w, h = config[roi_key]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Cho phép người dùng chọn ROI mới
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    
    # Nếu người dùng nhấn Enter mà không chọn ROI, giá trị roi sẽ là (0, 0, 0, 0)
    if roi != (0, 0, 0, 0):
        # Lưu ROI vào config
        config[roi_key] = list(roi)
        
        # Sau khi cập nhật config, lưu lại file
        save_config(config, file_path)

# Hàm chạy OpenCV trong luồng riêng
def handle_stream_in_thread(stream_num, config, file_path):
    path = f"stream{stream_num}"
    stream_path = config.get(path)
    roi_key = f"ROI{stream_num}"
    
    if stream_path is not None:
        threading.Thread(target=select_roi, args=(stream_path, roi_key, config, file_path)).start()
    else:
        messagebox.showerror("Error", f"No stream found for Stream {stream_num}")

# Giao diện chọn từng stream và ROI
def create_gui(config, file_path):
    window = tk.Tk()
    window.title("ROI Selector")

    # Hàm xử lý khi click nút cho từng stream
    def handle_stream(stream_num):
        handle_stream_in_thread(stream_num, config, file_path)
    
    # Xử lý sự kiện khi nhấn nút X
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            window.quit()

    # Đăng ký hàm sự kiện cho nút "X"
    window.protocol("WM_DELETE_WINDOW", on_closing)

    # Tạo giao diện các luồng
    for stream_num in range(1, 5):
        frame = tk.Frame(window)
        frame.pack(pady=10)
        
        label = tk.Label(frame, text=f"Stream {stream_num}")
        label.pack(side=tk.LEFT, padx=10)
        
        roi_button = tk.Button(frame, text="Choose ROI", command=lambda num=stream_num: handle_stream(num))
        roi_button.pack(side=tk.LEFT, padx=5)

    # Nút Close
    close_button = tk.Button(window, text="Close", command=window.quit)
    close_button.pack(pady=10)
    
    window.mainloop()

# Đường dẫn đến file config.yaml
config_file_path = "dashboard/config.yaml"

# Load config từ file
config = load_config(config_file_path)

# Tạo giao diện nếu load thành công
if config:
    create_gui(config, config_file_path)
