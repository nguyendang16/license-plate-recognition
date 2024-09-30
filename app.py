# app.py
from flask import Flask, render_template, send_file
from database import SessionLocal, LicensePlate
from sqlalchemy import desc
import os

app = Flask(__name__)

@app.route('/')
def index():
    db_session = SessionLocal()
    
    plates = db_session.query(LicensePlate).order_by(desc(LicensePlate.timestamp)).all()
    
    db_session.close()
    
    return render_template('index.html', plates=plates)

@app.route('/plate/<plate_number>')
def show_plate_image(plate_number):
    # Tìm đường dẫn hình ảnh dựa trên biển số
    image_path = None
    detected_images_dir = "detected_plates/"
    for file_name in os.listdir(detected_images_dir):
        if plate_number in file_name:
            image_path = os.path.join(detected_images_dir, file_name)
            break
    
    if image_path and os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return "Không tìm thấy hình ảnh cho biển số này", 404

if __name__ == '__main__':
    app.run(debug=True)
