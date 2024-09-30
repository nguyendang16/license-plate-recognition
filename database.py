# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Khởi tạo cơ sở dữ liệu SQLite
DATABASE_URL = 'sqlite:///license_plates.db'

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class LicensePlate(Base):
    __tablename__ = 'license_plates'
    
    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Tạo các bảng trong cơ sở dữ liệu
Base.metadata.create_all(bind=engine)
