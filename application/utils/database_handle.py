def save_info_to_database(db: Session, license_plate: str, guest_name: str, purpose: str):
    guest = GuestInfo(
        license_plate = license_plate,
        guest_name = guest_name,
        purpose = purpose,
        check_in_time = datetime.now()
    )

def get_user_info(db, lisence_plate: str = None):
    if lisence_plate:
        return db.query(GuestInfo).filter(GuestInfo.license_plate == lisence_plate).first()
    return db.query(GuestInfo).all()

def save_check_in_data(db: Session, license_plate: str):
    guest = db.query(GuestInfo).filter(GuestInfo.license_plate == license_plate).first()
    if guest:
        guest.check_in_time = datetime.now()  # Cập nhật thời gian check-in
        db.commit()
        db.refresh(guest)
        return guest
    return None

def edit_information(db: Session, license_plate: str, guest_name: str = None, purpose: str = None):
    guest = db.query(GuestInfo).filter(GuestInfo.license_plate == license_plate).first()
    if guest:
        if guest_name:
            guest.guest_name = guest_name
        if purpose:
            guest.purpose = purpose
        db.commit()
        db.refresh(guest)
        return guest
    return None  


