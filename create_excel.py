import pandas as pd

# Create sample data in English
data = {
    "request_type": ["once", "repeat", "once", "repeat", "once"],
    "start_date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-18"],
    "work_area": ["Area A", "Area B", "Area C", "Area D", "Area E"],
    "department": ["Business Dept", "IT Dept", "Marketing Dept", "HR Dept", "Finance Dept"],
    "visiting_unit": ["XYZ Corp", "ABC Company", "LMN Inc", "OPQ Ltd", "RST Group"],
    "purpose": ["Office visit", "System maintenance", "Periodic meeting", "Employee training", "Financial audit"],
    "reference_document": ["DOC12345", "DOC67890", "DOC54321", "DOC98765", "DOC11223"],
    "id_number": ["0123456789", "0987654321", "0234567890", "0345678901", "0456789012"],
    "guest_name": ["John Doe", "Jane Smith", "Emily Clark", "Peter Johnson", "Anna Lee"],
    "phone_number": ["0912345678", "0987654321", "0901234567", "0912345678", "0923456789"],
    "company_address": [
        "123 ABC Street, Hanoi",
        "456 DEF Road, HCMC",
        "789 GHI Avenue, Da Nang",
        "321 JKL Blvd, Hai Phong",
        "654 MNO Way, Can Tho"
    ],
    "representative_guest": ["Michael Brown", "David Wilson", "Sophia Davis", "Daniel Martinez", "Laura White"],
    "car_name": ["Toyota Camry", "Honda Civic", "Ford Focus", "Kia Rio", "BMW 3 Series"],
    "license_plate": ["99H77060", "88K12345", "77M54321", "66N67890", "55P11223"],
    "driver_name": ["Adam Johnson", "Paul Walker", "Luke Evans", "Mark Turner", "James Roberts"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to an Excel file
df.to_excel("admin_data_sample.xlsx", index=False)

print("File 'admin_data_sample.xlsx' has been successfully created!")
