# License plate recognition

## Description
This project proposes a method to identify vehicle license plates using model YOLOv8 to detect license plates and PaddleOCR to extract information on license plates. Then we use HTML+CSS as frontend and FastAPI and some part of JavaScript as backend for website.

## Setup

First, clone the project:

```git clone https://github.com/nguyendang16/license-plate-recognition.git ```

```cd license-plate-recognition/```

You can create virtual env if you want (Recommend), then install required packages:

```pip install -r requirements.txt```

We mostly use the folder ``/dashboard`` to develop the website, so next step is ```cd application/dashboard/```

To run the website:

```uvicorn main:app --reload```

### Note: Since we haven't build a database for this project so we decided to use a generated data as an example. You can edit and run ```python create_excel.py``` to generate data to use in website.

The flow of the project: Upload excel file in Admin Panel to receive information and license plate number.
Then, head to Guest Panel for real time recognition and auto-fill information (You need to change the camera_id for your own purpose).
After that, you can check the history in History Panel. 


