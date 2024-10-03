from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.requests import Request

app = FastAPI()

# Sử dụng thư mục static để lưu trữ các tệp tĩnh như CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Sử dụng thư mục templates để lưu trữ các tệp HTML
templates = Jinja2Templates(directory="templates")

# Định tuyến trang chủ - hiển thị form nhập dữ liệu
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Định tuyến xử lý form - nhận dữ liệu và xử lý
@app.post("/submit", response_class=HTMLResponse)
async def submit_form(request: Request, name: str = Form(...), age: int = Form(...)):
    message = ""
    if age < 18:
        message = f"Sorry {name}, you are not allowed because you're under 18."
    else:
        message = f"Welcome {name}, you are {age} years old!"
    return templates.TemplateResponse("result.html", {"request": request, "message": message})

# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
