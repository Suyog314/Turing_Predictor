

# 01webapp/main.py

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os

from predict.predict_params import predict_parameters

app = FastAPI()

# Setup templates and static folder
app.mount("/static", StaticFiles(directory="01webapp/static"), name="static")
templates = Jinja2Templates(directory="01webapp/templates")

UPLOAD_FOLDER = "01webapp/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "params": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    Du, Dv, F, k = predict_parameters(file_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "params": {
            "Du": round(Du, 5),
            "Dv": round(Dv, 5),
            "F": round(F, 5),
            "k": round(k, 5)
        },
        "image_path": f"/static/uploads/{file.filename}"
    })
