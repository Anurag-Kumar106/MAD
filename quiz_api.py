from fastapi import FastAPI, File, UploadFile
import shutil
import os
from Gemini_Quiz_Generator import Gemini_Quiz_Generator

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

quiz_generator = Gemini_Quiz_Generator(api_key="AIzaSyBM1JuQ5dlkOgglFPi7vQqMh0TkWQuN45s")

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed!"}
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return quiz_generator.generate_quiz(file_path)
    
    
