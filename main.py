import base64
import os
import shutil
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from pathlib import Path

app = FastAPI()

# Directory for saving images
image_directory = "static/images"
os.makedirs(image_directory, exist_ok=True)


matrix_directory = "static/matrices"
os.makedirs(matrix_directory, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename.endswith('.jpg'):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    file_id = str(uuid.uuid4())
    img_path = os.path.join(image_directory, f"{file_id}.jpg")
    
    contents = await file.read()
    image = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Save original image for static access
    cv2.imwrite(img_path, image)

    return JSONResponse({"file_id": file_id})

@app.post("/process-image/")
async def process_image(file_id: str = Form(...), 
                        magnification1: int = Form(...),
                        magnification2: int = Form(...),
                        area_thresh: int = Form(3000),
                        border_up: int = Form(150),
                        border_down: int = Form(150),
                        border_left: int = Form(150),
                        border_right: int = Form(150)):
    img_path = os.path.join(image_directory, f"{file_id}.jpg")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    m = magnification2/magnification1
    block_size = round(320/m)


    processed_image = apply_thresholds_and_contours(image, area_thresh)
    
    # Apply the border modifications
    processed_image[:border_up, :] = 0
    processed_image[-border_down:, :] = 0
    processed_image[:, :border_left] = 0
    processed_image[:, -border_right:] = 0
    
    matrix_path = generate_and_save_black_blocks_matrix(processed_image, file_id, block_size)
    _, buffer = cv2.imencode('.png', processed_image)
    processed_encoded = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({"processed": processed_encoded, "matrix_file": matrix_path})

def apply_thresholds_and_contours(image, area_thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_thresh]
    mask_tri = np.zeros_like(gray)
    cv2.drawContours(mask_tri, contours, -1, (255), thickness=cv2.FILLED)
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_thresh]
    mask_otsu = np.zeros_like(gray)
    cv2.drawContours(mask_otsu, contours, -1, (255), thickness=cv2.FILLED)
    mask = cv2.bitwise_or(mask_tri, mask_otsu)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def generate_and_save_black_blocks_matrix(image, file_id, block_size=80):
    height, width, _ = image.shape
    rows = height // block_size
    cols = width // block_size
    matrix = np.ones((rows, cols), dtype=int)  # Initialize matrix with 1s (assume non-black)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if np.all(block == 0):  # Check if the block is completely black
                matrix[i // block_size, j // block_size] = 0

    # Save the matrix to a text file
    matrix_path = os.path.join("static/matrices", f"{file_id}_matrix.txt")
    np.savetxt(matrix_path, matrix, fmt="%d")
    return matrix_path


@app.get("/")
async def main():
    return FileResponse("static/index.html")

