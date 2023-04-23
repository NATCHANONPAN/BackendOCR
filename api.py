# Running script:
# uvicorn api:app --host 0.0.0.0 --port 5000

import base64
import cv2
import numpy as np
import time
import os
from fastapi import FastAPI, File
from starlette.responses import FileResponse
from config import settings
from recognition.recog_model import model_recognition
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from OCR_thaiDocSeparator.Detector import detector

from typing import List

class Item(BaseModel):
    image:str

app = FastAPI()

# Automatically create uploads and results folders
if not os.path.exists(settings.upload_dest):
    os.makedirs(settings.upload_dest)
if not os.path.exists(settings.result_dest):
    os.makedirs(settings.result_dest)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def decodeByte2Numpy(inputImage):
    outputImage = np.frombuffer(inputImage, np.uint8)
    outputImage = cv2.imdecode(outputImage, cv2.IMREAD_COLOR)
    return outputImage


def recognition(imgs: List[np.ndarray]) -> List[str]:
    return model_recognition(imgs)


def detection(IMG_PATH, OUT_DIR_PATH):
    return detector(IMG_PATH, OUT_DIR_PATH)


@app.post('/ocr_test')
def ocr_test(image: bytes = File(...)):
    try:
        inputImage = decodeByte2Numpy(image)
        timestamp = time.time()
        input_ = f"upload{timestamp}.jpg"
        output_ = f"result{timestamp}.jpg"
        cv2.imwrite(settings.upload_dest + input_, inputImage)

        # Detection
        box_imgs = detection(IMG_PATH=settings.upload_dest + input_,
                             OUT_DIR_PATH=settings.result_dest + output_
                             )

        # Recognition
        result = recognition(box_imgs)

        outputDict = {
            'status': 'VALID',
            'paragraph': " ".join(result),
            'list': result,
            'box_img': FileResponse(settings.result_dest + output_)
        }
        return outputDict
    except Exception as e:
        print(e)
        return {
            'error': e,
            'status': 'INVALID'
        }

@app.post('/ocr')
def ocr(image: Item):
    dataByte = base64.b64decode(image.image)
    try:
        inputImage = decodeByte2Numpy(dataByte)
        timestamp = time.time()
        input_ = f"upload{timestamp}.jpg"
        output_ = f"result{timestamp}.jpg"
        cv2.imwrite(settings.upload_dest + input_, inputImage)

        # Detection
        box_imgs = detection(IMG_PATH=settings.upload_dest + input_,
                             OUT_DIR_PATH=settings.result_dest + output_
                             )

        # Recognition
        result = recognition(box_imgs)

        outputDict = {
            'status': 'VALID',
            'paragraph': " ".join(result),
            'list': result,
            'box_img': FileResponse(settings.result_dest + output_)
        }
        return outputDict
    except Exception as e:
        print(e)
        return {
            'error': e,
            'status': 'INVALID'
        }