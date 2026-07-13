from fastapi import FastAPI, UploadFile
from app.model import predict_mask

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    mask = predict_mask(image_bytes)
    return {"mask": mask}

from google.cloud import storage
import os

MODEL_PATH = "/tmp/model.h5"

def download_model():
    if os.path.exists(MODEL_PATH):
        return

    client = storage.Client()
    bucket = client.bucket("stroke-models")
    blob = bucket.blob("model/test_model.h5")
    blob.download_to_filename(MODEL_PATH)

download_model()
# 이후 MODEL_PATH로 모델 로드