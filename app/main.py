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
