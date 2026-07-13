import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from google.cloud import storage

BUCKET_NAME = "stroke-models"       # 예: cdss-models-호영
BLOB_PATH = "model/test_model.h5"       # 예: unet/v1/model.h5
LOCAL_MODEL_PATH = "/tmp/model.h5"

def download_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        return
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_PATH)
    blob.download_to_filename(LOCAL_MODEL_PATH)

download_model()
_model = tf.keras.models.load_model(LOCAL_MODEL_PATH)

def predict_mask(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((128, 128))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 128, 128, 1)

    pred = _model.predict(arr, verbose=0)
    mask = (pred[0, :, :, 0] > 0.5).astype(int).tolist()
    return mask