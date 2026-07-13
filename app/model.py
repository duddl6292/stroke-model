import io
import os

import numpy as np
import tensorflow as tf
from google.cloud import storage
from PIL import Image


BUCKET_NAME = "stroke-models"
BLOB_PATH = "model/test_model.h5"
LOCAL_MODEL_PATH = "/tmp/model.h5"


def download_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_PATH)

    if not blob.exists(client):
        raise FileNotFoundError(
            f"Model not found: gs://{BUCKET_NAME}/{BLOB_PATH}"
        )

    blob.download_to_filename(LOCAL_MODEL_PATH)

    return LOCAL_MODEL_PATH


download_model()

_model = tf.keras.models.load_model(
    LOCAL_MODEL_PATH,
    compile=False,
)


def predict_mask(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((128, 128))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.reshape(1, 128, 128, 1)

    prediction = _model.predict(array, verbose=0)

    mask = (
        prediction[0, :, :, 0] > 0.5
    ).astype(np.uint8)

    return mask.tolist()