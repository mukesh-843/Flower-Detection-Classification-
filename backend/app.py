import os
from io import BytesIO
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "models/my_model_exp2.h5")
CLASS_NAMES: List[str] = [
    "daisy",
    "sunflower",
    "tulip",
    "dandelion",
    "rose",
]
INPUT_SHAPE = (64, 64)

app = FastAPI(title="Flower Classifier", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def _build_model():
    # Recreate the training architecture so we can load weights-only checkpoints
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="Same", activation="relu", input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))
    return model

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        try:
            _model = load_model(MODEL_PATH)
        except ValueError:
            # If the file only contains weights, rebuild the model and load weights
            _model = _build_model()
            _model.load_weights(MODEL_PATH)
    return _model

def _preprocess_image(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    image = image.resize(INPUT_SHAPE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    bytes_data = file.file.read()
    model = _load_model()
    batch = _preprocess_image(bytes_data)

    predictions = model.predict(batch)
    probs = predictions[0].tolist()
    top_index = int(np.argmax(predictions[0]))
    return {
        "prediction": CLASS_NAMES[top_index],
        "confidence": float(probs[top_index]),
        "probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)},
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
