from fastapi import FastAPI, UploadFile, File
import tempfile
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/best_model.h5")

_model = None


def get_model():
    global _model

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    return _model


def preprocess_image(img_path: str, model):
    input_shape = model.input_shape[1:3]

    img = image.load_img(img_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    return x / 255.0


def predict_image(image_path: str):
    model = get_model()
    x = preprocess_image(image_path, model)

    output = model.predict(x, verbose=0)

    if output.shape != (1, 1):
        raise ValueError(
            f"expected binary sigmoid output with shape (1, 1), received {output.shape}"
        )

    prob_sick = float(output[0][0])
    label = "Sick" if prob_sick > 0.5 else "Healthy"
    confidence = prob_sick if label == "Sick" else 1.0 - prob_sick

    return {
        "prediction": label,
        "confidence_score": float(confidence),
        "raw_model_output": prob_sick,
        "model_input_shape": list(model.input_shape),
    }


def make_no_xai_placeholder(method_name: str, prediction: str, confidence_score: float):
    return {
        "status": "not_run_no_xai_condition",
        "method": method_name,
        "message": "XAI was intentionally not run for this experimental condition.",
        "visualizations": [],
        "coordinates": [],
        "prediction": prediction,
        "confidence_score": confidence_score,
    }


@app.get("/health")
def health():
    model = get_model()

    return {
        "status": "cnn-only service running",
        "model_path": MODEL_PATH,
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape),
        "xai_enabled": False,
        "available_endpoint": "/analyze",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "input.jpg").suffix

    if not suffix:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    try:
        prediction_result = predict_image(image_path)

        prediction = prediction_result["prediction"]
        confidence_score = prediction_result["confidence_score"]

        return {
            "status": "success",
            "mode": "cnn_only_no_xai",
            "prediction": prediction,
            "confidence_score": confidence_score,
            "classifier": prediction_result,
            "xai_results": {
                "gradcam": make_no_xai_placeholder(
                    "Grad-CAM",
                    prediction,
                    confidence_score,
                ),
                "lime": make_no_xai_placeholder(
                    "LIME",
                    prediction,
                    confidence_score,
                ),
                "shap": make_no_xai_placeholder(
                    "SHAP",
                    prediction,
                    confidence_score,
                ),
                "deeplift": make_no_xai_placeholder(
                    "DeepLIFT",
                    prediction,
                    confidence_score,
                ),
            },
        }

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
