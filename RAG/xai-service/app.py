from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

import tempfile
import subprocess
import json
import base64
import os
from pathlib import Path


app = FastAPI()

app.mount("/files", StaticFiles(directory="/app"), name="files")


#SHAP is handled separately so blur and inpaint run in separate subprocesses
SCRIPTS = {
    "gradcam": "bestgradn8n.py",
    "lime": "Limen8n.py",
    "deeplift": "deepliftn8n.py",
}


def file_to_base64(path: str):
    if not path or not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def attach_images(result: dict):
    if not isinstance(result, dict):
        return result

    if "heatmap_image_url" in result:
        result["heatmap_image_base64"] = file_to_base64(result["heatmap_image_url"])

    if "explanation_image_url" in result:
        result["explanation_image_base64"] = file_to_base64(result["explanation_image_url"])

    if "visualizations" in result:
        for item in result["visualizations"]:
            if not isinstance(item, dict):
                continue

            if "image_url" in item:
                item["image_base64"] = file_to_base64(item["image_url"])

    return result


def error_result(script_name: str, message: str, stdout="", stderr=""):
    return {
        "status": "error",
        "script": script_name,
        "message": message,
        "stdout": stdout or "",
        "stderr": stderr or "",
    }


def run_script(script_name: str, image_path: str, extra_args=None, timeout_seconds=600):
    if extra_args is None:
        extra_args = []

    command = ["python", script_name, image_path, *extra_args]

    try:
        completed = subprocess.run(
            command,
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    except subprocess.TimeoutExpired as e:
        return error_result(
            script_name,
            "subprocess_timeout",
            stdout=e.stdout,
            stderr=e.stderr,
        )

    if completed.returncode != 0:
        return error_result(
            script_name,
            "subprocess_failed",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    try:
        lines = completed.stdout.strip().splitlines()

        if not lines:
            return error_result(
                script_name,
                "empty_stdout",
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        result = json.loads(lines[-1])
        return attach_images(result)

    except Exception as e:
        return error_result(
            script_name,
            str(e),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def run_shap_split(image_path: str):
    visualizations = []
    coordinates = {}
    partial_errors = {}

    prediction = None
    confidence_score = None

    for masker_name in ["blur", "inpaint"]:
        result = run_script(
            "shapn8n.py",
            image_path,
            extra_args=[masker_name],
            timeout_seconds=600,
        )

        if result.get("status") == "success":
            if prediction is None:
                prediction = result.get("prediction")

            if confidence_score is None:
                confidence_score = result.get("confidence_score")

            for item in result.get("visualizations", []):
                if not isinstance(item, dict):
                    continue

                if item.get("status") == "error":
                    continue

                visualizations.append(item)

                masker = item.get("masker") or masker_name
                coordinates[masker] = item.get("coordinates", [])

        else:
            partial_errors[masker_name] = result

    if visualizations:
        shap_result = {
            "status": "success",
            "method": "SHAP",
            "prediction": prediction,
            "confidence_score": confidence_score,
            "visualizations": visualizations,
            "coordinates": coordinates,
            "partial_errors": partial_errors,
        }

        return shap_result

    return {
        "status": "error",
        "method": "SHAP",
        "message": "Both SHAP maskers failed.",
        "visualizations": [],
        "coordinates": coordinates,
        "partial_errors": partial_errors,
    }


@app.get("/health")
def health():
    return {
        "status": "xai service running",
        "available_methods": [
            "gradcam",
            "lime",
            "shap",
            "deeplift",
        ],
        "shap_mode": "split_subprocess_per_masker",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "input.jpg").suffix

    if not suffix:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    results = {}

    try:
        for method, script in SCRIPTS.items():
            results[method] = run_script(script, image_path)

        results["shap"] = run_shap_split(image_path)

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    return {
        "status": "success",
        "xai_results": results,
    }
