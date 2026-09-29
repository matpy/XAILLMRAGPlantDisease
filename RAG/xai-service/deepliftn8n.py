from pathlib import Path
import tensorflow as tf
import numpy as np
import shap
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import json
import os
import sys

tf.compat.v1.disable_eager_execution()



BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.h5"
BASELINE_PATH = BASE_DIR / "healthy_mean_baseline.npy"

OUTPUT_PATH = BASE_DIR / "deeplift_final_masked_ui.jpg"
BASELINE_DEBUG_PATH = BASE_DIR / "debug_baseline_blur.jpg"
COMPARISON_PATH = BASE_DIR / "deeplift_final_masked_comparison.jpg"



_BASELINE_CACHE = None


def normalize_map(values):
    values = values.astype(np.float32)
    min_val = np.min(values)
    max_val = np.max(values)

    if max_val - min_val <= 1e-10:
        return np.zeros_like(values, dtype=np.float32)

    return (values - min_val) / (max_val - min_val)

def make_linear_output_model(model):

    last_layer_name = model.layers[-1].name

    def clone_function(layer):
        config = layer.get_config()

        if layer.name == last_layer_name and "activation" in config:
            config["activation"] = "linear"

        return layer.__class__.from_config(config)

    linear_model = tf.keras.models.clone_model(
        model,
        clone_function=clone_function,
    )

    linear_model.set_weights(model.get_weights())

    return linear_model


def load_healthy_baseline():

    global _BASELINE_CACHE

    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Missing baseline file: {BASELINE_PATH}")

    baseline = np.load(str(BASELINE_PATH)).astype("float32")

    # Ensure shape is 224x224x3.
    if baseline.shape[:2] != (224, 224):
        baseline = cv2.resize(baseline, (224, 224))

    # If stored as 0-255, convert to 0-1.
    if np.max(baseline) > 1.5:
        baseline = baseline / 255.0

    _BASELINE_CACHE = baseline.astype("float32")
    return _BASELINE_CACHE


def get_leaf_mask_rgb(img_rgb_uint8):

    hsv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV)

    lower_green = np.array([10, 20, 20])
    upper_green = np.array([100, 255, 255])

    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return leaf_mask


def get_leaf_mask_bgr(img_bgr):

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return get_leaf_mask_rgb(img_rgb)


def compute_signed_deeplift_attribution(model, rgb_float01, target_idx=0, apply_leaf_mask=True):

    rgb_float01 = rgb_float01.astype("float32")

    if rgb_float01.shape[:2] != (224, 224):
        rgb_float01 = cv2.resize(rgb_float01, (224, 224)).astype("float32")

    baseline = load_healthy_baseline()

    x = np.expand_dims(rgb_float01, axis=0)
    baseline_batch = np.expand_dims(baseline, axis=0)

    explainer = shap.DeepExplainer(model, baseline_batch)
    shap_values = explainer.shap_values(x)

    signed_attr = (
        np.squeeze(shap_values[target_idx])
        if isinstance(shap_values, list)
        else np.squeeze(shap_values)
    )

    if apply_leaf_mask:
        img_uint8 = np.uint8(np.clip(rgb_float01 * 255.0, 0, 255))
        leaf_mask = get_leaf_mask_rgb(img_uint8)
        mask_3d = np.repeat(leaf_mask[:, :, np.newaxis], 3, axis=2) / 255.0
        signed_attr = signed_attr * mask_3d

    return signed_attr.astype("float32")


def attribution_to_2d_map(signed_attr_masked, mode="positive"):

    if mode == "positive":
        attr_map = np.maximum(signed_attr_masked, 0).sum(axis=-1)

    elif mode == "negative":
        attr_map = np.maximum(-signed_attr_masked, 0).sum(axis=-1)

    elif mode == "absolute":
        attr_map = np.abs(signed_attr_masked).sum(axis=-1)

    else:
        raise ValueError(f"Unknown attribution mode: {mode}")

    return normalize_map(attr_map)


def get_dl_attribution(
    external_model,
    external_image_array,
    target_idx=0,
    mode="absolute",
    apply_leaf_mask=False,
):
    signed_attr = compute_signed_deeplift_attribution(
        model=external_model,
        rgb_float01=external_image_array,
        target_idx=target_idx,
        apply_leaf_mask=apply_leaf_mask,
    )

    return attribution_to_2d_map(signed_attr, mode=mode)


def get_top_coordinates_from_attribution(signed_attr_masked, max_points=4, threshold=60):

    importance_map = np.abs(signed_attr_masked).sum(axis=-1)
    importance_map_rescaled = cv2.resize(importance_map, (224, 224))

    heatmap_norm = normalize_map(importance_map_rescaled)
    heatmap_bw = np.uint8(255 * heatmap_norm)

    _, thresh = cv2.threshold(heatmap_bw, threshold, 255, cv2.THRESH_BINARY)

    kernel_dilate = np.ones((11, 11), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_points]

    coords = []

    for cnt in contours:
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)

            coords.append({
                "x": cX,
                "y": cY,
                "w": int(w),
                "h": int(h),
            })

    return coords



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No image path provided."}))
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print(json.dumps({"status": "error", "message": f"File not found: {img_path}"}))
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(json.dumps({"status": "error", "message": f"Model not found: {MODEL_PATH}"}))
        sys.exit(1)

    model = tf.keras.models.load_model(str(MODEL_PATH))

    img = image.load_img(img_path, target_size=(224, 224))
    x_raw = image.img_to_array(img)
    x = x_raw.astype("float32") / 255.0
    x_batch = np.expand_dims(x, axis=0)

    #save baseline preview.
    baseline_mean = load_healthy_baseline()
    baseline_preview = np.uint8(np.clip(baseline_mean * 255.0, 0, 255))
    cv2.imwrite(str(BASELINE_DEBUG_PATH), cv2.cvtColor(baseline_preview, cv2.COLOR_RGB2BGR))

    prediction = model.predict(x_batch)
    prob_sick = float(prediction[0][0])

    label = "Sick" if prob_sick > 0.5 else "Healthy"
    confidence = prob_sick if label == "Sick" else 1.0 - prob_sick

    #use a cloned linear-output model only for DeepLIFT attribution the original model above is still used for prediction
    xai_model = make_linear_output_model(model)
    raw_sick_score = float(xai_model.predict(x_batch)[0][0])

    #explain the raw pre sigmoid Sick score instead of saturated probability.
    signed_attr_masked = compute_signed_deeplift_attribution(
        model=xai_model,
        rgb_float01=x,
        target_idx=0,
    )

    shap.image_plot(np.expand_dims(signed_attr_masked, axis=0), x_batch, show=False)

    fig = plt.gcf()
    fig.suptitle("DeepLIFT", fontsize=16)

    fig.savefig(str(COMPARISON_PATH), bbox_inches="tight", dpi=120)

    fig.canvas.draw()

    image_axes = [
        ax for ax in fig.axes
        if ax.get_images()
    ]

    if len(image_axes) >= 2:
        attribution_ax = image_axes[1]
    else:
        attribution_ax = image_axes[-1]

    bbox = attribution_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    pad = 0.03
    bbox = bbox.expanded(1.0 + pad, 1.0 + pad)

    fig.savefig(str(OUTPUT_PATH), bbox_inches=bbox, dpi=160)

    plt.close(fig)

    coords = get_top_coordinates_from_attribution(signed_attr_masked) if label == "Sick" else []

    output_data = {
        "status": "success",
        "method": "DeepLIFT",
        "prediction": label,
        "confidence_score": float(confidence),
        "raw_sick_score": raw_sick_score,
        "visualizations": [
            {
                "type": "DeepLIFT_Masked",
                "image_url": os.path.abspath(str(OUTPUT_PATH)),
                "comparison_image_url": os.path.abspath(str(COMPARISON_PATH)),
                "coordinates": coords,
            }
        ],
        "coordinates": coords,
        "coordinate_count": len(coords),
    }

    print(json.dumps(output_data))