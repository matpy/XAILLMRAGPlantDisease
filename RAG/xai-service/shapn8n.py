import os
import sys
import json
import gc
from pathlib import Path

import cv2
import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


tf.compat.v1.disable_eager_execution()


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.h5"

INPAINT_OUTPUT = BASE_DIR / "n8n_shap_inpaint_masker.jpg"
BLUR_OUTPUT = BASE_DIR / "n8n_shap_blur_masker.jpg"

INPAINT_COMPARISON = BASE_DIR / "n8n_shap_inpaint_comparison.jpg"
BLUR_COMPARISON = BASE_DIR / "n8n_shap_blur_comparison.jpg"

DEBUG_LEAF_MASK = BASE_DIR / "debug_shap_leaf_mask.jpg"

#safe Docker defaults, can override these in Docker env later if needed
DEFAULT_MAX_EVALS = int(os.environ.get("SHAP_MAX_EVALS", "600"))
DEFAULT_BATCH_SIZE = int(os.environ.get("SHAP_BATCH_SIZE", "30"))


def log(message):
    print(f"SHAP_DEBUG: {message}", file=sys.stderr, flush=True)


def normalize_map(values):
    values = values.astype(np.float32)
    min_val = float(np.min(values))
    max_val = float(np.max(values))

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


def make_leaf_mask_from_bgr(original_bgr):
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)

    lower_bg = np.array([0, 0, 40])
    upper_bg = np.array([180, 45, 220])

    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    leaf_mask = cv2.bitwise_not(bg_mask)

    return leaf_mask


def extract_importance_map(shap_values_obj, mask_3d, target_idx=0):
    raw_values = np.squeeze(shap_values_obj.values)

    if raw_values.ndim != 3:
        raise ValueError(
            f"Expected SHAP values with 3 dimensions after squeezing, received {raw_values.shape}"
        )

    importance_map = np.abs(raw_values * mask_3d).sum(axis=-1)

    if np.max(importance_map) <= 1e-12:
        importance_map = np.abs(raw_values).sum(axis=-1)

    return normalize_map(importance_map)


def get_attribution_quality(shap_values_obj, zero_threshold=1e-10):
    raw_values = np.squeeze(shap_values_obj.values)

    max_abs = float(np.max(np.abs(raw_values)))
    mean_abs = float(np.mean(np.abs(raw_values)))

    if max_abs <= zero_threshold:
        quality = "weak_or_zero_attribution"
    else:
        quality = "informative"

    return {
        "quality": quality,
        "max_abs_attribution": max_abs,
        "mean_abs_attribution": mean_abs,
    }


def coordinates_from_heatmap(importance_map, max_points=4, threshold=60):
    heatmap_bw = np.uint8(255 * normalize_map(importance_map))

    _, thresh = cv2.threshold(heatmap_bw, threshold, 255, cv2.THRESH_BINARY)

    kernel_dilate = np.ones((11, 11), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_points]

    coords = []

    for cnt in contours:
        moments = cv2.moments(cnt)

        if moments["m00"] != 0:
            x, y, w, h = cv2.boundingRect(cnt)

            coords.append({
                "x": int(moments["m10"] / moments["m00"]),
                "y": int(moments["m01"] / moments["m00"]),
                "w": int(w),
                "h": int(h),
            })

    return coords


def apply_leaf_mask_to_shap_values(shap_values_obj, mask_3d):
    values = shap_values_obj.values[0]

    if values.ndim != 4:
        raise ValueError(
            f"Expected SHAP values with 4 dimensions before masking, received {values.shape}"
        )

    shap_values_obj.values[0] = values * mask_3d[:, :, :, np.newaxis]

    return shap_values_obj


def render_shap_images(shap_value_obj, img_array_rgb_255, comparison_path, clean_output_path):
    plt.close("all")

    shap.image_plot(
        shap_value_obj,
        img_array_rgb_255 / 255.0,
        show=False,
    )

    fig = plt.gcf()
    fig.canvas.draw()

    fig.savefig(str(comparison_path), bbox_inches="tight", dpi=140)

    image_axes = [
        ax for ax in fig.axes
        if len(ax.get_images()) > 0
    ]

    if len(image_axes) >= 2:
        attribution_ax = image_axes[1]
    elif len(image_axes) == 1:
        attribution_ax = image_axes[0]
    else:
        fig.savefig(str(clean_output_path), bbox_inches="tight", dpi=140)
        plt.close(fig)
        return

    bbox = attribution_ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )

    bbox = bbox.expanded(1.05, 1.05)

    fig.savefig(str(clean_output_path), bbox_inches=bbox, dpi=180)

    plt.close(fig)


def get_kernel_shap_attribution(
    external_model,
    external_image_array,
    mask_type="inpaint",
    target_idx=0,
    max_evals=100,
    batch_size=2,
    use_logit=True,
    apply_leaf_mask=True,
):
    img_width, img_height = 224, 224

    img_norm = external_image_array.astype("float32")

    if img_norm.max() > 1.5:
        img_norm = img_norm / 255.0

    img_norm = np.clip(img_norm, 0.0, 1.0)

    img_255 = img_norm * 255.0
    input_tensor = np.expand_dims(img_255, axis=0).astype("float32")

    def predict_for_shap(x_255):
        tmp = x_255.astype(np.float32) / 255.0
        tmp = np.clip(tmp, 0.0, 1.0)

        p = external_model.predict(tmp, verbose=0)

        if use_logit:
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            return np.log(p / (1.0 - p))

        return p

    if mask_type == "blur":
        masker = shap.maskers.Image("blur(64,64)", (img_width, img_height, 3))
    elif mask_type == "inpaint":
        masker = shap.maskers.Image("inpaint_telea", (img_width, img_height, 3))
    else:
        raise ValueError("mask_type must be either 'blur' or 'inpaint'.")

    explainer = shap.Explainer(
        predict_for_shap,
        masker,
        output_names=["Sick logit"] if use_logit else ["Sick probability"],
    )

    shap_values = explainer(
        input_tensor,
        max_evals=max_evals,
        batch_size=batch_size,
    )

    img_uint8 = np.uint8(np.clip(img_255, 0, 255))
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    leaf_mask = make_leaf_mask_from_bgr(img_bgr)
    mask_3d = np.repeat(leaf_mask[:, :, np.newaxis], 3, axis=2) / 255.0

    if apply_leaf_mask:
        shap_values = apply_leaf_mask_to_shap_values(shap_values, mask_3d)
    else:
        mask_3d = np.ones_like(mask_3d, dtype=np.float32)

    importance_map = extract_importance_map(
        shap_values[0],
        mask_3d,
        target_idx=target_idx,
    )

    del explainer
    del shap_values
    gc.collect()

    return importance_map


def build_masker_configs(img_width, img_height):
    return [
        {
            "name": "blur",
            "masker": shap.maskers.Image(
                "blur(64,64)",
                (img_width, img_height, 3),
            ),
            "output": BLUR_OUTPUT,
            "comparison": BLUR_COMPARISON,
        },
        {
            "name": "inpaint",
            "masker": shap.maskers.Image(
                "inpaint_telea",
                (img_width, img_height, 3),
            ),
            "output": INPAINT_OUTPUT,
            "comparison": INPAINT_COMPARISON,
        },
    ]


def parse_requested_masker():
    requested_masker = "both"

    if len(sys.argv) >= 3:
        requested_masker = sys.argv[2].strip().lower()

    valid_maskers = {"both", "blur", "inpaint"}

    if requested_masker not in valid_maskers:
        print(json.dumps({
            "status": "error",
            "method": "SHAP",
            "message": f"Invalid SHAP masker: {requested_masker}. Use both, blur, or inpaint.",
        }))
        sys.exit(1)

    return requested_masker


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "method": "SHAP",
            "message": "No image path provided.",
        }))
        sys.exit(1)

    requested_masker = parse_requested_masker()
    input_image_path = sys.argv[1]

    if not os.path.exists(input_image_path):
        print(json.dumps({
            "status": "error",
            "method": "SHAP",
            "message": f"File not found: {input_image_path}",
        }))
        sys.exit(1)

    log(f"script started, requested_masker={requested_masker}")
    log(f"input_image_path={input_image_path}")

    model = None
    xai_model = None

    try:
        log("loading probability model")
        model = tf.keras.models.load_model(str(MODEL_PATH))

        img_width = model.input_shape[1]
        img_height = model.input_shape[2]

        log(f"model input size: {img_width}x{img_height}")

        img = image.load_img(
            input_image_path,
            target_size=(img_width, img_height),
        )

        img_array = image.img_to_array(img).astype(np.float32)
        input_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

        def predict_probability(x):
            tmp = x.astype(np.float32) / 255.0
            tmp = np.clip(tmp, 0.0, 1.0)
            return model.predict(tmp, verbose=0)

        log("running probability prediction")
        prediction = predict_probability(input_tensor)
        prob_sick = float(prediction[0][0])

        label = "Sick" if prob_sick > 0.5 else "Healthy"
        confidence = prob_sick if label == "Sick" else 1.0 - prob_sick

        log(f"prediction={label}, confidence={confidence}")

        log("creating linear-output XAI model from already-loaded model")
        xai_model = make_linear_output_model(model)

        log("deleting probability model before SHAP")
        del model
        model = None
        gc.collect()

        def explain_raw_score(x):
            tmp = x.astype(np.float32) / 255.0
            tmp = np.clip(tmp, 0.0, 1.0)
            return xai_model.predict(tmp, verbose=0)

        original_bgr = cv2.imread(input_image_path)

        if original_bgr is None:
            raise FileNotFoundError(f"OpenCV could not read image: {input_image_path}")

        original_bgr = cv2.resize(original_bgr, (img_width, img_height))

        leaf_mask = make_leaf_mask_from_bgr(original_bgr)
        cv2.imwrite(str(DEBUG_LEAF_MASK), leaf_mask)

        mask_3d = np.repeat(leaf_mask[:, :, np.newaxis], 3, axis=2) / 255.0

        masker_configs = build_masker_configs(img_width, img_height)

        if requested_masker != "both":
            masker_configs = [
                config for config in masker_configs
                if config["name"] == requested_masker
            ]

        visualizations = []
        combined_coordinates = {}

        for config in masker_configs:
            explainer = None
            shap_values = None

            try:
                log(
                    f"starting masker={config['name']}, "
                    f"max_evals={DEFAULT_MAX_EVALS}, "
                    f"batch_size={DEFAULT_BATCH_SIZE}"
                )

                explainer = shap.Explainer(
                    explain_raw_score,
                    config["masker"],
                    output_names=["Sick raw score"],
                )

                shap_values = explainer(
                    input_tensor,
                    max_evals=DEFAULT_MAX_EVALS,
                    batch_size=DEFAULT_BATCH_SIZE,
                )

                log(f"finished shap computation for masker={config['name']}")

                attribution_info = get_attribution_quality(shap_values[0])

                shap_values = apply_leaf_mask_to_shap_values(shap_values, mask_3d)

                render_shap_images(
                    shap_values[0],
                    img_array,
                    config["comparison"],
                    config["output"],
                )

                log(f"rendered images for masker={config['name']}")

                importance_map = extract_importance_map(
                    shap_values[0],
                    mask_3d,
                    target_idx=0,
                )

                coords = coordinates_from_heatmap(importance_map) if label == "Sick" else []

                combined_coordinates[config["name"]] = coords

                visualizations.append({
                    "masker": config["name"],
                    "image_url": os.path.abspath(str(config["output"])),
                    "comparison_image_url": os.path.abspath(str(config["comparison"])),
                    "coordinates": coords,
                    "attribution_quality": attribution_info["quality"],
                    "max_abs_attribution": attribution_info["max_abs_attribution"],
                    "mean_abs_attribution": attribution_info["mean_abs_attribution"],
                })

                log(f"completed masker={config['name']}")

            except Exception as masker_error:
                log(f"masker failed: {config['name']}: {masker_error}")

                combined_coordinates[config["name"]] = []

                visualizations.append({
                    "masker": config["name"],
                    "status": "error",
                    "message": str(masker_error),
                    "coordinates": [],
                    "attribution_quality": "error",
                    "max_abs_attribution": 0.0,
                    "mean_abs_attribution": 0.0,
                })

            finally:
                del explainer
                del shap_values
                gc.collect()

        successful_visualizations = [
            item for item in visualizations
            if item.get("status") != "error"
        ]

        status = "success" if successful_visualizations else "error"

        output_data = {
            "status": status,
            "method": "SHAP",
            "prediction": label,
            "confidence_score": float(confidence),
            "requested_masker": requested_masker,
            "max_evals": DEFAULT_MAX_EVALS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "visualizations": visualizations,
            "coordinates": combined_coordinates,
        }

        print(json.dumps(output_data))

    except Exception as error:
        print(json.dumps({
            "status": "error",
            "method": "SHAP",
            "message": str(error),
            "requested_masker": requested_masker,
        }))
        sys.exit(1)

    finally:
        del model
        del xai_model
        plt.close("all")
        gc.collect()