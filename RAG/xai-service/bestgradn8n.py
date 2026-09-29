import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import sys
import json
import os
import base64

MODEL_PATH = "best_model.h5"
OUTPUT_PATH = "n8n_gradcam_result.jpg"
COMPARISON_PATH = "n8n_gradcam_comparison.jpg"


def resolve_input_path(input_data: str) -> str:
    if len(input_data) > 200:
        image_bytes = base64.b64decode(input_data)
        save_path = r"C:\Users\Dell\Desktop\PlantThesis22\input_leaf.jpg"
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        return save_path
    return input_data


def preprocess_image(img_path, input_shape):
    img = image.load_img(img_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x / 255.0


def normalize_map(values):
    values = values.astype(np.float32)
    min_val = np.min(values)
    max_val = np.max(values)

    if max_val - min_val <= 1e-10:
        return np.zeros_like(values, dtype=np.float32)

    return (values - min_val) / (max_val - min_val)


def make_linear_output_model(model):
    model.layers[-1].activation = tf.keras.activations.linear
    return model


def make_leaf_mask_bgr(original_bgr):
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([10, 20, 20])
    upper_green = np.array([100, 255, 255])
    return cv2.inRange(hsv, lower_green, upper_green)


def overlay_heatmap(heatmap, original_bgr, alpha=0.45, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(
        heatmap,
        (original_bgr.shape[1], original_bgr.shape[0])
    )

    heatmap_resized = normalize_map(heatmap_resized)
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        colormap
    )

    leaf_mask = make_leaf_mask_bgr(original_bgr)
    heatmap_color = cv2.bitwise_and(
        heatmap_color,
        heatmap_color,
        mask=leaf_mask
    )

    overlay = cv2.addWeighted(
        original_bgr,
        1.0 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay


def grad_cam(input_model, image_array, layer_name):
    last_conv_layer = input_model.get_layer(layer_name)

    intermediate_model = tf.keras.Model(
        inputs=input_model.inputs,
        outputs=[last_conv_layer.output, input_model.output],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image_array, tf.float32)
        tape.watch(inputs)
        last_conv_layer_output, model_output = intermediate_model(inputs)
        top_class_channel = model_output[:, 0]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(
        tf.multiply(pooled_grads, last_conv_layer_output),
        axis=-1
    )

    heatmap = np.maximum(heatmap, 0)[0]

    return normalize_map(heatmap)


def get_gradcam_attribution(model, img_array):
    heatmap = grad_cam(model, img_array, "conv2d_2")

    img_for_mask = (img_array[0] * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_for_mask, cv2.COLOR_RGB2HSV)
    leaf_mask = cv2.inRange(
        hsv,
        np.array([25, 35, 35]),
        np.array([95, 255, 255])
    )

    mask_small = cv2.resize(
        leaf_mask,
        (heatmap.shape[1], heatmap.shape[0])
    ) / 255.0

    return heatmap * mask_small


def get_top_coordinates(heatmap, original_shape, max_points=4, threshold=150):
    heatmap_full = cv2.resize(
        heatmap,
        (original_shape[1], original_shape[0])
    )

    heatmap_bw = np.uint8(255 * normalize_map(heatmap_full))
    _, thresh = cv2.threshold(
        heatmap_bw,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    top_contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )[:max_points]

    coords = []

    for cnt in top_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            coords.append({
                "x": int(M["m10"] / M["m00"]),
                "y": int(M["m01"] / M["m00"]),
            })

    return coords


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "message": "No image path provided."
        }))
        sys.exit(1)

    input_image_path = resolve_input_path(sys.argv[1])

    original_image = cv2.imread(input_image_path)

    if original_image is None:
        print(json.dumps({
            "status": "error",
            "message": f"Could not read image: {input_image_path}"
        }))
        sys.exit(1)

    model = load_model(MODEL_PATH)

    input_shape = model.input_shape[1:3]
    preprocessed_image = preprocess_image(input_image_path, input_shape)

    prediction = model.predict(preprocessed_image)

    prob_sick = float(prediction[0][0])
    label = "Sick" if prob_sick > 0.5 else "Healthy"
    confidence = prob_sick if label == "Sick" else 1.0 - prob_sick

    xai_model = make_linear_output_model(model)

    heatmap = grad_cam(xai_model, preprocessed_image, "conv2d_2")

    leaf_mask = make_leaf_mask_bgr(original_image)
    mask_small = cv2.resize(
        leaf_mask,
        (heatmap.shape[1], heatmap.shape[0])
    ) / 255.0

    heatmap = heatmap * mask_small

    clean_overlay = overlay_heatmap(heatmap, original_image)
    cv2.imwrite(OUTPUT_PATH, clean_overlay)

    comparison = cv2.hconcat([original_image, clean_overlay])
    cv2.imwrite(COMPARISON_PATH, comparison)

    detected_coords = get_top_coordinates(
        heatmap,
        original_image.shape
    ) if label == "Sick" else []

    result_data = {
        "status": "success",
        "method": "Grad-CAM",
        "prediction": label,
        "confidence_score": float(confidence),
        "heatmap_image_url": os.path.abspath(OUTPUT_PATH),
        "comparison_image_url": os.path.abspath(COMPARISON_PATH),
        "coordinates": detected_coords,
        "coordinate_count": len(detected_coords),
    }

    print(json.dumps(result_data))