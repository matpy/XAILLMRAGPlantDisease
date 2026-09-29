import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import sys
import json
import os

MODEL_PATH = "best_model.h5"
OUTPUT_PATH = "n8n_lime_result.jpg"
COMPARISON_PATH = "n8n_lime_comparison.jpg"

model = load_model(MODEL_PATH)
explainer = lime_image.LimeImageExplainer()


def find_affected_areas(heatmap, threshold=128):
    _, thresh_heatmap = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def preprocess_image(img_path, input_shape):
    img = image.load_img(img_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x / 255.0


def segment_plant(image_bgr, lower_hsv, upper_hsv):
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    masked_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return masked_image, mask


def apply_mask_to_heatmap(heatmap, mask):
    #convert any positive LIME region to 255 and everything else to 0.
    heatmap_binary = np.where(heatmap > 0, 255, 0).astype(np.uint8)

    return cv2.bitwise_and(heatmap_binary, heatmap_binary, mask=mask)


def lime_heatmap(input_model, image_array, label_idx=1, num_samples=1000, hide_color=None):

    def model_predict(images):
        # images are already in 0-1 range because image_array[0] is 0-1.
        p_sick = input_model.predict(images, verbose=0)

        #safety in case shape is (N,) instead of (N, 1)
        p_sick = np.asarray(p_sick).reshape(-1, 1)

        p_healthy = 1.0 - p_sick

        return np.concatenate([p_healthy, p_sick], axis=1)

    explanation = explainer.explain_instance(
        image_array[0],
        model_predict,
        labels=(label_idx,),
        hide_color=hide_color,
        num_samples=num_samples,
    )

    _, mask = explanation.get_image_and_mask(
        label=label_idx,
        positive_only=True,
        num_features=8,
        hide_rest=False,
    )

    return mask


def get_lime_attribution(input_model, image_array, label_idx):
    mask = lime_heatmap(
        input_model,
        np.expand_dims(image_array, axis=0),
        label_idx=label_idx,
        num_samples=1000,
        hide_color=0,
    )
    img_uint8 = (image_array * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    _, plant_mask = segment_plant(img_bgr, np.array([25, 50, 50]), np.array([90, 255, 255]))
    resized_plant_mask = cv2.resize(plant_mask, (mask.shape[1], mask.shape[0]))
    masked_heatmap = apply_mask_to_heatmap(mask, resized_plant_mask)
    return masked_heatmap.astype(np.float32) / 255.0


def make_lime_overlay(original_bgr_resized, masked_heatmap):
    original_rgb = cv2.cvtColor(original_bgr_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay_rgb = mark_boundaries(original_rgb, masked_heatmap, color=(0, 1, 1), mode="thick")
    overlay_bgr = cv2.cvtColor((overlay_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return overlay_bgr


def coords_from_contours(contours):
    coords = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        coords.append({"x": int(x + w / 2), "y": int(y + h / 2), "w": int(w), "h": int(h)})
    return coords

def lime_mask_quality(masked_heatmap):
    nonzero_pixels = int(np.count_nonzero(masked_heatmap))
    total_pixels = int(masked_heatmap.size)

    nonzero_fraction = nonzero_pixels / total_pixels if total_pixels > 0 else 0.0

    quality = "informative" if nonzero_pixels > 0 else "weak_or_zero_attribution"

    return {
        "quality": quality,
        "nonzero_pixels": nonzero_pixels,
        "nonzero_fraction": float(nonzero_fraction),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No image path provided."}))
        sys.exit(1)

    input_image_path = sys.argv[1]
    original_image = cv2.imread(input_image_path)
    if original_image is None:
        print(json.dumps({"status": "error", "message": f"Could not read image: {input_image_path}"}))
        sys.exit(1)

    input_shape = model.input_shape[1:3]
    preprocessed_image = preprocess_image(input_image_path, input_shape)

    lower_hsv = np.array([25, 50, 50])
    upper_hsv = np.array([90, 255, 255])
    original_image_resized = cv2.resize(original_image, input_shape[::-1])
    _, plant_mask = segment_plant(original_image_resized, lower_hsv, upper_hsv)

    prediction = model.predict(preprocessed_image)
    prob_sick = float(prediction[0][0])
    label = "Sick" if prob_sick > 0.5 else "Healthy"
    confidence = prob_sick if label == "Sick" else 1.0 - prob_sick

    affected_areas = []

    lime_label_idx = 1 if label == "Sick" else 0

    if label == "Sick":
        heatmap = lime_heatmap(
            model,
            preprocessed_image,
            label_idx=lime_label_idx,
            num_samples=1000,
            hide_color=0,
        )

        resized_plant_mask = cv2.resize(plant_mask, (heatmap.shape[1], heatmap.shape[0]))
        masked_heatmap = apply_mask_to_heatmap(heatmap, resized_plant_mask)

        affected_areas = find_affected_areas(masked_heatmap)
        overlayed_image = make_lime_overlay(original_image_resized, masked_heatmap)

    else:
        masked_heatmap = np.zeros(input_shape, dtype=np.uint8)
        overlayed_image = original_image_resized.copy()

    quality_info = lime_mask_quality(masked_heatmap)

    cv2.imwrite(OUTPUT_PATH, overlayed_image)

    #optional debug img
    comparison = cv2.hconcat([original_image_resized, overlayed_image])
    cv2.imwrite(COMPARISON_PATH, comparison)

    coords = coords_from_contours(affected_areas) if label == "Sick" else []

    result_data = {
        "status": "success",
        "method": "LIME",
        "prediction": label,
        "confidence_score": float(confidence),
        "explanation_image_url": os.path.abspath(OUTPUT_PATH),
        "comparison_image_url": os.path.abspath(COMPARISON_PATH),
        "superpixel_count": len(affected_areas),
        "affected_coordinates": coords,
        "attribution_quality": quality_info["quality"],
        "nonzero_pixels": quality_info["nonzero_pixels"],
        "nonzero_fraction": quality_info["nonzero_fraction"],
    }

    print(json.dumps(result_data))
