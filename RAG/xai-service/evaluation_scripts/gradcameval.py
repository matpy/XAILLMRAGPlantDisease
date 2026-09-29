from pathlib import Path
import sys
import time
import math
import re

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import quantus

tf.compat.v1.disable_eager_execution()


#paths, change depending in relation to local env

EVALUATION_DIR = Path(r"C:\Users\Dell\Desktop\testset\evaluation")
IMAGE_DIR = EVALUATION_DIR / "images"
OUTPUT_DIR = EVALUATION_DIR / "outputs"
ANNOTATION_PATH = OUTPUT_DIR / "annotations_all_sick.csv"

# This should point to the folder that contains: best_model.h5, bestgradn8n.py
XAI_SERVICE_DIR = Path(r"C:\Users\Dell\n8n-project\xai-service")
MODEL_PATH = XAI_SERVICE_DIR / "best_model.h5"

RESULTS_PATH = OUTPUT_DIR / "gradcam_quantus_evaluation_results.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(XAI_SERVICE_DIR))


#settings

MODEL_SIZE = 224

MAX_IMAGES = 30

GRADCAM_LAYER_NAME = "conv2d_2"

LOCALIZATION_FRACTIONS = [0.20, 0.50, 0.70]

# Consistency/repeated run stability.
RUN_CONSISTENCY_IOU = True
CONSISTENCY_RUNS = 3
CONSISTENCY_FRACTION = 0.20

RUN_PIXEL_FLIPPING = True


PIXEL_FLIPPING_FEATURES_IN_STEP = 1024


#helpers

def fraction_tag(fraction):
    return f"top{int(round(fraction * 100))}"


def normalize_map(arr): #attribution map rescaler
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))

    if mx - mn <= 1e-10:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - mn) / (mx - mn)


def attribution_stats(attr): #attribution stat calculator
    attr = np.asarray(attr, dtype=np.float32)

    return {
        "attr_is_blank": bool(np.max(attr) - np.min(attr) <= 1e-10),
    }


def safe_number(value): #for if none or inf values
    if value is None:
        return ""

    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return ""
        return value
    except Exception:
        return ""



def get_metric_value(metric_dict, metric_name, index): #extract pixel flipping result for image
    values = metric_dict.get(metric_name, "")

    try:
        if isinstance(values, (list, tuple, np.ndarray)):
            if len(values) <= index:
                return ""
            return safe_number(values[index])

        return safe_number(values)

    except Exception:
        return ""


def load_rgb_float01(image_path): #load image and standardize
    bgr = cv2.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    bgr = cv2.resize(bgr, (MODEL_SIZE, MODEL_SIZE))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb.astype(np.float32) / 255.0


def load_image_batch(image_paths): #make output of above to array
    return np.array(
        [load_rgb_float01(path) for path in image_paths],
        dtype=np.float32,
    )


def annotation_name_from_file(image_name): #converts copied names to normal
    return re.sub(r"^\d+_", "", image_name)


def choose_annotation_name(image_name, annotations): #matches copied filenames to their original annotation filename
    names = set(annotations["image_name"].astype(str).tolist())

    if image_name in names:
        return image_name

    stripped = annotation_name_from_file(image_name)

    if stripped in names:
        return stripped

    return image_name


def category_from_filename(image_name):
    stem = Path(image_name).stem
    stem = re.sub(r"^\d+_", "", stem)
    prefix = "".join([c for c in stem if not c.isdigit()])

    mapping = {
        "Bs": "bacterial_spot",
        "Eb": "early_blight",
        "H": "healthy",
        "Lb": "late_blight",
        "Lm": "leaf_mold",
        "Slf": "septoria_leaf_spot",
        "Ts": "target_spot",
        "Tmv": "tomato_mosaic_virus",
        "Tssm": "two_spotted_spider_mite",
        "Tyl": "tomato_yellow_leaf_curl_virus",
    }

    return mapping.get(prefix, "unknown")


def predict_prob_sick(base_model, x_batch): #for running the model
    preds = base_model.predict(x_batch, verbose=0)
    return preds.reshape(-1).astype(float)


def make_quantus_compatible_model(base_model): #quantus compatibility for the output
    inputs = tf.keras.Input(shape=(MODEL_SIZE, MODEL_SIZE, 3))
    outputs = base_model(inputs)

    if outputs.shape[-1] == 1:
        combined = tf.keras.layers.Concatenate()([1.0 - outputs, outputs])
        final_output = tf.keras.layers.Activation(
            "linear",
            name="quantus_compat_out",
        )(combined)

        return tf.keras.Model(inputs=inputs, outputs=final_output)

    return base_model


def make_linear_output_model(model): #pre sigmoid used for raw scores
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


def make_leaf_mask_rgb_float(rgb_float01):#mask maker
    rgb_uint8 = np.uint8(np.clip(rgb_float01 * 255.0, 0, 255))
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)

    lower_green = np.array([10, 20, 20])
    upper_green = np.array([100, 255, 255])

    return cv2.inRange(hsv, lower_green, upper_green)




def load_annotations():
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {ANNOTATION_PATH}\n"
            "Run validate_annotations.py first."
        )

    df = pd.read_csv(ANNOTATION_PATH)

    required = {
        "image_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "image_width",
        "image_height",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    for col in [
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "image_width",
        "image_height",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["bbox_width"] > 0) & (df["bbox_height"] > 0)].copy()

    return df


def scale_box(row, padding=0):
    src_w = float(row["image_width"])
    src_h = float(row["image_height"])

    if src_w <= 0 or src_h <= 0:
        return None

    x1 = int(round(float(row["bbox_x"]) * MODEL_SIZE / src_w))
    y1 = int(round(float(row["bbox_y"]) * MODEL_SIZE / src_h))
    x2 = int(round((float(row["bbox_x"]) + float(row["bbox_width"])) * MODEL_SIZE / src_w))
    y2 = int(round((float(row["bbox_y"]) + float(row["bbox_height"])) * MODEL_SIZE / src_h))

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    x1 = max(0, min(MODEL_SIZE - 1, x1))
    y1 = max(0, min(MODEL_SIZE - 1, y1))
    x2 = max(0, min(MODEL_SIZE, x2))
    y2 = max(0, min(MODEL_SIZE, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def ground_truth_mask(image_name, annotations, padding=0):
    rows = annotations[annotations["image_name"].astype(str) == image_name]

    if rows.empty:
        return None, 0

    mask = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=bool)
    valid_boxes = 0

    for _, row in rows.iterrows():
        box = scale_box(row, padding=padding)

        if box is None:
            continue

        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = True
        valid_boxes += 1

    if mask.sum() == 0:
        return None, valid_boxes

    return mask, valid_boxes


#localization and consistency

def top_mask(attr_map, fraction): #selects the strongest nonzero attribution pixels by rank, limited by the number of positive pixels
    attr = normalize_map(attr_map)

    flat = attr.flatten()
    positive_indices = np.where(flat > 0)[0]

    if len(positive_indices) == 0:
        return np.zeros_like(attr, dtype=bool)

    total_pixels = flat.size
    k = int(np.ceil(fraction * total_pixels))
    k = max(1, min(k, len(positive_indices)))

    positive_values = flat[positive_indices]

    top_local_indices = np.argpartition(positive_values, -k)[-k:]
    top_global_indices = positive_indices[top_local_indices]

    mask_flat = np.zeros_like(flat, dtype=bool)
    mask_flat[top_global_indices] = True

    return mask_flat.reshape(attr.shape)


def mask_iou(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return ""

    return float(intersection / union)


def pairwise_iou_stats(masks):
    if len(masks) < 2:
        return {
            "mean_iou": "",
            "min_iou": "",
            "max_iou": "",
            "std_iou": "",
        }

    values = []

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            value = mask_iou(masks[i], masks[j])

            if value != "":
                values.append(value)

    if not values:
        return {
            "mean_iou": "",
            "min_iou": "",
            "max_iou": "",
            "std_iou": "",
        }

    values = np.array(values, dtype=np.float32)

    return {
        "mean_iou": float(np.mean(values)),
        "min_iou": float(np.min(values)),
        "max_iou": float(np.max(values)),
        "std_iou": float(np.std(values)),
    }


def localization_score(attr_map, gt_mask, fraction): #of highlighted pixels, what fraction is inside the symptom annotation boxes
    if gt_mask is None or gt_mask.sum() == 0:
        return None

    pred_mask = top_mask(attr_map, fraction)

    if pred_mask.sum() == 0:
        return 0.0

    inside = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum()

    return float(inside / total)


def box_coverage_metrics(image_name, annotations, attr_map, fraction, padding=0): #how many boxes are touched and how much of it are covered
    rows = annotations[annotations["image_name"].astype(str) == image_name]

    if rows.empty:
        return {
            "boxes_hit": "",
            "total_boxes": 0,
            "box_hit_rate": "",
            "mean_box_coverage": "",
        }

    highlighted = top_mask(attr_map, fraction)

    boxes_hit = 0
    coverages = []

    for _, row in rows.iterrows():
        box = scale_box(row, padding=padding)

        if box is None:
            continue

        x1, y1, x2, y2 = box

        box_mask = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=bool)
        box_mask[y1:y2, x1:x2] = True

        box_area = box_mask.sum()

        if box_area == 0:
            continue

        overlap = np.logical_and(highlighted, box_mask).sum()
        coverage = float(overlap / box_area)

        coverages.append(coverage)

        if overlap > 0:
            boxes_hit += 1

    total_boxes = len(coverages)

    if total_boxes == 0:
        return {
            "boxes_hit": "",
            "total_boxes": 0,
            "box_hit_rate": "",
            "mean_box_coverage": "",
        }

    return {
        "boxes_hit": boxes_hit,
        "total_boxes": total_boxes,
        "box_hit_rate": float(boxes_hit / total_boxes),
        "mean_box_coverage": float(np.mean(coverages)),
    }


#quantus helpers

def reduce_metric_element(value):
    try:
        arr = np.array(value, dtype=float)

        if arr.size == 0:
            return ""

        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return ""

        return float(np.mean(arr))

    except Exception:
        return safe_number(value)


def standardize_quantus_result(result, n_images):
    if result is None:
        return [""] * n_images

    if isinstance(result, dict):
        try:
            result = list(result.values())
        except Exception:
            return [""] * n_images

    if isinstance(result, np.ndarray):
        if result.ndim == 0:
            return [safe_number(float(result))] * n_images

        if result.shape[0] == n_images:
            return [reduce_metric_element(result[i]) for i in range(n_images)]

        return [reduce_metric_element(result)] * n_images

    if isinstance(result, (list, tuple)):
        if len(result) == n_images:
            return [reduce_metric_element(x) for x in result]

        return [reduce_metric_element(result)] * n_images

    return [safe_number(result)] * n_images


def run_quantus_metric(metric_class_name, attempts, call_kwargs, n_images):
    metric_class = getattr(quantus, metric_class_name, None)

    if metric_class is None:
        print(f"Quantus {metric_class_name} not available in this Quantus version.")
        return [""] * n_images, "not_available"

    last_error = None

    for kwargs in attempts:
        try:
            metric = metric_class(**kwargs)
            values = metric(**call_kwargs)
            print(f"Quantus {metric_class_name} succeeded with args: {kwargs}")
            return standardize_quantus_result(values, n_images), metric_class_name

        except Exception as error:
            last_error = error
            print(f"Quantus {metric_class_name} failed with args {kwargs}: {error}")

    print(f"All Quantus attempts failed for {metric_class_name}. Last error: {last_error}")
    return [""] * n_images, "failed"


#gradcam wrapper section

def grad_cam_graph_mode(model, image_batch, layer_name, target_index=0): #explains the raw sick score via the cloned linear output model, instead of sigmoid probability
    #too much saturation in some images otherwise
    from tensorflow.keras import backend as K

    conv_layer = model.get_layer(layer_name)

    target_output = model.output[:, target_index]

    grads = K.gradients(target_output, conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function(
        [model.input],
        [pooled_grads, conv_layer.output[0]],
    )

    pooled_grads_value, conv_layer_output_value = iterate([image_batch])

    for channel_idx in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, channel_idx] *= pooled_grads_value[channel_idx]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    max_value = np.max(heatmap)

    if max_value <= 1e-10:
        return np.zeros_like(heatmap, dtype=np.float32)

    heatmap = heatmap / max_value

    return heatmap.astype(np.float32)


def build_gradcam_wrapper(base_model):
    import bestgradn8n

    print("Using Grad-CAM xai-service version from:", bestgradn8n.__file__)
    print(f"Using Grad-CAM layer: {GRADCAM_LAYER_NAME}")
    xai_model = make_linear_output_model(base_model)

    def gradcam_wrapper(model, inputs, targets, **kwargs):
        maps = []

        for idx, x in enumerate(inputs):
            image_batch = np.expand_dims(x, axis=0).astype(np.float32)

            heatmap_small = grad_cam_graph_mode(
                model=xai_model,
                image_batch=image_batch,
                layer_name=GRADCAM_LAYER_NAME,
                target_index=0,
            )

            heatmap_full = cv2.resize(
                heatmap_small,
                (MODEL_SIZE, MODEL_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

            leaf_mask = make_leaf_mask_rgb_float(x).astype(np.float32) / 255.0
            heatmap_full = heatmap_full * leaf_mask

            attr = normalize_map(heatmap_full)
            maps.append(attr)

        return np.array(maps, dtype=np.float32)

    return gradcam_wrapper


# consistency

def compute_gradcam_consistency_iou(explain_func, x, fraction=0.20, runs=3):#runs gradcam multiple times on the same image,
# converts each attribution map to a topk binary mask and computes pairwise IoU

    if runs < 2:
        return {
            "mean_iou": "",
            "min_iou": "",
            "max_iou": "",
            "std_iou": "",
        }

    masks = []

    single_input = np.expand_dims(x, axis=0)
    dummy_target = np.array([1], dtype=int)

    for _ in range(runs):
        attr_batch = explain_func(None, single_input, dummy_target)
        attr_map = attr_batch[0]
        masks.append(top_mask(attr_map, fraction))

    return pairwise_iou_stats(masks)


#main

def main():
    print("Starting Grad-CAM evaluation")

    annotations = load_annotations()

    image_paths = sorted([
        p for p in IMAGE_DIR.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    print(f"Images to evaluate: {len(image_paths)}")
    print(f"Results path: {RESULTS_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    base_model = tf.keras.models.load_model(str(MODEL_PATH))
    quantus_model = make_quantus_compatible_model(base_model)

    x_batch = load_image_batch(image_paths)

    prob_sick = predict_prob_sick(base_model, x_batch)
    y_batch = (prob_sick > 0.5).astype(int)

    explain_func = build_gradcam_wrapper(base_model)

    print("Generating Grad-CAM attribution maps...")
    a_batch = explain_func(quantus_model, x_batch, y_batch)

    print("Running selected Quantus metrics...")

    base_call_kwargs = dict(
        model=quantus_model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        explain_func=explain_func,
        device="cpu",
        channel_first=False,
    )

    metric_values = {}
    metric_sources = {}

    if RUN_PIXEL_FLIPPING:
        values, source = run_quantus_metric(
            "PixelFlipping",
            attempts=[
                {
                    "features_in_step": PIXEL_FLIPPING_FEATURES_IN_STEP,
                    "perturb_baseline": "black",
                    "disable_warnings": True,
                },
                {
                    "features_in_step": PIXEL_FLIPPING_FEATURES_IN_STEP,
                    "perturb_baseline": "mean",
                    "disable_warnings": True,
                },
                {
                    "disable_warnings": True,
                },
            ],
            call_kwargs=base_call_kwargs,
            n_images=len(image_paths),
        )
        metric_values["quantus_pixel_flipping"] = values
        metric_sources["quantus_pixel_flipping"] = source
    else:
        print("Skipping Quantus PixelFlipping.")
        metric_values["quantus_pixel_flipping"] = [""] * len(image_paths)
        metric_sources["quantus_pixel_flipping"] = "skipped"

#results

    rows = []

    for i, image_path in enumerate(image_paths):
        image_name = image_path.name
        annotation_image_name = choose_annotation_name(image_name, annotations)

        category = category_from_filename(annotation_image_name)
        ground_truth_label = "Healthy" if category == "healthy" else "Sick"
        prediction_label = "Sick" if y_batch[i] == 1 else "Healthy"

        pf = get_metric_value(metric_values, "quantus_pixel_flipping", i)

        gt_mask_main, box_count = ground_truth_mask(annotation_image_name, annotations, padding=0)

        attr_stats = attribution_stats(a_batch[i])

        if RUN_CONSISTENCY_IOU:
            consistency_stats = compute_gradcam_consistency_iou(
                explain_func=explain_func,
                x=x_batch[i],
                fraction=CONSISTENCY_FRACTION,
                runs=CONSISTENCY_RUNS,
            )
        else:
            consistency_stats = {
                "mean_iou": "",
                "min_iou": "",
                "max_iou": "",
                "std_iou": "",
            }

        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image_name": image_name,
            "annotation_image_name": annotation_image_name,
            "category": category,
            "ground_truth_label": ground_truth_label,
            "annotation_box_count": box_count,

            "classifier_prediction": prediction_label,
            "classifier_confidence": prob_sick[i] if y_batch[i] == 1 else 1.0 - prob_sick[i],
            "prob_sick": prob_sick[i],

            "gradcam_layer_name": GRADCAM_LAYER_NAME,

            "gradcam_attr_is_blank": attr_stats["attr_is_blank"],

            "gradcam_consistency_runs": CONSISTENCY_RUNS if RUN_CONSISTENCY_IOU else "",
            "gradcam_consistency_fraction": CONSISTENCY_FRACTION if RUN_CONSISTENCY_IOU else "",
            "gradcam_consistency_mean_iou": safe_number(consistency_stats["mean_iou"]),

            "gradcam_quantus_pixel_flipping": pf,

        }

        for fraction in LOCALIZATION_FRACTIONS:
            tag = fraction_tag(fraction)

            gt_mask, _ = ground_truth_mask(annotation_image_name, annotations, padding=0)

            loc = localization_score(a_batch[i], gt_mask, fraction)

            box_metrics = box_coverage_metrics(
                image_name=annotation_image_name,
                annotations=annotations,
                attr_map=a_batch[i],
                fraction=fraction,
                padding=0,
            )

            row[f"gradcam_localization_{tag}"] = safe_number(loc)
            row[f"gradcam_box_hit_rate_{tag}"] = safe_number(box_metrics["box_hit_rate"])
            row[f"gradcam_boxes_hit_{tag}"] = box_metrics["boxes_hit"]
            row[f"gradcam_total_boxes_{tag}"] = box_metrics["total_boxes"]
            row[f"gradcam_mean_box_coverage_{tag}"] = safe_number(box_metrics["mean_box_coverage"])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)

    print("\nDone.")
    print(f"Saved: {RESULTS_PATH}")
    print(df)


if __name__ == "__main__":
    main()