import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load your trained CNN model
model = load_model('best_model.h5')

def preprocess_image(img_path, input_shape):
    img = image.load_img(img_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

def grad_cam(input_model, image_array, layer_name):
    last_conv_layer = input_model.get_layer(layer_name)
    intermediate_model = tf.keras.Model(inputs=input_model.inputs, outputs=[last_conv_layer.output, input_model.output])

    with tf.GradientTape() as tape:
        inputs = tf.cast(image_array, tf.float32)
        tape.watch(inputs)
        last_conv_layer_output, model_output = intermediate_model(inputs)
        top_class_channel = model_output[:, np.argmax(model_output)]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap[0]

    # Normalize the heatmap
    min_val, max_val = np.min(heatmap), np.max(heatmap)
    if max_val - min_val > 0:
        heatmap = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap

# Overlay heatmap on the original image
def overlay_heatmap(heatmap, original_img, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    combined = cv2.addWeighted(original_img, alpha, heatmap, 1 - alpha, 0)
    return combined
# Load and preprocess the image
input_image_path = '00355ec2-f12c-4015-b8f9-94354f69ce22___RS_Late.B 5311.jpg'
original_image = cv2.imread(input_image_path)
input_shape = model.input_shape[1:3]
preprocessed_image = preprocess_image(input_image_path, input_shape)

# Classify the image using the model
prediction = model.predict(preprocessed_image)
label = 'Sick' if np.argmax(prediction) == 0 else 'Healthy'

# Apply Grad-CAM to generate the heatmap
heatmap = grad_cam(model, preprocessed_image, 'conv2d_2')

# Overlay the heatmap on the original image
combined_image = overlay_heatmap(heatmap, original_image)

# Determine the sick region based on red color in the heatmap
red_threshold = 200
sick_region_mask = (combined_image[:, :, 2] > red_threshold)
sick_region_coordinates = np.argwhere(sick_region_mask)

# Add text to the image
text_scale = 0.5  # Adjust the text size
text = f"Plant Status: {label}"
cv2.putText(combined_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 2)

if label == 'Sick':
    sick_region_text_lines = [f"Sick region coordinates:"]
    for coordinate in sick_region_coordinates:
        sick_region_text_lines.append(f"{coordinate}")
    sick_region_text = "\n".join(sick_region_text_lines)

    y_pos = 60
    for line in sick_region_text.split('\n'):
        cv2.putText(combined_image, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 2)
        y_pos += 30

# Save the output image
cv2.imwrite('image_with_heatmap_and_sick_region3.jpg', combined_image)

# Display the output image
cv2.imshow('Grad-CAM', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the output image
cv2.imshow('Grad-CAM', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()