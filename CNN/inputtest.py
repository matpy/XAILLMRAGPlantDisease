import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_sick_or_healthy(model, img_array):
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return 'sick'
    else:
        return 'healthy'

def main():
    model_path = 'best_model.h5'
    img_path = '0036c89d-7743-4895-9fcf-b8d2c1fc8455___YLCV_NREC 0313.jpg'
    target_size = (224, 224)

    # Load the trained model
    model = load_model(model_path)
    print(model.summary())
    # Load and preprocess the input image
    img_array = load_and_preprocess_image(img_path, target_size)

    # Predict if the image is sick or healthy
    result = predict_sick_or_healthy(model, img_array)
    print(f'The input image is predicted to be {result}.')

if __name__ == '__main__':
    main()
