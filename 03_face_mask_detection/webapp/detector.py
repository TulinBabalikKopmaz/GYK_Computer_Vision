import numpy as np
import cv2
from keras.models import load_model

model = load_model("models/mobilenet_model.h5")
CLASS_NAMES = ["With Mask", "Without Mask"]

def predict_mask(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0

    resized = cv2.resize(img, (224, 224)) / 255.0
    array = np.expand_dims(resized, axis=0)
    prediction = model.predict(array, verbose=0)[0][0]

    label = "Without Mask" if prediction > 0.5 else "With Mask"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, round(confidence * 100, 2)
