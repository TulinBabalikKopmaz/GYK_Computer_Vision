# scripts/01_img_preprocessing.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir="data", img_size=224):
    """
    Maske tespit projesi için with_mask / without_mask görsellerini yükler.
    Görselleri yeniden boyutlandırır, normalize eder ve %80 eğitim / %20 doğrulama olarak böler.

    Args:
        data_dir (str): Ana veri klasörü (varsayılan: "data")
        img_size (int): Görsel boyutu (varsayılan: 224x224)

    Returns:
        X_train, X_val, y_train, y_val (numpy dizileri)
    """
    class_names = ["with_mask", "without_mask"]
    label_map = {"with_mask": 0, "without_mask": 1}

    images = []
    labels = []

    # Klasörlerin varlığını kontrol et
    for class_name in class_names:
        folder_path = os.path.join(data_dir, class_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Klasör bulunamadı: {folder_path}")

        label = label_map[class_name]

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img = (img / 255.0).astype("float32")  
                # normalize ederken float64 yerine float32 seçildi çünkü TensorFlow varsayılan olarak float32 ile çalışır; bu hem bellek kullanımını azaltır hem de GPU ile daha uyumludur
                images.append(img)
                labels.append(label)
            else:
                print(f"⚠️ Uyarı: Görüntü yüklenemedi: {img_path}")

    if len(images) == 0:
        raise ValueError("Hiç geçerli görüntü bulunamadı!")

    images = np.array(images)
    labels = np.array(labels)

    return train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
