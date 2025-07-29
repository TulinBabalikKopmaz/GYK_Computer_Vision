# scripts/model_train.py

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from scripts.img_preprocessing import load_data

# -------------------------- #
# 💻 Donanım Bilgisi Göster  #
# -------------------------- #
def show_device_info():
    print("📟 Donanım Kullanımı")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU bulundu: {gpus[0].name}")
    else:
        print("⚠️ GPU bulunamadı, CPU kullanılacak.")

# ---------------------------- #
# 🧠 Model Yapısını Oluştur    #
# ---------------------------- #
def build_model(base_model):
    x = base_model.output                         # ⬅️ Base model (örneğin MobileNetV2)
    x = GlobalAveragePooling2D()(x)               # ⬅️ Global pooling
    x = Dense(256, activation='relu')(x)          # ⬅️ Dense katman 1
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)          # ⬅️ Dense katman 2
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)    # ⬅️ Çıkış katmanı
    return Model(inputs=base_model.input, outputs=output)

# ---------------------- #
# 📊 Eğitim Grafiği Çizici #
# ---------------------- #
def plot_training(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Train Acc')
    plt.plot(epochs, val_acc, 'r', label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs("results/loss_accuracy_graphs", exist_ok=True)
    plt.savefig(f"results/loss_accuracy_graphs/{model_name}_plot.png")
    plt.show()

# ---------------------- #
# 🚀 2 Aşamalı Eğitim     #
# ---------------------- #
# 🔁 Eğitim fonksiyonu (2 aşamalı: freeze + fine-tune)
def train_model(name, base_model_fn, model_prefix):
    print(f"🚀 {name} modeli eğitiliyor...")
    X_train, X_val, y_train, y_val = load_data()

    base_model = base_model_fn(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = False
    model = build_model(base_model)

    os.makedirs("models", exist_ok=True)
    with open(f"models/{model_prefix}_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(f"models/{model_prefix}_model.h5", monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # 🧠 1. Aşama: sadece dense katmanlar eğitilir
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=20, batch_size=32, callbacks=[checkpoint, early_stop])

    plot_training(history, model_prefix)
    evaluate_model(f"models/{model_prefix}_model.h5", X_val, y_val)
    predict_examples(f"models/{model_prefix}_model.h5", X_val, y_val)

    # 🔓 2. Aşama: fine-tune (son 30 katmanı aç)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    history_finetune = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                 epochs=10, batch_size=32, callbacks=[checkpoint, early_stop])

    plot_training(history_finetune, model_prefix + "_finetune")
    evaluate_model(f"models/{model_prefix}_model.h5", X_val, y_val)
    predict_examples(f"models/{model_prefix}_model.h5", X_val, y_val)

# ---------------------- #
# 🔁 Model Eğitim Çağrıları #
# ---------------------- #
def train_mobilenet():
    train_model("MobileNetV2", MobileNetV2, "mobilenet")

def train_resnet():
    train_model("ResNet50", ResNet50, "resnet")

def train_efficientnet():
    train_model("EfficientNetB0", EfficientNetB0, "efficientnet")

# --------------------------- #
# 📈 Değerlendirme Fonksiyonu #
# --------------------------- #
def evaluate_model(model_path, X_val, y_val):
    model = load_model(model_path)
    y_pred = (model.predict(X_val) > 0.5).astype("int32")

    print("\n📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["With Mask", "Without Mask"]))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["With Mask", "Without Mask"],
                yticklabels=["With Mask", "Without Mask"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    os.makedirs("results/confusion_matrices", exist_ok=True)
    model_name = os.path.basename(model_path).replace(".h5", "")
    plt.savefig(f"results/confusion_matrices/{model_name}_confusion_matrix.png")
    plt.show()

# ----------------------------------------- #
# 🖼️ 5 Görsel Üzerinden Tahmin Gösterimi    #
# ----------------------------------------- #
def predict_examples(model_path, X_val, y_val, count=5):
    model = load_model(model_path)
    indices = np.random.choice(len(X_val), count, replace=False)
    for i in indices:
        img = X_val[i]
        true_label = y_val[i]
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        pred_label = 1 if pred > 0.5 else 0

        plt.imshow(img)
        plt.title(f"Gerçek: {'Maskeli' if true_label==0 else 'Maskesiz'} | Tahmin: {'Maskeli' if pred_label==0 else 'Maskesiz'}")
        plt.axis("off")
        plt.show()
