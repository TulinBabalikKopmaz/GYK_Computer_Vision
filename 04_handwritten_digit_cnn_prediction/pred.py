import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
model = tf.keras.models.load_model("OpenCV\04_handwritten_digit_cnn_prediction\model.keras")

model.summary()

from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image


plot_model(model, to_file="OpenCV\04_handwritten_digit_cnn_prediction\model.png", show_shapes=True, show_layer_names=True)


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizasyon => RGB kanallarının 0-255 aralığındansa 0-1 aralığına çekilmesi.
X_train = X_train / 255
X_test = X_test / 255 # 0-1
#

# CNN'in input formatı => (örnek sayısı, genişlik, yükseklik, kanal sayısı)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)  # Sebebini CNN'e geçtiğimizde konuşacağız.


test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")


# Kendi çizdiğiniz bir rakamı tahmin ettirelim.
# Örneğin: kendi_rakam.png dosyasını kullanıyoruz.
img_path = "gyk_computer_vision/cnn/number-157.png"  # Dosya adını kendi resmine göre değiştir.

# Orijinal resmi yükle
orig_img = image.load_img(img_path, color_mode="grayscale")
orig_img_array = image.img_to_array(orig_img).astype(np.uint8)

# Görüntüyü yükle, gri tonlamaya çevir, 28x28'e yeniden boyutlandır
img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
img_array = image.img_to_array(img)

# Görüntüyü yeniden boyutlandır
resized_img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
resized_img_array = image.img_to_array(resized_img).astype(np.uint8)

# Normalizasyon ve model için hazırlık
img_array = resized_img_array / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Tahmin yap
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

# Sonuçları konsolda göster
print(f"Tahmin edilen rakam: {predicted_class}")
print(f"Güven skoru: {confidence:.2f}")
print("Tüm skorlar:", predictions[0])

# Orijinal resmi ve skorları matplotlib ile göster
plt.figure(figsize=(15, 5))

# Orijinal resmi göster
plt.subplot(1, 3, 1)
plt.imshow(orig_img_array[:, :, 0], cmap="gray")
plt.title("Orijinal Resim")
plt.axis("off")

# Yeniden boyutlandırılmış resmi göster
plt.subplot(1, 3, 2)
plt.imshow(resized_img_array[:, :, 0], cmap="gray")
plt.title("28x28 Boyutunda")
plt.axis("off")

# Güven skorlarını göster
plt.subplot(1, 3, 3)
plt.bar(range(10), predictions[0])
plt.title(f"Tahmin Sonucu: {predicted_class}\nGüven: {confidence:.2f}")
plt.xlabel("Rakam")
plt.ylabel("Güven Skoru")
plt.xticks(range(10))

plt.tight_layout()
plt.show()