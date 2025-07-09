import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, ImageOps

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
import cv2

# Model dosya yolları
MODEL_PATH = "model.keras"                 # Eğitilmiş modelin yolu
MODEL_IMG_PATH = "model.png"               # Model diyagram görseli
model = tf.keras.models.load_model(MODEL_PATH)

# Tuval (çizim alanı) boyutu
WIDTH = HEIGHT = 280

# Tahmin edilen çizimlerin kaydedileceği klasör
SAVE_DIR = "saved_digits"
os.makedirs(SAVE_DIR, exist_ok=True)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 El Yazısı Rakam Tanıma")

        # Sol panel: tuval ve butonlar
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, padx=10)

        # Tuval (Canvas): kullanıcı burada rakam çizer
        self.canvas = tk.Canvas(self.left_frame, width=WIDTH, height=HEIGHT, bg='white', cursor='cross')
        self.canvas.pack()

        # Butonlar: Tahmin Et / Temizle
        self.button_frame = tk.Frame(self.left_frame)
        self.button_frame.pack(pady=5)
        tk.Button(self.button_frame, text="Tahmin Et", command=self.predict_digit).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Temizle", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)

        # Tahmin sonucu etiketi
        self.label = tk.Label(self.left_frame, text="", font=("Arial", 16), fg="blue")
        self.label.pack(pady=5)

        # Tuvale çizim yapılabilmesi için PIL resmi
        self.image = Image.new("L", (WIDTH, HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

        # Sağ panel: geçmiş ve model görseli
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, padx=10)

        # Tahmin geçmişi listesi
        tk.Label(self.right_frame, text="📋 Tahmin Geçmişi:", font=("Arial", 12)).pack()
        self.history_list = tk.Listbox(self.right_frame, width=30)
        self.history_list.pack(pady=5)

        # Model diyagram görseli (varsa gösterilir)
        tk.Label(self.right_frame, text="🧩 Model Mimarisi:", font=("Arial", 12)).pack(pady=(10, 0))
        if os.path.exists(MODEL_IMG_PATH):
            model_img = Image.open(MODEL_IMG_PATH).resize((240, 300))
            self.model_photo = ImageTk.PhotoImage(model_img)
            tk.Label(self.right_frame, image=self.model_photo).pack()
        else:
            tk.Label(self.right_frame, text="Model görseli bulunamadı.").pack()

    def draw_on_canvas(self, event):
        # Fareyle çizim yapılırken tuval ve PIL resminde siyah daire çiz
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def clear_canvas(self):
        # Tuvali temizle
        self.canvas.delete("all")
        self.image = Image.new("L", (WIDTH, HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="")

    def predict_digit(self):
        # Çizimi .png dosyası olarak kaydet
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_DIR}/digit_{timestamp}.png"
        self.image.save(filename)

          # Görseli 28x28'e küçült, ters çevir
        img = self.image.resize((28, 28)).convert('L')
        img = ImageOps.invert(img)

        # PIL -> NumPy
        img_array = np.array(img).astype(np.uint8)

        # Gaussian Blur
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

        # Median Blur
        img_array = cv2.medianBlur(img_array, 3)

        # Adaptive Thresholding
        img_array = cv2.adaptiveThreshold(img_array, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

        # Normalize ve reshape
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Tahmin yap
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Sonucu etikete yaz
        self.label.config(text=f"Sonuç: {predicted_class}  (%{confidence*100:.1f})")

        # Geçmiş listesine ekle
        self.history_list.insert(tk.END, f"{timestamp[-6:]} → {predicted_class} (%{confidence*100:.1f})")

        # Tüm skorları bar grafikte göster
        plt.bar(range(10), predictions[0])
        plt.title(f"Tahmin: {predicted_class}, Güven: %{confidence*100:.1f}")
        plt.xlabel("Rakam")
        plt.ylabel("Skor")
        plt.xticks(range(10))
        plt.show()

# Uygulama başlat
root = tk.Tk()
app = App(root)
root.mainloop()
