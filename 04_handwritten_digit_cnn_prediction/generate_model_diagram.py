from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Yüklenmiş model dosyası
MODEL_PATH = "model.keras"
OUTPUT_IMG_PATH = "model.png"

# Modeli yükle
model = load_model(MODEL_PATH)

# Yapıyı çiz ve kaydet
plot_model(model, to_file=OUTPUT_IMG_PATH, show_shapes=True, show_layer_names=True)

print(f"✅ Model mimarisi '{OUTPUT_IMG_PATH}' olarak kaydedildi.")
