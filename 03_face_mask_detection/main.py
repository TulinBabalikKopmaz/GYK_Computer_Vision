# main.py

from scripts.model_train import train_mobilenet, train_resnet, train_efficientnet, show_device_info
import os

def create_folders():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/loss_accuracy_graphs", exist_ok=True)

def main():
    print("📁 Gerekli klasörler oluşturuluyor...")
    create_folders()

    # Donanım bilgisi
    show_device_info()

    print("\n🔧 Eğitim için model seç:")
    print("1 - MobileNetV2")
    print("2 - ResNet50")
    print("3 - EfficientNetB0")
    choice = input("Seçiminizi girin (1/2/3): ").strip()

    if choice == '1':
        print("🚀 MobileNetV2 eğitimi başlıyor...")
        train_mobilenet()
        print("✅ MobileNetV2 eğitimi tamamlandı.\n")
    elif choice == '2':
        print("🚀 ResNet50 eğitimi başlıyor...")
        train_resnet()
        print("✅ ResNet50 eğitimi tamamlandı.\n")
    elif choice == '3':
        print("🚀 EfficientNetB0 eğitimi başlıyor...")
        train_efficientnet()
        print("✅ EfficientNetB0 eğitimi tamamlandı.\n")
    else:
        print("❌ Geçersiz seçim. Program sonlandırılıyor.")

if __name__ == "__main__":
    main()
