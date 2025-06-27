import pytesseract
import cv2
import os
import re
import pandas as pd
from tkinter import Tk, Button, Label, filedialog, Canvas, messagebox
from PIL import Image, ImageTk

# Tesseract yolu (güncelleme gerekirse elle değiştir)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
EXCEL_PATH = os.path.join(os.getcwd(), "fatura_kayitlari.xlsx")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    thresh = cv2.adaptiveThreshold(
        denoised, 100,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    noise_removal = cv2.medianBlur(thresh, 5)
    return noise_removal

import re

def extract_info(text):
    import re

    # Tüm TL geçen değerleri ara
    amount_matches = re.findall(r"(\d[\d.,]{2,})\s*TL", text)
    amount = None

    if amount_matches:
        for raw in amount_matches:
            cleaned = raw.replace(" ", "")
            # Eğer hem nokta hem virgül varsa, virgül ondalık olarak kabul edilir
            if "," in cleaned and "." in cleaned:
                cleaned = cleaned.replace(".", "").replace(",", ".")
            elif "," in cleaned:
                cleaned = cleaned.replace(",", ".")
            # Sadece nokta varsa direkt geç
            try:
                float_val = float(cleaned)
                if float_val > 0:
                    amount = f"{float_val:.2f}"
                    break
            except:
                continue

    # Tarih formatı: 06.08.2012 veya 06082012
    date_match = re.search(r"(\d{2}[./-]?\d{2}[./-]?\d{4})", text)
    date = None
    if date_match:
        raw_date = date_match.group(1).replace("-", "").replace("/", "").replace(".", "")
        if len(raw_date) == 8:
            date = f"{raw_date[:2]}.{raw_date[2:4]}.{raw_date[4:]}"
    
    return amount, date

def save_to_excel(filename, amount, date):
    import openpyxl
    df_new = pd.DataFrame({
        "Görsel Adı": [filename],
        "Fatura Tutarı (TL)": [amount],
        "Son Ödeme Tarihi": [date]
    })

    if os.path.exists(EXCEL_PATH):
        # Var olan dosyaya ekle
        with pd.ExcelWriter(EXCEL_PATH, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df_new.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        # Yeni dosya oluştur
        df_new.to_excel(EXCEL_PATH, index=False)

class FaturaOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fatura OCR Uygulaması")
        self.image_path = None

        self.label = Label(root, text="Fatura Görseli Seçin", font=("Arial", 12))
        self.label.pack(pady=5)

        self.canvas = Canvas(root, width=500, height=400, bg="white")
        self.canvas.pack()

        self.select_button = Button(root, text="Görsel Seç", command=self.select_image)
        self.select_button.pack(pady=5)

        self.save_button = Button(root, text="Kaydet", command=self.process_image)
        self.save_button.pack(pady=5)

        self.clear_button = Button(root, text="Temizle", command=self.clear_image)
        self.clear_button.pack(pady=5)

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Görsel Dosyaları", "*.png *.jpg *.jpeg")])
        if path:
            self.image_path = path
            img = Image.open(path)
            img.thumbnail((500, 400))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(250, 200, image=self.tk_img)

    def process_image(self):
        if not self.image_path:
            messagebox.showwarning("Uyarı", "Lütfen bir görsel seçin.")
            return

        processed = preprocess_image(self.image_path)
        text = pytesseract.image_to_string(processed, config='--psm 6', lang='tur')
        print("==== OCR ÇIKTISI ====\n", text)

        amount, date = extract_info(text)
        if amount and date:
            save_to_excel(os.path.basename(self.image_path), amount, date)
            messagebox.showinfo("Başarılı", f"Tutar: {amount} TL\nTarih: {date}\nExcel'e kaydedildi.")
        else:
            messagebox.showerror("Hata", "Fatura bilgileri tespit edilemedi.")

    def clear_image(self):
        self.canvas.delete("all")
        self.image_path = None

if __name__ == "__main__":
    root = Tk()
    app = FaturaOCRApp(root)
    root.mainloop()
