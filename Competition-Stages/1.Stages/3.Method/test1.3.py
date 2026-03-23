import cv2
import numpy as np
from ultralytics import YOLO

# 1. Hazır YOLOv8 Nano modelini yükle (İlk çalışmada internetten modeli indirecektir)
model = YOLO("yolov8n.pt")

def gercek_hedef_takip():
    # Görüntüyü yükle (Buraya kendi jet veya helikopter resminin yolunu gir)
    frame = cv2.imread(r"C:\\Users\\LENOVO\\Image_processing\\HSS-Detect-Track\\heli-test.png")
    if frame is None:
        print("Görüntü okunamadı! Dosya yolunu kontrol et.")
        return

    sonuc_ekrani = frame.copy()

    # 2. Gerçek YOLO ile tespit yap
    results = model(frame)

    # Bulunan tüm araçları döngüye al
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Sınıf ID'sini ve ismini al
            cls_id = int(box.cls[0])
            sinif_adi = model.names[cls_id] # Örn: 'airplane', 'car', 'person'

            # Eğer bulunan nesne bir uçaksa (Helikopteri de genelde airplane sayar)
            if sinif_adi == "airplane":
                # Kutu koordinatlarını al
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ana ekranda YOLO kutusunu çiz (Mavi)
                cv2.rectangle(sonuc_ekrani, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(sonuc_ekrani, f"YOLO: {sinif_adi}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 3. ROI (İlgi Alanı) Kesimi
                roi = frame[y1:y2, x1:x2]
                
                # Güvenlik önlemi: Eğer kesilen alan herhangi bir sebepten boşsa hatayı önlemek için atla
                if roi.size == 0:
                    continue

                # 4. Mikro Kilitlenme (HSV + Moments)
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Turuncu renk sınırları
                lower_color = np.array([5, 150, 150])
                upper_color = np.array([20, 255, 255])
                
                maske = cv2.inRange(hsv_roi, lower_color, upper_color)
                konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if konturlar:
                    en_buyuk_kontur = max(konturlar, key=cv2.contourArea)
                    M = cv2.moments(en_buyuk_kontur)
                    
                    if M["m00"] != 0:
                        cx_roi = int(M["m10"] / M["m00"])
                        cy_roi = int(M["m01"] / M["m00"])
                        
                        # Yerel merkezi, Ana ekran koordinatlarına çevir
                        cx_genel = x1 + cx_roi
                        cy_genel = y1 + cy_roi
                        
                        # Hedef noktasını çiz (Kırmızı)
                        cv2.circle(sonuc_ekrani, (cx_genel, cy_genel), 6, (0, 0, 255), -1)
                        
                        # Servo motorlara gidecek veriyi ekrana yazdır
                        hedef_verisi = f"HEDEF -> X: {cx_genel}, Y: {cy_genel}"
                        cv2.putText(sonuc_ekrani, hedef_verisi, (20, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        print(hedef_verisi)

    cv2.imshow("Gercek Yapay Zeka - Hava Savunma", sonuc_ekrani)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

gercek_hedef_takip()
