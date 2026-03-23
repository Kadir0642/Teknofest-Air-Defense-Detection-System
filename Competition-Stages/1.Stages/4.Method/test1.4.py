import cv2
import numpy as np

def saf_test_sistemi():
    # Görüntüyü yükle (Sırayla tüm İHA, Jet ve Füze resimlerinin yolunu buraya gir)
    frame = cv2.imread(r"C:\\Users\\LENOVO\\Image_processing\\HSS-Detect-Track\\fuze-test.png")
    if frame is None:
        print("Görüntü okunamadı! Dosya yolunu kontrol et.")
        return

    sonuc_ekrani = frame.copy()

    # 1. Görüntüyü HSV'ye çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Turuncu renk sınırları (Işık değişimlerine göre bu değerleri hassaslaştırmak gerekebilir)
    lower_color = np.array([0, 120, 120]) # Önceki [5,150,150] idi, biraz daha genişlettik
    upper_color = np.array([30, 255, 255]) # Önceki [20,255,255] idi, biraz daha genişlettik
    
    # 3. Maskeyi oluştur
    maske = cv2.inRange(hsv, lower_color, upper_color)
    
    # 4. Gürültüyü azaltmak için maskeyi temizle (Morphological Operations)
    # Bu adım, arkaplandaki küçük turuncu yansımaları yok eder
    kernel = np.ones((5, 5), np.uint8)
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel)
    
    # 5. Konturları bul
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if konturlar:
        # Ekranda birden fazla turuncu kütle olabilir (örn: yansımalar), biz en büyüğünü alıyoruz
        en_buyuk_kontur = max(konturlar, key=cv2.contourArea)
        
        # Bu kütlenin "Ağırlık Merkezini" (Moments) hesapla
        M = cv2.moments(en_buyuk_kontur)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Hedef noktasını ana ekrana çiz (Kırmızı)
            cv2.circle(sonuc_ekrani, (cx, cy), 10, (0, 0, 255), -1)
            
            # Kilitlenme bilgisini ekrana yazdır
            cv2.putText(sonuc_ekrani, f"TAKIPTE! X: {cx}, Y: {cy}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            print(f"HEDEF TAKIPTE! -> X: {cx}, Y: {cy}")

    # Hem filtremizin ne gördüğünü hem de sonucu ekrana yazdıralım
    cv2.imshow("Aşama 1 Saf Test - Maske", maske)
    cv2.imshow("Aşama 1 Saf Test - Sonuç", sonuc_ekrani)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

saf_test_sistemi()
