import cv2
import numpy as np


image = cv2.imread(r"C:\\Users\\LENOVO\\Image_processing\\HSS-Detect-Track\\jet-test.png")
output = image.copy()

# 1. Görüntüyü BGR'dan HSV renk uzayına çevir (Işık değişimlerine çok daha dayanıklıdır)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 2. Turuncu renk için alt ve üst sınırları belirle 
# (Yarışmada hedef mavi olursa bu değerleri maviye göre değiştireceğiz)
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([20, 255, 255])

# 3. Sadece turuncu olan yerleri beyaz, gerisini siyah yapan bir maske (filtre) oluştur
mask = cv2.inRange(hsv, lower_orange, upper_orange)

# 4. Maskedeki beyaz bölgelerin sınırlarını (konturlarını) bul
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Ekranda birden fazla turuncu yansıma olabilir, biz en büyük alana sahip olanı alıyoruz
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 5. Bu alanın "Ağırlık Merkezini" (Moments) hesapla
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Hedefin etrafını sar (Yeşil) ve tam merkezine kilitlen (Kırmızı)
        cv2.drawContours(output, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
        
        print(f"Hedefe Kilitlenildi! Merkez Koordinatları: cx={cx}, cy={cy}")

# Hem filtremizin (maskenin) ne gördüğünü hem de sonucu ekrana yazdıralım
cv2.imshow("Renk Filtresi (Maske)", mask)
cv2.imshow("Hedef Kilidi", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
