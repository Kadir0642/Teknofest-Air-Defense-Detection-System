import cv2
import numpy as np

# 1. fotoğraftaki araçlardan birini (ROI) kırpıp kaydettiğini varsayıyoruz.
# Örneğin sadece uçağın olduğu kısmı 'ucak_roi.jpg' olarak kaydet ve buraya yolunu gir.
image = cv2.imread(r"C:\\Users\\LENOVO\\Image_processing\\HSS-Detect-Track\\fuze-test.png")
output = image.copy()

# 2. Görüntüyü gri tonlamaya çevir (Renk bağımlılığını yok et)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Pürüzleri ve gereksiz detayları yok etmek için bulanıklaştır
gray_blurred = cv2.medianBlur(gray, 5)

# 4. HoughCircles ile yuvarlak şekilleri ara
# Not: param1 ve param2 değerleri görüntünün ışığına göre hassas ayar gerektirir.
circles = cv2.HoughCircles(gray_blurred, 
                           cv2.HOUGH_GRADIENT, 
                           dp=1, 
                           minDist=20,
                           param1=50, 
                           param2=30, # Bu değeri düşürürsen daha çok çember bulur, artırırsan daha seçici olur
                           minRadius=10, 
                           maxRadius=100)

# 5. Çember bulunduysa merkeze kilitlen
if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # En belirgin çemberi alıyoruz (zaten ROI içinde tek balon var)
    for i in circles[0, :1]:
        # Çemberin dış hatlarını çiz (Yeşil)
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Çemberin tam merkezini (cx, cy) çiz (Kırmızı)
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        print(f"Hedefe Kilitlenildi! Merkez Koordinatları: cx={i[0]}, cy={i[1]}")

# Sonucu ekranda göster
cv2.imshow("1. Asama - Hedef Kilidi", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
