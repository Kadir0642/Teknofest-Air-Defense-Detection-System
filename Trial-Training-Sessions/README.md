Project Summary: 

🚀 Teknofest Air Defense System An autonomous system module developed for the target 
   detection category, detecting balloon targets with 87-95% accuracy.

🛠 Technologies Used

Python 3.9

OpenCV (Görüntü İşleme)

Ultralytics (YOLOv8 & Tracking)


## 📊 Model Performance Benchmarks

Two distinct models were trained to address different air defense scenarios. 
Following optimization efforts, **Model v2 (2-Class)** was selected for the final deployed system.

| Metric | Model v1 (Single-Class) | Model v2 (Multi-Class - Final) | Impact / Analysis |
| :--- | :--- | :--- | :--- |
| **Target Classes** | 1 (Balloon) | **2 (Balloon + [Other Class Name])** | Operational scope expanded. |
| **Training Duration** | 15 Epochs | **50 Epochs** | Achieved more stable learning convergence. |
| **mAP @50** | 0.995 | **0.994** | High accuracy maintained despite increased complexity. |
| **mAP @50-95** | 0.920 | **0.940** | 📈 **+2% Increase:** Localization precision significantly improved. |
| **Loss (Val Box)** | ~0.35 | **~0.29** | Lower error rate achieved in validation. |
| **Verdict** | R&D / Proof of Concept | ✅ **Production Ready** | Selected due to superior precision and robustness. |
