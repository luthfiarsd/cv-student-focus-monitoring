# cv-student-focus-monitoring
COMPUTER VISION - MONITORING FOKUS DAN ATENSI SISWA DALAM PEMBELAJARAN ONLINE &amp; OFFLINE

![Infografis](images/Infografis_Attentix.png)

Sistem deteksi fokus siswa berbasis computer vision menggunakan YOLOv11n. Digunakan untuk memantau atensi siswa selama pembelajaran daring secara real-time.

Fitur Utama:

- Deteksi kondisi: Normal, Distracted, Face Covered

- Live camera monitoring (Streamlit-based interface)

- Confidence threshold control

- Statistik sesi: jumlah terdistraksi, durasi, deteksi per menit

- Alert otomatis saat gangguan terdeteksi terus-menerus

- Timeline confidence visualisasi

Teknologi:

- YOLOv11n (ultralytics)

- Python, OpenCV

- Streamlit

Cara Menjalankan:

```
git clone https://github.com/namakamu/student-attention-monitor.git
cd student-attention-monitor
pip install -r requirements.txt
streamlit run app.py
```

Preview:
![Gambar1](images/gambar1.png)
![Gambar2](images/gambar2.png)
![Gambar3](images/gambar3.png)
![Gambar4](images/gambar4.png)
