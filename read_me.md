# Laporan Proyek Machine Learning  - Rahayu Kartika Sari
## Domain proyek
![PCOS](https://img-cdn.medkomtek.com/J_Q2GT78W2CZQO8GFX8s0L_AEUI=/0x0/smart/filters:quality(100):format(webp)/article/nv-LVYH1oNKCXGkGGZR86/original/038160000_1569585751-7-Hal-tentang-PCOS-yang-Perlu-Diketahui-Wanita-By-CHEN-I-CHUN-Shutterstock.jpg)

Polycystic Ovary Syndrome (PCOS) adalah gangguan hormonal yang umum terjadi pada wanita usia reproduktif. PCOS dapat menyebabkan berbagai gejala, termasuk tidak teraturnya siklus menstruasi, infertilitas, jerawat, peningkatan berat badan, dan gejala terkait metabolisme. Menyebabkan dampak signifikan baik secara fisik maupun psikologis, penting untuk mengidentifikasi PCOS secara dini agar pengobatan yang tepat dapat diberikan.

Seiring dengan kemajuan dalam teknologi dan kemampuan machine learning, pemanfaatan algoritma untuk mendiagnosis PCOS dapat memberikan solusi yang cepat dan akurat. Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat membantu dalam mengidentifikasi PCOS berdasarkan dataset medis yang relevan. Dengan adanya model ini, diharapkan akan ada peningkatan dalam deteksi awal dan manajemen kesehatan wanita.

Selain itu, proyek ini memperhitungkan berbagai faktor risiko yang berkontribusi terhadap PCOS, seperti indeks massa tubuh (BMI), kadar insulin, dan riwayat kesehatan keluarga, untuk meningkatkan akurasi diagnosis.

Referensi: [Sindrom ovarium polikistik: Penyebab, gejala, patofisiologi, dan pengobatan](https://www.sciencedirect.com/science/article/abs/pii/S2451847623000040)

### Mengapa dan Bagaimana Masalah Ini Dapat Dipecahkan
Pemahaman yang lebih baik terhadap faktor-faktor yang menyebabkan PCOS dan pengaplikasian algoritma pembelajaran mesin dapat meningkatkan akurasi diagnosis. Penggunaan model machine learning seperti regresi logistik, pohon keputusan, dan jaringan saraf dapat membantu mengidentifikasi pola dalam data medis yang sulit dicermati oleh manusia.

Proyek ini diharapkan dapat memberikan kontribusi dalam bidang kesehatan wanita dan memberikan alat yang berguna bagi profesional medis dalam proses diagnosis PCOS.

## Business Understanding
### Problem Statements
- Gejala PCOS sangat bervariasi antar individu, sehingga sulit untuk mengidentifikasi pola yang jelas hanya berdasarkan pengamatan klinis. Hal ini menyulitkan dokter dalam membuat diagnosis yang akurat dan tepat waktu.
- Di beberapa daerah, terutama di daerah pedesaan, akses ke spesialis endokrinologi atau ginekologi yang berpengalaman dalam mendiagnosis PCOS mungkin terbatas. Hal ini menyebabkan kesenjangan dalam perawatan kesehatan bagi wanita yang berpotensi memiliki PCOS.
- Metode diagnosis PCOS saat ini seringkali memakan waktu dan bergantung pada interpretasi subjektif dari berbagai gejala dan tes medis. Hal ini dapat menyebabkan keterlambatan diagnosis dan penanganan yang tepat.
- Terdapat tantangan dalam mengidentifikasi PCOS berdasarkan data medis yang tidak lengkap atau tidak terstruktur dengan baik.

### Goals
- Menggunakan machine learning untuk mendeteksi PCOS lebih cepat dan dengan akurasi tinggi, memungkinkan diagnosa lebih dini dan pengobatan yang lebih tepat waktu.
- Memanfaatkan kemampuan machine learning untuk menemukan pola dan hubungan antar gejala yang mungkin terlewatkan oleh manusia, sehingga meningkatkan pemahaman tentang variabilitas PCOS dan membantu dalam diagnosis yang lebih personal.
- Mengembangkan alat diagnosis berbasis machine learning yang mudah digunakan dan dapat diakses oleh praktisi kesehatan umum atau bahkan pasien secara mandiri, sehingga memperluas akses ke diagnosis PCOS di daerah dengan keterbatasan sumber daya.
- Membersihkan dan mengolah data medis yang ada, serta menggunakan teknik machine learning untuk menangani ketidaklengkapan atau ketidakteraturan dalam data.

### Solution statements
- Menggunakan algoritma regresi logistik untuk memprediksi kemungkinan seseorang mengidap PCOS berdasarkan data faktor risiko seperti BMI, kadar insulin, dan riwayat kesehatan keluarga.
- Melakukan hyperparameter tuning pada model klasifikasi yang dipilih untuk meningkatkan performanya secara optimal. Proses ini akan melibatkan penggunaan teknik seperti Grid Search atau Random Search untuk menemukan kombinasi hyperparameter terbaik yang menghasilkan kinerja model tertinggi.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah [PCOS Diagnosis Dataset](https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset) yang berisi data medis terkait PCOS dari Kaggle, yang berisi data terkait pasien dengan Polycystic Ovary Syndrome (PCOS). Dataset ini terdiri dari 1000 entri yang mewakili pasien, dengan lima fitur utama yang sering digunakan dalam diagnosis dan penilaian risiko PCOS.

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- **Age** (years): Usia pasien dalam tahun, dengan rentang dari 18 hingga 45 tahun.
- **BMI** (kg/m²): Indeks Massa Tubuh (Body Mass Index) pasien, yang merupakan ukuran lemak tubuh berdasarkan tinggi dan berat, dengan rentang dari 18 hingga 35.
- **Menstrual Irregularity** (binary): Indikator biner yang menunjukkan apakah pasien mengalami siklus menstruasi yang tidak teratur (0 = Tidak, 1 = Ya).
- **Testosterone Level** (ng/dL): Tingkat testosteron dalam darah pasien, indikator hormonal penting dari PCOS, dengan rentang dari 20 hingga 100 ng/dL.
- **Antral Follicle Count**: Jumlah folikel antral yang terdeteksi selama ultrasonografi, dengan rentang dari 5 hingga 30, yang membantu dalam menilai cadangan ovarium dan keberadaan PCOS.
- **PCOS Diagnosis** (binary): Indikator biner yang menunjukkan apakah pasien telah didiagnosis dengan PCOS (0 = Tidak, 1 = Ya), berdasarkan kombinasi faktor risiko seperti BMI tinggi, kadar testosteron, ketidakteraturan menstruasi, dan jumlah folikel antral. **(Variabel Target)**

### Tahapan Exploratory Data Analysis (EDA):
- Distribusi Data: Visualisasi distribusi masing-masing fitur untuk memahami variasi dan pola dalam dataset.


- Missing Data: Memeriksa keberadaan data yang hilang dan menangani masalah tersebut melalui imputasi atau penghapusan data yang tidak lengkap. Pada data yang digunakan, tidak ditemukan adanya missing values.

- Outliers: Deteksi outlier dilakukan terutama pada fitur numerik seperti BMI dan testosteron, yang dapat mempengaruhi akurasi model. Pada data yang digunakan, tidak ditemukan adanya outliers.
- Korelasi antar Fitur: Menggunakan heatmap untuk melihat korelasi antara fitur-fitur yang ada, seperti apakah ada hubungan signifikan. 

## Data Preparation
Data Preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan machine learning bersih, konsisten, dan siap untuk dianalisis. Tanpa tahapan ini, model mungkin tidak akan bekerja dengan baik, baik karena masalah dalam kualitas data (seperti data yang hilang atau outlier) atau karena ketidaksesuaian skala antar fitur yang dapat mempengaruhi hasil pelatihan.
### Proses Data Preparation:
1. **Handling Unbalance Data**
    
    Masalah data yang tidak seimbang sering kali muncul dalam dataset medis seperti ini, di mana jumlah kasus positif (misalnya, pasien yang didiagnosis dengan PCOS) jauh lebih sedikit dibandingkan dengan kasus negatif (pasien yang tidak didiagnosis dengan PCOS). Dalam situasi ini, model machine learning mungkin akan cenderung memprediksi kelas mayoritas (negatif) dan gagal mendeteksi kelas minoritas (positif), yang dapat mengurangi efektivitas model.
    
    **Langkah yang Diambil**:
    Mengimplementasikan SMOTE. SMOTE adalah teknik oversampling yang menghasilkan contoh sintetis dari kelas minoritas dengan cara membuat titik data baru di ruang fitur berdasarkan contoh yang ada. Teknik ini digunakan untuk meningkatkan jumlah sampel dari kelas yang lebih sedikit dengan menghasilkan data sintetis yang mendekati distribusi data asli.

3. **Splitting Data (Train-Test Split)**
    
     Tanpa pemisahan data untuk pelatihan dan pengujian, model dapat mengingat (overfit) data pelatihan, yang menyebabkan penurunan kinerja pada data yang belum pernah dilihat (data uji). Oleh karena itu, sangat penting untuk memisahkan dataset menjadi dua bagian: satu untuk melatih model dan satu untuk menguji seberapa baik model tersebut bekerja pada data baru. Dataset dibagi menjadi 80% data pelatihan dan 20% data pengujian atau validation data untuk lebih menilai kinerja model. Pembagian ini membantu untuk mendapatkan gambaran yang lebih akurat mengenai performa model pada data yang belum pernah dilihat sebelumnya.

4. **Normalisasi**
    
    Masalah Skala Fitur: Beberapa fitur dalam dataset, seperti BMI dan Testosterone Level, memiliki skala yang sangat berbeda. Misalnya, nilai BMI bisa berkisar antara 18 hingga 35, sementara Testosterone Level bisa berkisar antara 20 hingga 100. Model yang sensitif terhadap skala fitur (seperti regresi logistik) dapat terpengaruh jika fitur dengan skala yang lebih besar mendominasi proses pelatihan. Untuk menghindari dominasi fitur dengan skala lebih besar, fitur numerik harus dinormalisasi atau distandarisasi agar mereka berada dalam skala yang sama.
    
    **Langkah yang Diambil**:
    Menggunakan Min-Max Scaling pada fitur numerik seperti BMI, Testosterone Level, dan Antral Follicle Count agar nilai-nilai tersebut berada dalam rentang yang seragam (misalnya antara 0 dan 1 untuk Min-Max Scaling).

## Modeling
Model machine learning yang digunakan dalam proyek ini adalah model klasifikasi, karena tujuan utamanya adalah untuk memprediksi apakah seorang pasien memiliki PCOS atau tidak. Algoritma yang akan dieksplorasi meliputi:
1. **Regresi Logistik**
    
    Algoritma ini digunakan untuk klasifikasi biner (PCOS: 1 atau Tidak PCOS: 0). Model ini bekerja dengan menghitung probabilitas suatu kelas (0 atau 1) berdasarkan fitur-fitur yang ada, kemudian menggunakan fungsi sigmoid untuk mengubah probabilitas tersebut menjadi prediksi biner.
    - Kelebihan: Mudah untuk diinterpretasikan, cocok untuk dataset dengan hubungan linier antara fitur dan target, dan cepat dalam pelatihan.
    - Kekurangan: Hanya efektif pada hubungan linier antar fitur. Rentan terhadap overfitting jika terdapat banyak fitur yang tidak relevan.

2. **Decision Tree**

    Digunakan untuk klasifikasi dan memahami hubungan antara berbagai fitur, membangun aturan bercabang untuk keputusan yang lebih baik..Model ini digunakan untuk memahami hubungan antara fitur dan target secara visual dan eksplisit.
    - Kelebihan: Mudah diinterpretasikan dan tidak memerlukan normalisasi data.
    - Kekurangan: Rentan terhadap overfitting, terutama jika pohon terlalu dalam, Sensitif terhadap perubahan kecil dalam data (fluktuasi).
    
3. **Random Forest**
    
    Algoritma ensemble yang menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi. Setiap pohon dibangun berdasarkan subset acak dari data dan fitur.
    - Kelebihan: Lebih akurat dibandingkan dengan pohon keputusan tunggal dan mengurangi risiko overfitting. Kuat terhadap data yang hilang dan tidak memerlukan normalisasi fitur.
    - Kekurangan: Lebih kompleks dan memerlukan waktu lebih lama dalam pelatihan. Sulit untuk diinterpretasikan dibandingkan dengan pohon keputusan tunggal.
4. **XGBoost** 
    
    XGBoost adalah algoritma boosting yang menggunakan banyak model yang lebih lemah (pohon keputusan) dan menggabungkannya untuk meningkatkan performa. 
    - Kelebihan: Sangat efisien dan sering kali memberikan hasil terbaik dalam kompetisi machine learning. Mampu menangani data yang besar dan kompleks. Memiliki kontrol yang lebih baik terhadap overfitting melalui regularisasi.
    - Kekurangan: Lebih kompleks dan memerlukan waktu lebih lama dalam pelatihan. Memerlukan lebih banyak perhatian pada pemilihan hyperparameter.

### Hyperparameter Tuning
Setiap model mengalami proses hyperparameter tuning untuk meningkatkan performa. Teknik yang digunakan untuk tuning ini adalah Grid Search. Grid Search mencoba semua kombinasi hyperparameter yang mungkin. Proses ini dilakukan untuk menemukan hyperparameter yang menghasilkan model terbaik.


## Evaluation
Setelah melakukan pelatihan model menggunakan dataset yang sudah dipersiapkan, evaluasi dilakukan untuk menilai kinerja model-model yang telah dibangun. Evaluasi menggunakan beberapa metrik yang penting untuk masalah klasifikasi biner ini.

### Metrik Evaluasi yang Digunakan:
1. **Akurasi**
    
    Metrik yang paling sederhana dan umum digunakan. Ini mengukur proporsi prediksi yang benar dari total prediksi.
    
    Rumus: Akurasi = (TP + TN) / (TP + TN + FP + FN)
    
    Keterangan:
    TP (True Positive): Prediksi positif yang benar.
    TN (True Negative): Prediksi negatif yang benar.
    FP (False Positive): Prediksi positif yang salah.
    FN (False Negative): Prediksi negatif yang salah.
2. **Precision**
    
    Presisi mengukur proporsi prediksi positif yang benar dari total prediksi positif.
    
    Rumus: Presisi = TP / (TP + FP)
    
    Presisi penting ketika kesalahan prediksi positif memiliki konsekuensi yang tinggi.
3. **Recall** (Sensitivitas)
    
    Recall mengukur proporsi kasus positif yang benar-benar teridentifikasi dari total kasus positif aktual.

    Rumus: Recall = TP / (TP + FN)
    
    Recall penting ketika penting untuk menghindari kesalahan prediksi negatif.
4. **F1-Score**

    F1-score adalah rata-rata harmonik dari presisi dan recall. Ini memberikan keseimbangan antara kedua metrik tersebut.
    
    Rumus: F1-score = 2 * (Presisi * Recall) / (Presisi + Recall)
    
    F1-score berguna ketika Anda ingin menyeimbangkan presisi dan recall, terutama dalam kasus dataset yang tidak seimbang.
5. **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve)

    ROC-AUC mengukur kemampuan model untuk membedakan antara kelas-kelas yang berbeda. Semakin tinggi nilai AUC, semakin baik kemampuan model.
    ROC-AUC berguna untuk mengevaluasi model klasifikasi biner.

### Hasil Evaluasi:
1. Logistic Regression:
    - Akurasi: 0.90 (Test)
    - Precision: 0.89 (Test)
    - Recall: 0.85 (Test)
    - F1-score: 0.80 (Test)
    - ROC-AUC: 0.92 (Test)
2. Decision Tree:
    - Akurasi: 1.0 (Test)
    - Precision: 1.0 (Test)
    - Recall: 1.0 (Test)
    - F1-score: 1.0 (Test)
    - ROC-AUC: 1.0 (Test)

3. Random Forest:
    - Akurasi: 1.0 (Test)
    - Precision: 1.0 (Test)
    - Recall: 1.0 (Test)
    - F1-score: 1.0 (Test)
    - ROC-AUC: 1.0 (Test)

4. XGBoost:
    - Akurasi: 1.0 (Test)
    - Precision: 1.0 (Test)
    - Recall: 1.0 (Test)
    - F1-score: 1.0 (Test)
    - ROC-AUC: 1.0 (Test)

Berdasarkan hasil evaluasi, model **XGBoost** memberikan hasil terbaik dengan 100% di semua metrik (Akurasi, Precision, Recall, F1-Score, dan ROC AUC). Oleh karena itu, XGBoost dipilih sebagai model terbaik untuk digunakan dalam mengidentifikasi PCOS. Keunggulannya dalam menangani data yang kompleks dan tidak seimbang menjadikannya pilihan yang tepat, terutama di bidang medis yang memerlukan akurasi tinggi dan kemampuan untuk menggeneralisasi. 

Namun, perlu diwaspadai potensi overfitting pada model ini.