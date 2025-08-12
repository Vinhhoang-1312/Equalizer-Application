# 🎵 AUDIO PROCESSING APP - FLOW DIAGRAM

## 📁 CẤU TRÚC THƯ MỤC VÀ LUỒNG CHẠY

```
xlth/
├── 🚀 ENTRY POINTS
│   ├── main.py                    ← ĐIỂM VÀO CHÍNH
│   ├── web_app.py                 ← WEB SERVER TRỰC TIẾP
│   └── test_app.py                ← TEST SUITE
│
├── 🧠 CORE PROCESSING
│   ├── advanced_audio_processor.py ← XỬ LÝ ÂM THANH CHÍNH
│   ├── audio_processor.py         ← XỬ LÝ CƠ BẢN
│   └── advanced_model_trainer.py  ← HUẤN LUYỆN MÔ HÌNH
│
├── 🌐 WEB INTERFACE
│   ├── templates/
│   │   └── index.html             ← GIAO DIỆN CHÍNH
│   └── static/
│       └── results/               ← FILES KẾT QUẢ
│
├── 🤖 AI MODELS
│   ├── models/
│   │   ├── advanced_genre_classifier.pkl    ← PHÂN LOẠI THỂ LOẠI
│   │   ├── advanced_noise_reducer.h5        ← GIẢM NHIỄU ML
│   │   ├── advanced_scaler.pkl              ← CHUẨN HÓA
│   │   ├── feature_scaler.pkl               ← CHUẨN HÓA FEATURES
│   │   ├── train_models.py                  ← HUẤN LUYỆN
│   │   └── create_simple_models.py          ← TẠO MODELS
│
├── 🎵 DATA & FILES
│   ├── data/
│   │   └── gtzan/                 ← DATASET TRAINING
│   │       ├── blues/             ← 50 files
│   │       ├── classical/         ← 50 files
│   │       ├── country/           ← 50 files
│   │       ├── disco/             ← 50 files
│   │       ├── hiphop/            ← 50 files
│   │       ├── jazz/              ← 50 files
│   │       ├── metal/             ← 50 files
│   │       ├── pop/               ← 50 files
│   │       ├── reggae/            ← 50 files
│   │       └── rock/              ← 50 files
│   ├── uploads/                   ← FILES NGƯỜI DÙNG UPLOAD
│   └── static/results/            ← FILES ĐÃ XỬ LÝ
│
└── 🔗 INTEGRATIONS
    └── spotify_integration.py     ← SPOTIFY API
```

## 🔄 LUỒNG CHẠY CHI TIẾT

### 1️⃣ **KHỞI ĐỘNG APP**
```
┌─────────────────┐
│   python main.py │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  check_deps()   │ ← Kiểm tra dependencies
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  run_web_app()  │ ← Chạy web server
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   web_app.py    │ ← Flask app
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ AdvancedAudio   │ ← Khởi tạo processor
│ Processor       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Load Models    │ ← Load AI models
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Server Ready   │ ← http://localhost:5000
└─────────────────┘
```

### 2️⃣ **LUỒNG XỬ LÝ FILE UPLOAD**
```
┌─────────────────┐
│   User Upload   │ ← Người dùng upload file
│   audio file    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  web_app.py     │ ← Nhận file
│  /upload route  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Save to        │ ← Lưu vào uploads/
│  uploads/       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ AdvancedAudio   │ ← Gọi processor
│ Processor       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Load Audio     │ ← librosa.load()
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Equalizer      │ ← 6-band equalizer
│  Processing     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Denoise        │ ← ML noise reduction
│  Processing     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Genre          │ ← AI classification
│  Classification │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Audio          │ ← Phân tích đặc tính
│  Analysis       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Save Results   │ ← Lưu vào static/results/
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Create         │ ← Tạo visualization
│  Visualization  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Return JSON    │ ← Trả về kết quả
│  Response       │
└─────────────────┘
```

### 3️⃣ **CÁC MODULE LIÊN KẾT**

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN ENTRY POINTS                        │
├─────────────────────────────────────────────────────────────┤
│  main.py ──┐                                                │
│            ├───► web_app.py ──┐                            │
│  test_app.py ──┘              ├───► advanced_audio_processor.py
│                                │                            │
│                                └───► spotify_integration.py │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    CORE PROCESSING                          │
├─────────────────────────────────────────────────────────────┤
│  advanced_audio_processor.py                                │
│  ├───► Load Models (models/*.pkl, *.h5)                    │
│  ├───► Process Audio (librosa)                              │
│  ├───► Apply Equalizer (6-band)                             │
│  ├───► Denoise (ML methods)                                 │
│  ├───► Genre Classification (AI)                            │
│  └───► Audio Analysis (features)                            │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA FLOW                                │
├─────────────────────────────────────────────────────────────┤
│  User Upload ──► uploads/ ──► Processing ──► static/results/│
│                                                             │
│  Training Data ──► data/gtzan/ ──► Model Training ──► models/│
└─────────────────────────────────────────────────────────────┘
```

### 4️⃣ **AI MODELS PIPELINE**

```
┌─────────────────┐
│  Input Audio    │ ← File âm thanh
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Feature        │ ← Trích xuất đặc trưng
│  Extraction     │   - MFCC
│                 │   - Spectral features
│                 │   - Rhythm features
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Feature        │ ← Chuẩn hóa features
│  Scaling        │   (feature_scaler.pkl)
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Genre          │ ← Phân loại thể loại
│  Classification │   (advanced_genre_classifier.pkl)
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Noise          │ ← Giảm nhiễu
│  Reduction      │   (advanced_noise_reducer.h5)
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Output         │ ← Kết quả cuối cùng
│  Results        │
└─────────────────┘
```

### 5️⃣ **WEB INTERFACE FLOW**

```
┌─────────────────┐
│  Browser        │ ← http://localhost:5000
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  templates/     │ ← Giao diện HTML
│  index.html     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  User Upload    │ ← Drag & drop file
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  AJAX Request   │ ← JavaScript
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Flask Route    │ ← /upload endpoint
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Processing     │ ← Audio processing
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  JSON Response  │ ← Kết quả
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Display        │ ← Hiển thị kết quả
│  Results        │
└─────────────────┘
```

## 🎯 **TÓM TẮT LUỒNG CHẠY**

1. **Khởi động:** `main.py` → `web_app.py` → Flask server
2. **Upload:** Browser → `templates/index.html` → `web_app.py/upload`
3. **Xử lý:** `advanced_audio_processor.py` → AI models → Kết quả
4. **Lưu trữ:** `uploads/` → Processing → `static/results/`
5. **Hiển thị:** JSON response → Browser → Visualization

## 🔧 **CÁC LỆNH CHẠY CHÍNH**

```bash
# Chạy web app
python web_app.py
# Hoặc
python main.py --web

# Test app
python test_app.py

# Huấn luyện models
python main.py --train

# Xử lý real-time
python main.py --realtime

# Demo
python main.py --demo
```

## 📊 **TRẠNG THÁI HIỆN TẠI**

✅ **HOẠT ĐỘNG TỐT:**
- Web server chạy tại port 5000
- File upload và xử lý hoạt động
- AI models load thành công
- Visualization tạo được
- Test suite pass 5/5

⚠️ **CẢNH BÁO:**
- Spotify credentials chưa cấu hình (không bắt buộc)
- Genre classification đôi khi trả về "Unknown"

**App đã sẵn sàng sử dụng!** 🎵

