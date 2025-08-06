# Hướng dẫn Triển khai Dự án Audio Processing

## Yêu cầu Hệ thống

- Python 3.8 hoặc cao hơn
- pip (Python package manager)
- Git

## Bước 1: Clone Repository

```bash
git clone <repository-url>
cd xlth

# Hoặc download ZIP từ GitHub và giải nén
```

## Bước 2: Tạo Virtual Environment (Tùy chọn)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Hoặc có thể cài đặt trực tiếp vào global environment
```

## Bước 3: Cài đặt Dependencies

```bash
# Cài đặt dependencies cơ bản
pip install -r requirements.txt

# Nếu cần tính năng nâng cao
pip install -r requirements_advanced.txt

# Hoặc cài đặt tất cả cùng lúc
pip install -r requirements.txt -r requirements_advanced.txt
```

## Bước 4: Tạo Thư mục Cần thiết

```bash
# Tạo các thư mục cho upload và kết quả (nếu chưa có)
mkdir -p uploads
mkdir -p static/results
mkdir -p data
mkdir -p models
mkdir -p logs
```

## Bước 5: Cấu hình Môi trường

Tất cả file cấu hình đã được push lên GitHub, không cần tạo thêm file nào.

## Bước 6: Chạy Ứng dụng

### Chế độ Development
```bash
python web_app.py
```

### Chế độ Production
```bash
# Sử dụng Gunicorn (Linux/macOS)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app

# Hoặc sử dụng Waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 web_app:app
```

## Bước 7: Truy cập Ứng dụng

Mở trình duyệt và truy cập: `http://localhost:5000`

## Cấu trúc Dự án

```
xlth/
├── main.py                 # Entry point chính
├── web_app.py             # Flask web application
├── audio_processor.py     # Xử lý audio cơ bản
├── advanced_audio_processor.py  # Xử lý audio nâng cao
├── spotify_integration.py # Tích hợp Spotify
├── models/                # Thư mục chứa models
│   ├── create_simple_models.py
│   └── train_models.py
├── templates/             # HTML templates
│   └── index.html
├── static/                # Static files
│   └── results/           # Kết quả xử lý
├── uploads/               # File upload từ user
├── data/                  # Dữ liệu mẫu và cấu hình
├── requirements.txt       # Dependencies cơ bản
├── requirements_advanced.txt  # Dependencies nâng cao
├── README.md              # Hướng dẫn tổng quan
├── README_ADVANCED.md     # Hướng dẫn nâng cao
└── QUICK_START.md         # Hướng dẫn nhanh
```

## Xử lý Lỗi Thường gặp

### Lỗi Import
```bash
# Nếu gặp lỗi import, kiểm tra virtual environment
pip list
# Đảm bảo tất cả packages đã được cài đặt
```

### Lỗi Permission
```bash
# Trên Linux/macOS, có thể cần cấp quyền
chmod +x web_app.py
```

### Lỗi Port đã được sử dụng
```bash
# Thay đổi port trong web_app.py hoặc kill process đang sử dụng port
lsof -ti:5000 | xargs kill -9
```

## Tính năng Chính

1. **Xử lý Audio Cơ bản**: Upload và xử lý file audio
2. **Xử lý Audio Nâng cao**: Các thuật toán xử lý phức tạp
3. **Tích hợp Spotify**: Kết nối với Spotify API
4. **Machine Learning**: Training và sử dụng models
5. **Web Interface**: Giao diện web thân thiện

## Lưu ý Quan trọng

- **Dự án môn học**: Tất cả file cấu hình, API keys, và dữ liệu mẫu đã được push lên GitHub
- **Không cần setup phức tạp**: Chỉ cần clone và cài đặt dependencies là chạy được
- **Demo sẵn sàng**: Có sẵn file audio mẫu và models để test

## Liên hệ Hỗ trợ

Nếu gặp vấn đề trong quá trình triển khai, vui lòng:
1. Kiểm tra logs trong thư mục `logs/`
2. Đọc file `README.md` và `README_ADVANCED.md`
3. Liên hệ team development
