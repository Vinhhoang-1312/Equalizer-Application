# 🔬 CHI TIẾT KỸ THUẬT - CÁCH ỨNG DỤNG HOẠT ĐỘNG

## 🎵 **1. CÁCH ĐỌC ÂM THANH**

### **Bước 1: Đọc file âm thanh**
```python
# Trong advanced_audio_processor.py, dòng 580
audio, sr = librosa.load(file_path, sr=self.sample_rate)
```

**Chuyện gì xảy ra:**
- `librosa.load()` đọc file âm thanh (.wav, .mp3, .flac)
- Chuyển đổi thành **mảng số** (NumPy array) biểu diễn biên độ sóng âm
- `sr = 22050 Hz` (tốc độ lấy mẫu)
- Ví dụ: 1 giây âm thanh = 22050 số

### **Bước 2: Chuyển sang miền tần số**
```python
# Trong advanced_equalizer(), dòng 80-82
fft_audio = fft(audio)  # Fast Fourier Transform
freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
```

**Chuyện gì xảy ra:**
- **FFT** chuyển đổi từ miền thời gian → miền tần số
- Âm thanh = tổng của nhiều sóng sin với tần số khác nhau
- Kết quả: biểu diễn năng lượng ở mỗi tần số (20Hz - 20000Hz)

## 🎛️ **2. CÁCH CẮT TẦN SỐ (EQUALIZER)**

### **6 dải tần số được xử lý:**

```python
# Dòng 88-109 trong advanced_equalizer()
# Sub-bass: 20-60 Hz (tiếng trống bass sâu)
sub_bass_mask = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 60)
freq_response[sub_bass_mask] *= sub_bass_gain

# Bass: 60-250 Hz (tiếng bass, trống)
bass_mask = (np.abs(freqs) >= 60) & (np.abs(freqs) <= 250)
freq_response[bass_mask] *= bass_gain

# Mid: 250-2000 Hz (giọng hát, guitar)
mid_mask = (np.abs(freqs) >= 250) & (np.abs(freqs) <= 2000)
freq_response[mid_mask] *= mid_gain

# Treble: 2000-8000 Hz (tiếng cymbal, violin cao)
treble_mask = (np.abs(freqs) >= 2000) & (np.abs(freqs) <= 8000)
freq_response[treble_mask] *= treble_gain

# Presence: 8000-12000 Hz (tiếng sibilant, sparkle)
presence_mask = (np.abs(freqs) >= 8000) & (np.abs(freqs) <= 12000)
freq_response[presence_mask] *= presence_gain

# Air: 12000-20000 Hz (tiếng "không khí", brilliance)
air_mask = (np.abs(freqs) >= 12000) & (np.abs(freqs) <= 20000)
freq_response[air_mask] *= air_gain
```

**Cách hoạt động:**
- Tăng/giảm biên độ của từng dải tần số
- `gain = 1.0` = không thay đổi
- `gain = 2.0` = tăng gấp đôi
- `gain = 0.5` = giảm một nửa

## 🎯 **3. CÁCH NHẬN DIỆN THỂ LOẠI NHẠC**

### **Bước 1: Trích xuất đặc trưng (Feature Extraction)**

```python
# Trong extract_advanced_features(), dòng 130-180
# MFCC (Mel-frequency Cepstral Coefficients) - 26 features
mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
features.extend([np.mean(mfccs[i]) for i in range(13)])  # Trung bình
features.extend([np.std(mfccs[i]) for i in range(13)])   # Độ lệch chuẩn

# Spectral Centroid - "độ sáng" của âm thanh
spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
features.append(np.mean(spectral_centroids))

# Chroma - năng lượng của 12 nốt nhạc (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
features.extend([np.mean(chroma[i]) for i in range(12)])

# Zero Crossing Rate - độ "ồn" của âm thanh
zcr = librosa.feature.zero_crossing_rate(audio)[0]
features.append(np.mean(zcr))
```

### **Bước 2: Phân loại bằng Machine Learning**

```python
# Trong advanced_genre_classification(), dòng 410-420
# Scale features
if self.feature_scaler is not None:
    features = self.feature_scaler.transform(features.reshape(1, -1))

# Predict genre
prediction = self.genre_classifier.predict_proba(features.reshape(1, -1))[0]
genre_idx = np.argmax(prediction)
confidence = prediction[genre_idx]

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
         'jazz', 'metal', 'pop', 'reggae', 'rock']
```

### **Các đặc trưng quan trọng cho từng thể loại:**

| Thể loại | Đặc trưng chính |
|----------|-----------------|
| **Rock** | Spectral centroid cao (>3000), ZCR cao (>0.1), energy cao |
| **Jazz** | Spectral centroid thấp (<1500), ZCR thấp (<0.05), harmonic ratio cao |
| **Pop** | Spectral centroid trung bình (>2500), danceability cao |
| **Metal** | ZCR rất cao (>0.15), energy cao, tempo nhanh |
| **Classical** | Harmonic ratio cao, acousticness cao, tempo chậm |

## 📁 **4. VỀ FOLDER DATA VÀ DOWNLOAD**

### **Hiện tại:**
- **KHÔNG CÓ** câu lệnh download riêng
- Tất cả data đã được push lên GitHub
- Khi `git clone` → tất cả file trong `data/` sẽ được tải về

### **Nếu muốn thêm data mới:**
```bash
# Tạo script download
python models/train_models.py  # Có thể tạo synthetic data
# Hoặc tải thủ công từ Kaggle, Spotify API
```

## 🎵 **5. TẠI SAO CHỈ CẦN ĐOẠN NGẮN MÀ NHẬN DIỆN ĐƯỢC CẢ BÀI?**

### **Nguyên lý hoạt động:**

1. **Phân tích từng đoạn nhỏ (3-5 giây)**
2. **Tổng hợp kết quả** để đưa ra quyết định cuối cùng

```python
# Ví dụ: Bài hát 3 phút = 180 giây
# Chia thành 36 đoạn, mỗi đoạn 5 giây
# Mỗi đoạn → dự đoán genre
# Tổng hợp: thể loại nào xuất hiện nhiều nhất = thể loại của bài hát
```

### **Các phương pháp tổng hợp:**
- **Majority Voting**: Thể loại nào được dự đoán nhiều nhất
- **Averaging Probabilities**: Tính trung bình xác suất
- **Weighted Average**: Đoạn đầu/cuối có trọng số cao hơn

## 🔧 **6. 4 PHƯƠNG PHÁP NOISE REDUCTION**

### **Tất cả đều làm cùng 1 việc: GIẢM NHIỄU**
### **Nhưng cách làm khác nhau:**

| Phương pháp | Cách hoạt động | Cần training? |
|-------------|----------------|---------------|
| **Autoencoder (ML)** | Neural network học cách "làm sạch" | ✅ Có |
| **Wiener Filter** | Toán học: ước tính noise và trừ đi | ❌ Không |
| **Spectral Subtraction** | Trừ phổ noise khỏi phổ signal | ❌ Không |
| **Adaptive Filter** | Tự điều chỉnh theo thời gian thực | ❌ Không |

### **Chi tiết từng phương pháp:**

#### **1. Autoencoder (ML)**
```python
# Dòng 288-317 trong advanced_audio_processor.py
def _autoencoder_denoise(self, audio: np.ndarray) -> np.ndarray:
    # Chuyển thành spectrogram
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Neural network predict magnitude "sạch"
    clean_magnitude = self.noise_reducer.predict(magnitude_reshaped)
    
    # Reconstruct audio
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft)
```

#### **2. Wiener Filter**
```python
# Dòng 319-339
def _advanced_wiener_filter(self, audio: np.ndarray) -> np.ndarray:
    # Ước tính noise từ nhiều đoạn
    noise_estimates = []
    for i in range(0, len(audio) - segment_length, segment_length):
        segment = audio[i:i + segment_length]
        noise_estimates.append(np.mean(segment**2))
    
    noise_estimate = np.median(noise_estimates)
    
    # Công thức Wiener: gain = signal_power / (signal_power + noise)
    signal_power = np.convolve(audio**2, np.ones(1000)/1000, mode='same')
    wiener_gain = signal_power / (signal_power + noise_estimate)
    
    # Áp dụng gain
    denoised = audio * wiener_gain
```

#### **3. Spectral Subtraction**
```python
# Dòng 341-364
def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
    # Ước tính noise spectrum từ 0.1s đầu
    noise_samples = int(0.1 * self.sample_rate)
    noise_spectrum = np.abs(fft_audio[:noise_samples])
    noise_estimate = np.mean(noise_spectrum)
    
    # Trừ noise spectrum
    signal_spectrum = np.abs(fft_audio)
    clean_spectrum = signal_spectrum - alpha * noise_estimate
    clean_spectrum = np.maximum(clean_spectrum, beta * signal_spectrum)
```

#### **4. Adaptive Filter**
```python
# Dòng 366-394
def _adaptive_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
    # Sử dụng LMS algorithm
    mu = 0.01  # Step size
    filter_coeffs = np.zeros(filter_length)
    
    # Cập nhật filter coefficients theo thời gian thực
    for i in range(filter_length, len(audio)):
        x = audio[i-filter_length:i]  # Input vector
        y = np.dot(filter_coeffs, x)  # Filter output
        error = audio[i] - y          # Error
        filter_coeffs += mu * error * x  # Update coefficients
```

## 🎯 **KẾT LUẬN**

- **Đọc âm thanh**: `librosa.load()` → mảng số
- **Cắt tần số**: FFT → điều chỉnh từng dải → IFFT
- **Nhận diện genre**: Trích xuất 100+ features → ML model
- **Giảm nhiễu**: 4 phương pháp khác nhau, cùng mục đích
- **Đoạn ngắn**: Phân tích từng đoạn → tổng hợp kết quả
- **Data**: Đã có sẵn trong repo, không cần download riêng
