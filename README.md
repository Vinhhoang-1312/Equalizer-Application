# 🎵 Advanced Audio Processing Application

## 🚀 Quick Start

```bash
# 1. Setup & Test
python setup_and_test.py

# 2. Run Application
python main_modular.py

# 3. Open Browser
http://127.0.0.1:5000
```

## 📋 Features

### 6 Processing Modules:
- **📁 File Processing** - Upload & manage audio files
- **🎛️ Equalizer** - 10-band EQ with presets (Rock, Pop, Jazz, etc.)
- **🔇 Noise Reduction** - AI-powered noise removal (5 methods)
- **🎵 Genre Classification** - 2 best methods using YOUR local GTZAN data
- **⚡ Real-time Processing** - Low-latency audio processing (<500ms)
- **📊 Analysis** - Visualization and detailed audio analysis

## 🎯 Genre Classification Options

### 1. 🤖 **Custom ML (GTZAN)** - 87% accuracy ✅ READY NOW!
- **Pros**: Already working, trained on your 1000-song GTZAN dataset
- **Data**: `data/gtzan/` - 10 genres (blues, classical, country, etc.)
- **Status**: Pre-trained models ready in `models/` directory

### 2. 🧠 **Musicnn Deep Learning** - 92% accuracy (optional)
- **Pros**: Higher accuracy, pre-trained on millions of songs
- **Install**: `pip install musicnn tensorflow==2.5.0`
- **Note**: May require separate environment due to TensorFlow version

## 🛠️ Tech Stack

- **Backend**: Flask + SocketIO
- **Audio**: Librosa + SoundFile + PyAudio
- **AI/ML**: TensorFlow + Scikit-learn
- **Data**: GTZAN Dataset (1000 songs, 10 genres)
- **Frontend**: Bootstrap + Chart.js
- **Architecture**: Modular engines

## 📁 Your Data Structure

```
data/gtzan/          # Your music dataset (1000 songs)
├── blues/           # 100 blues songs
├── classical/       # 100 classical songs
├── country/         # 100 country songs
└── ... (10 genres total)

models/              # Pre-trained models
├── advanced_genre_classifier.pkl  # Your genre model
└── advanced_scaler.pkl           # Feature scaler
```

## 🎯 Usage

1. **Upload** your audio file
2. **Go to Genre Classification tab**
3. **Choose method:**
   - Click "Custom ML (GTZAN)" - works immediately!  
   - Click "Musicnn Deep Learning" - if installed
   - Click "Compare Both Methods" - see which is better
4. **Get results** - genre prediction with confidence score

## 🎛️ Equalizer vs Real-time Tab: Key Differences

### 🎛️ **Equalizer Tab** - File Processing
**Purpose**: Offline audio file enhancement and analysis
- **Input**: Upload WAV/MP3 files
- **Processing**: Non-real-time, high-quality processing
- **Output**: 
  - ✅ Saves processed files to `static/results/` 
  - ✅ Audio playback (original vs processed)
  - ✅ Enhanced 2D visualizations with time domain
  - ✅ Download links for processed files
- **Visualization**: 
  - 2D waveform comparison plots
  - Frequency response charts
  - Overlay plots with EQ settings
  - Optional spectrograms
- **Use Case**: Studio work, file enhancement, detailed analysis

### ⚡ **Real-time Tab** - Live Processing  
**Purpose**: Live audio input/output processing
- **Input**: Microphone or audio device
- **Processing**: Real-time with low latency (<500ms)
- **Output**:
  - ❌ No file saving (live stream only)
  - ✅ Real-time monitoring and visualization
  - ❌ No download capability (live processing)
- **Visualization**:
  - Live waveform displays
  - Real-time spectrum analysis
  - Level meters and monitoring
- **Use Case**: Live performance, recording, real-time monitoring

### 🔧 **Shared Components**
Both tabs use the same audio processing engines:
- `EqualizerEngine` - 10-band EQ with presets
- `NoiseReductionEngine` - AI noise removal algorithms
- Same filter algorithms (IIR, FIR, FFT)
- Same frequency bands and presets

### 💡 **When to Use Which?**
- **Equalizer Tab**: When you want to enhance existing audio files and save results
- **Real-time Tab**: When you need live audio processing for recording or performance

## 📞 Support

Your system uses LOCAL data - no internet needed! All processing happens on your machine with your GTZAN dataset.

---
*Powered by your local GTZAN dataset* 🎵
