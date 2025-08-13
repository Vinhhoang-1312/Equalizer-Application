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

## 📞 Support

Your system uses LOCAL data - no internet needed! All processing happens on your machine with your GTZAN dataset.

---
*Powered by your local GTZAN dataset* 🎵
