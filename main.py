#!/usr/bin/env python3
"""
Advanced Audio Processing Application
Ứng dụng xử lý âm thanh với Equalizer, Giảm nhiễu ML, và Phân loại thể loại nhạc

Tính năng chính:
1. Equalizer 6-band với điều chỉnh tần số
2. Giảm nhiễu bằng Machine Learning (Autoencoder, Wiener, Spectral Subtraction)
3. Phân loại thể loại nhạc với độ chính xác >85%
4. Xử lý real-time với độ trễ <200ms
5. Giao diện web hiện đại
6. Tích hợp Spotify API cho training data
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    required_packages = [
        'librosa', 'numpy', 'scipy', 'tensorflow', 'sklearn',
        'flask', 'flask_socketio', 'sounddevice', 'soundfile',
        'matplotlib', 'seaborn', 'pandas', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Thiếu các package: {', '.join(missing_packages)}")
        print("Cài đặt bằng lệnh: pip install -r requirements.txt")
        return False
    
    print("✅ Tất cả dependencies đã sẵn sàng")
    return True

def train_advanced_models():
    """Huấn luyện các mô hình nâng cao"""
    try:
        from models.train_models import ModelTrainer
        print("🎵 Bắt đầu huấn luyện mô hình...")
        trainer = ModelTrainer()
        trainer.train_all_models()
        print("✅ Huấn luyện mô hình hoàn tất!")
    except Exception as e:
        print(f"❌ Lỗi huấn luyện mô hình: {e}")
        return False
    return True
# //////////
def run_web_app():
    """Chạy ứng dụng web"""
    try:
        from web_app import app, socketio
        print("🌐 Khởi động ứng dụng web...")
        print("📱 Truy cập tại: http://localhost:5001")
        socketio.run(app, debug=True, host='0.0.0.0', port=5001)
    except ImportError as e:
        print(f"❌ Không thể import web app: {e}")
        print("Cài đặt web dependencies: pip install flask flask-socketio")
        return False
    except Exception as e:
        print(f"❌ Lỗi khởi động web app: {e}")
        return False

def run_realtime_processing():
    """Chạy xử lý real-time"""
    try:
        from advanced_audio_processor import AdvancedAudioProcessor
        import sounddevice as sd
        
        processor = AdvancedAudioProcessor()
        print("🎤 Bắt đầu xử lý real-time...")
        print("Nhấn Ctrl+C để dừng")
        
        def callback(result):
            print(f"🎵 Thể loại: {result['genre']} (Độ tin cậy: {result['confidence']:.2%})")
        
        processor.start_advanced_real_time_processing(callback)
        
    except KeyboardInterrupt:
        print("\n⏹️ Dừng xử lý real-time")
    except Exception as e:
        print(f"❌ Lỗi xử lý real-time: {e}")

def run_spotify_integration():
    """Chạy tích hợp Spotify"""
    try:
        from spotify_integration import SpotifyIntegration
        
        # Kiểm tra credentials
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            print("❌ Thiếu Spotify credentials")
            print("Đặt SPOTIFY_CLIENT_ID và SPOTIFY_CLIENT_SECRET")
            return False
        
        spotify = SpotifyIntegration()
        print("🎵 Tích hợp Spotify đã sẵn sàng")
        
        # Demo search
        genres = ['pop', 'rock', 'jazz', 'classical']
        print(f"🔍 Tìm kiếm {len(genres)} thể loại...")
        
        for genre in genres:
            tracks = spotify.search_tracks_by_genre(genre, limit=5)
            print(f"  {genre}: {len(tracks)} tracks")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tích hợp Spotify: {e}")
        return False

def run_demo():
    """Chạy demo với file test"""
    try:
        from advanced_audio_processor import AdvancedAudioProcessor
        
        processor = AdvancedAudioProcessor()
        
        # Kiểm tra file test
        test_file = "test_audio.wav"
        if not os.path.exists(test_file):
            print(f"❌ Không tìm thấy file test: {test_file}")
            return False
        
        print(f"🎵 Xử lý file demo: {test_file}")
        
        # Xử lý với các tham số khác nhau
        equalizer_params = {
            'bass_gain': 1.5,
            'mid_gain': 0.8,
            'treble_gain': 1.2,
            'sub_bass_gain': 1.3,
            'presence_gain': 0.9,
            'air_gain': 1.1
        }
        
        results = processor.process_audio_file_advanced(
            test_file,
            equalizer_params=equalizer_params,
            denoise_method='autoencoder',
            analyze=True
        )
        
        print(f"🎵 Kết quả phân loại:")
        print(f"  Thể loại: {results['genre']}")
        print(f"  Độ tin cậy: {results['confidence']:.2%}")
        print(f"  Tempo: {results['analysis'].get('tempo', 'N/A')} BPM")
        print(f"  Năng lượng: {results['analysis'].get('energy', 'N/A'):.3f}")
        
        # Tạo visualization
        output_path = "static/results/demo_analysis.png"
        processor.create_visualization(results, output_path)
        print(f"📊 Đã tạo biểu đồ: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi demo: {e}")
        return False

def process_single_file(input_file, output_file, equalizer_params=None, denoise_method='autoencoder'):
    """Xử lý một file âm thanh"""
    try:
        from advanced_audio_processor import AdvancedAudioProcessor
        
        if not os.path.exists(input_file):
            print(f"❌ Không tìm thấy file: {input_file}")
            return False
        
        processor = AdvancedAudioProcessor()
        
        # Tham số equalizer mặc định
        if equalizer_params is None:
            equalizer_params = {
                'bass_gain': 1.0,
                'mid_gain': 1.0,
                'treble_gain': 1.0,
                'sub_bass_gain': 1.0,
                'presence_gain': 1.0,
                'air_gain': 1.0
            }
        
        print(f"🎵 Xử lý file: {input_file}")
        print(f"⚙️ Tham số equalizer: {equalizer_params}")
        print(f"🔇 Phương pháp giảm nhiễu: {denoise_method}")
        
        results = processor.process_audio_file_advanced(
            input_file,
            equalizer_params=equalizer_params,
            denoise_method=denoise_method,
            analyze=True
        )
        
        # Lưu file đã xử lý
        processor.save_audio(results['processed_audio'], output_file)
        
        print(f"✅ Đã lưu file: {output_file}")
        print(f"🎵 Thể loại: {results['genre']} (Độ tin cậy: {results['confidence']:.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi xử lý file: {e}")
        return False

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description="Advanced Audio Processing Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python main.py --web                    # Chạy web app
  python main.py --train                  # Huấn luyện mô hình
  python main.py --realtime               # Xử lý real-time
  python main.py --demo                   # Chạy demo
  python main.py --input file.wav --output processed.wav  # Xử lý file
        """
    )
    
    # Các tùy chọn chính
    parser.add_argument('--web', action='store_true', help='Chạy ứng dụng web')
    parser.add_argument('--train', action='store_true', help='Huấn luyện mô hình')
    parser.add_argument('--realtime', action='store_true', help='Xử lý real-time')
    parser.add_argument('--spotify', action='store_true', help='Tích hợp Spotify')
    parser.add_argument('--demo', action='store_true', help='Chạy demo')
    parser.add_argument('--check-deps', action='store_true', help='Kiểm tra dependencies')
    
    # Xử lý file
    parser.add_argument('--input', type=str, help='File âm thanh đầu vào')
    parser.add_argument('--output', type=str, help='File âm thanh đầu ra')
    
    # Tham số equalizer
    parser.add_argument('--bass', type=float, default=1.0, help='Bass gain (0.0-3.0)')
    parser.add_argument('--mid', type=float, default=1.0, help='Mid gain (0.0-3.0)')
    parser.add_argument('--treble', type=float, default=1.0, help='Treble gain (0.0-3.0)')
    parser.add_argument('--sub-bass', type=float, default=1.0, help='Sub-bass gain (0.0-3.0)')
    parser.add_argument('--presence', type=float, default=1.0, help='Presence gain (0.0-3.0)')
    parser.add_argument('--air', type=float, default=1.0, help='Air gain (0.0-3.0)')
    
    # Tham số khác
    parser.add_argument('--denoise', type=str, default='autoencoder', 
                       choices=['autoencoder', 'wiener', 'spectral_subtraction', 'adaptive'],
                       help='Phương pháp giảm nhiễu')
    
    args = parser.parse_args()
    
    # Hiển thị banner
    print("🎵 Advanced Audio Processing Application")
    print("=" * 50)
    
    # Kiểm tra dependencies nếu được yêu cầu
    if args.check_deps:
        check_dependencies()
        return
    
    # Xử lý file đơn lẻ
    if args.input and args.output:
        equalizer_params = {
            'bass_gain': args.bass,
            'mid_gain': args.mid,
            'treble_gain': args.treble,
            'sub_bass_gain': args.sub_bass,
            'presence_gain': args.presence,
            'air_gain': args.air
        }
        
        success = process_single_file(
            args.input, 
            args.output, 
            equalizer_params, 
            args.denoise
        )
        return
    
    # Các chế độ khác
    if args.train:
        train_advanced_models()
    elif args.realtime:
        run_realtime_processing()
    elif args.spotify:
        run_spotify_integration()
    elif args.demo:
        run_demo()
    else:
        # Mặc định chạy web app
        run_web_app()

if __name__ == "__main__":
    main() 