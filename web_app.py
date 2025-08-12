from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_socketio import SocketIO, emit
import os
import json
import base64
import io
import numpy as np
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename
import threading
import time
from advanced_audio_processor import AdvancedAudioProcessor
from spotify_integration import SpotifyIntegration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()  # Temporarily disabled due to encoding issues

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize processors
audio_processor = AdvancedAudioProcessor()
spotify_integration = SpotifyIntegration()

# Debug: Check model loading
print(f"Debug - Genre classifier loaded: {audio_processor.genre_classifier is not None}")
print(f"Debug - Noise reducer loaded: {audio_processor.noise_reducer is not None}")
print(f"Debug - Feature scaler loaded: {audio_processor.feature_scaler is not None}")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Global variables for real-time processing
real_time_processing = False
real_time_thread = None

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get processing parameters
        equalizer_params = {
            'bass_gain': float(request.form.get('bass_gain', 1.0)),
            'mid_gain': float(request.form.get('mid_gain', 1.0)),
            'treble_gain': float(request.form.get('treble_gain', 1.0)),
            'sub_bass_gain': float(request.form.get('sub_bass_gain', 1.0)),
            'presence_gain': float(request.form.get('presence_gain', 1.0)),
            'air_gain': float(request.form.get('air_gain', 1.0))
        }
        
        denoise_method = request.form.get('denoise_method', 'autoencoder')
        
        try:
            print(f"Web app - Starting audio processing for {filepath}")
            # Process audio
            results = audio_processor.process_audio_file_advanced(
                filepath, 
                equalizer_params=equalizer_params,
                denoise_method=denoise_method,
                analyze=True
            )
            print(f"Web app - Audio processing completed")
            
            # Save processed audio
            output_filename = f"processed_{filename}"
            output_path = os.path.join('static/results', output_filename)
            audio_processor.save_audio(results['processed_audio'], output_path)
            
            # Create visualization
            viz_filename = f"analysis_{filename.rsplit('.', 1)[0]}.png"
            viz_path = os.path.join('static/results', viz_filename)
            try:
                audio_processor.create_visualization(results, viz_path)
            except Exception as viz_error:
                print(f"⚠️ Main visualization failed: {viz_error}")
                # Create a simple text-based visualization as fallback
                try:
                    plt.clf()
                    plt.close('all')
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    ax.text(0.5, 0.7, f'Audio Processing Completed', ha='center', va='center', fontsize=20, weight='bold')
                    ax.text(0.5, 0.5, f'Genre: {results["genre"]}', ha='center', va='center', fontsize=16)
                    ax.text(0.5, 0.3, f'Confidence: {results["confidence"]:.1%}', ha='center', va='center', fontsize=16)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    
                    os.makedirs('static/results', exist_ok=True)
                    fig.canvas.draw()
                    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    plt.close('all')
                    print(f"📊 Simple fallback visualization created: {viz_path}")
                except Exception as fallback_error:
                    print(f"❌ Even simple visualization failed: {fallback_error}")
                    viz_filename = None
            
            # Debug output
            print(f"Web app debug - Genre: {results['genre']}")
            print(f"Web app debug - Confidence: {results['confidence']}")
            print(f"Web app debug - Confidence type: {type(results['confidence'])}")
            
            # Force genre classification if it's Unknown
            if results['genre'] == 'Unknown' or results['confidence'] == 0.0:
                print("Web app - Re-running genre classification...")
                import librosa
                audio, sr = librosa.load(filepath, sr=audio_processor.sample_rate)
                genre, confidence, info = audio_processor.advanced_genre_classification(audio)
                results['genre'] = genre
                results['confidence'] = confidence
                print(f"Web app - New genre: {genre}, confidence: {confidence}")
            
            # Prepare response
            response_data = {
                'success': True,
                'original_file': filename,
                'processed_file': output_filename,
                'visualization': viz_filename if viz_filename else None,
                'genre': results['genre'],
                'confidence': f"{results['confidence']:.1%}",
                'analysis': results['analysis'],
                'additional_info': results['additional_info']
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/realtime')
def realtime():
    """Real-time processing page"""
    return render_template('realtime.html')

@socketio.on('start_realtime')
def handle_start_realtime(data):
    """Start real-time audio processing"""
    global real_time_processing, real_time_thread
    
    if real_time_processing:
        emit('error', {'message': 'Real-time processing already running'})
        return
    
    equalizer_params = {
        'bass_gain': float(data.get('bass_gain', 1.0)),
        'mid_gain': float(data.get('mid_gain', 1.0)),
        'treble_gain': float(data.get('treble_gain', 1.0)),
        'sub_bass_gain': float(data.get('sub_bass_gain', 1.0)),
        'presence_gain': float(data.get('presence_gain', 1.0)),
        'air_gain': float(data.get('air_gain', 1.0))
    }
    
    denoise_method = data.get('denoise_method', 'autoencoder')
    
    def realtime_callback(result):
        if real_time_processing:
            socketio.emit('realtime_update', {
                'genre': result['genre'],
                'confidence': f"{result['confidence']:.1%}",
                'timestamp': time.time()
            })
    
    real_time_processing = True
    real_time_thread = threading.Thread(
        target=audio_processor.start_advanced_real_time_processing,
        args=(realtime_callback, equalizer_params, denoise_method)
    )
    real_time_thread.daemon = True
    real_time_thread.start()
    
    emit('realtime_started', {'message': 'Real-time processing started'})

@socketio.on('stop_realtime')
def handle_stop_realtime():
    """Stop real-time audio processing"""
    global real_time_processing
    
    real_time_processing = False
    audio_processor.stop_real_time_processing()
    
    emit('realtime_stopped', {'message': 'Real-time processing stopped'})

@app.route('/spotify')
def spotify():
    """Spotify integration page"""
    return render_template('spotify.html')

@app.route('/api/spotify/search', methods=['POST'])
def spotify_search():
    """Search Spotify for tracks by genre"""
    data = request.get_json()
    genre = data.get('genre', 'pop')
    limit = data.get('limit', 10)
    
    tracks = spotify_integration.search_tracks_by_genre(genre, limit)
    
    return jsonify({
        'success': True,
        'tracks': tracks,
        'genre': genre
    })

@app.route('/api/spotify/features', methods=['POST'])
def spotify_features():
    """Get audio features for tracks"""
    data = request.get_json()
    track_ids = data.get('track_ids', [])
    
    features = spotify_integration.get_audio_features(track_ids)
    
    return jsonify({
        'success': True,
        'features': features
    })

@app.route('/api/spotify/dataset', methods=['POST'])
def create_spotify_dataset():
    """Create training dataset from Spotify"""
    data = request.get_json()
    genres = data.get('genres', ['pop', 'rock', 'classical', 'jazz'])
    tracks_per_genre = data.get('tracks_per_genre', 50)
    
    try:
        df = spotify_integration.create_training_dataset(genres, tracks_per_genre)
        
        return jsonify({
            'success': True,
            'message': f'Dataset created with {len(df)} tracks',
            'genres': genres,
            'tracks_per_genre': tracks_per_genre
        })
    except Exception as e:
        return jsonify({'error': f'Failed to create dataset: {str(e)}'}), 500

@app.route('/analysis')
def analysis():
    """Audio analysis page"""
    return render_template('analysis.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """Analyze audio characteristics"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load audio
            audio, sr = librosa.load(filepath, sr=22050)
            
            # Analyze audio characteristics
            analysis = audio_processor.analyze_audio_characteristics(audio)
            
            # Create detailed analysis visualization
            try:
                # Clear any existing plots
                plt.clf()
                plt.close('all')
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Waveform
                time_axis = np.linspace(0, len(audio)/sr, len(audio))
                axes[0, 0].plot(time_axis[:22050], audio[:22050])
                axes[0, 0].set_title('Waveform')
                axes[0, 0].set_xlabel('Time (s)')
                axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].grid(True)
                
                # Spectrogram
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1])
                axes[0, 1].set_title('Spectrogram')
                
                # MFCC
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1, 0])
                axes[1, 0].set_title('MFCC')
                
                # Chromagram
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[1, 1])
                axes[1, 1].set_title('Chroma')
                
                plt.tight_layout()
                
                # Save visualization
                viz_filename = f"detailed_analysis_{filename.rsplit('.', 1)[0]}.png"
                viz_path = os.path.join('static/results', viz_filename)
                
                # Force draw before saving
                fig.canvas.draw()
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plt.close('all')
                
            except Exception as viz_error:
                print(f"⚠️ Detailed visualization failed: {viz_error}")
                # Create simple fallback
                viz_filename = f"detailed_analysis_{filename.rsplit('.', 1)[0]}.png"
                viz_path = os.path.join('static/results', viz_filename)
                
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, f'Audio Analysis Completed\nFile: {filename}', 
                       ha='center', va='center', fontsize=16)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                fig.canvas.draw()
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                plt.close('all')
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'visualization': viz_filename
            })
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed file"""
    try:
        return send_file(
            os.path.join('static/results', filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/status')
def status():
    """Get application status"""
    try:
        return jsonify({
            'status': 'running',
            'real_time_processing': real_time_processing,
            'models_loaded': {
                'noise_reducer': audio_processor.noise_reducer is not None,
                'genre_classifier': audio_processor.genre_classifier is not None
            },
            'spotify_available': not spotify_integration.demo_mode
        })
    except Exception as e:
        print(f"❌ Error in /api/status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🌐 Khởi động Advanced Audio Processing Web App...")
    print("📱 Truy cập tại: http://localhost:5000")
    print("🎵 Để dừng server, nhấn Ctrl+C")
    
    try:
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n⏹️ Dừng server...")
    except Exception as e:
        print(f"❌ Lỗi khởi động server: {e}")
        print("💡 Thử chạy với port khác: python web_app.py --port 5001") 