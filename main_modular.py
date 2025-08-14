#!/usr/bin/env python3
"""
Main Application Controller
Điều phối tất cả các module và xử lý giao diện web
"""

from flask import Flask, render_template, request, jsonify, send_file
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
from typing import Dict, List, Optional

# Import our engines
from modules.equalizer_engine import EqualizerEngine
from modules.noise_reduction_engine import NoiseReductionEngine
from modules.genre_classification_engine import GenreClassificationEngine
from modules.realtime_processing_engine import RealTimeProcessingEngine

class MainApplication:
    def __init__(self):
        """Initialize main application with all engines"""
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'advanced-audio-processing-key'
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.config['RESULTS_FOLDER'] = 'static/results'
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Create directories
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(self.app.config['RESULTS_FOLDER'], exist_ok=True)
        
        # Initialize processing engines
        self.sample_rate = 22050
        self.equalizer_engine = EqualizerEngine(sample_rate=self.sample_rate)
        self.noise_reduction_engine = NoiseReductionEngine(sample_rate=self.sample_rate)
        self.genre_classification_engine = GenreClassificationEngine(sample_rate=self.sample_rate)
        self.realtime_engine = RealTimeProcessingEngine(sample_rate=self.sample_rate)
        
        # Set up real-time engine with processing modules
        self.realtime_engine.set_processing_modules(
            equalizer_engine=self.equalizer_engine,
            noise_reduction_engine=self.noise_reduction_engine,
            genre_classification_engine=self.genre_classification_engine
        )
        
        # Application state
        self.current_audio = None
        self.current_file_path = None
        self.processing_results = {}
        
        # Set up routes and socket events
        self._setup_routes()
        self._setup_socket_events()
        
        print("✓ Advanced Audio Processing Application initialized")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Equalizer bands: {len(self.equalizer_engine.frequency_bands)}")
        print(f"  Noise reduction methods: {len(self.noise_reduction_engine.get_available_methods())}")
        print(f"  Genre classification methods: {len(self.genre_classification_engine.get_available_methods())}")
    
    def _setup_routes(self):
        """Set up Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return render_template('index_modular.html')
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_file():
            """Handle file upload"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if file:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Load and analyze audio
                    audio, sr = librosa.load(filepath, sr=self.sample_rate)
                    self.current_audio = audio
                    self.current_file_path = filepath
                    
                    # Basic audio info
                    duration = len(audio) / self.sample_rate
                    rms = np.sqrt(np.mean(audio**2))
                    
                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'filepath': filepath,
                        'duration': float(duration),
                        'rms_level': float(rms),
                        'sample_rate': sr
                    })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/equalizer/process', methods=['POST'])
        def process_equalizer():
            """Process audio with equalizer"""
            try:
                if self.current_audio is None:
                    return jsonify({'error': 'No audio file loaded'}), 400
                
                data = request.get_json()
                method = data.get('method', 'fft')
                preset = data.get('preset')
                custom_gains = data.get('gains', {})
                
                # Apply equalizer
                if preset:
                    processed_audio = self.equalizer_engine.apply_preset(
                        self.current_audio, preset, method
                    )
                    gains_used = self.equalizer_engine.get_preset_gains(preset)
                else:
                    processed_audio = self.equalizer_engine.apply_equalizer_fft(
                        self.current_audio, custom_gains
                    )
                    gains_used = custom_gains
                
                # Save processed audio
                output_path = os.path.join(
                    self.app.config['UPLOAD_FOLDER'], 
                    'processed_eq.wav'
                )
                sf.write(output_path, processed_audio, self.sample_rate)
                
                # Generate frequency response plot
                freqs, response_db = self.equalizer_engine.get_frequency_response(gains_used)
                plot_path = self._plot_frequency_response(freqs, response_db)
                
                return jsonify({
                    'success': True,
                    'output_path': output_path,
                    'gains_used': gains_used,
                    'method': method,
                    'preset': preset,
                    'plot_path': plot_path,
                    'rms_change_db': float(20 * np.log10(
                        np.sqrt(np.mean(processed_audio**2)) / 
                        max(np.sqrt(np.mean(self.current_audio**2)), 1e-10)
                    ))
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/noise_reduction/process', methods=['POST'])
        def process_noise_reduction():
            """Process audio with noise reduction"""
            try:
                if self.current_audio is None:
                    return jsonify({'error': 'No audio file loaded'}), 400
                
                data = request.get_json()
                method = data.get('method', 'autoencoder')
                reduction_level = data.get('reduction_level', 0.7)
                
                # Apply noise reduction
                processed_audio = self.noise_reduction_engine.reduce_noise(
                    self.current_audio, method, reduction_level
                )
                
                # Save processed audio
                output_path = os.path.join(
                    self.app.config['UPLOAD_FOLDER'], 
                    'processed_nr.wav'
                )
                sf.write(output_path, processed_audio, self.sample_rate)
                
                # Analyze noise characteristics
                noise_analysis = self.noise_reduction_engine.analyze_noise_characteristics(
                    self.current_audio
                )
                processed_analysis = self.noise_reduction_engine.analyze_noise_characteristics(
                    processed_audio
                )
                
                # Generate before/after spectrograms
                plot_path = self._plot_noise_reduction_comparison(
                    self.current_audio, processed_audio
                )
                
                return jsonify({
                    'success': True,
                    'output_path': output_path,
                    'method': method,
                    'reduction_level': reduction_level,
                    'original_analysis': noise_analysis,
                    'processed_analysis': processed_analysis,
                    'snr_improvement': float(
                        processed_analysis.get('snr_estimate', 0) - 
                        noise_analysis.get('snr_estimate', 0)
                    ),
                    'plot_path': plot_path,
                    'comparison_plot': plot_path is not None
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/genre_classification/classify', methods=['POST'])
        def classify_genre():
            """Classify audio genre (old method)"""
            try:
                if self.current_audio is None:
                    return jsonify({'error': 'No audio file loaded'}), 400
                
                data = request.get_json()
                method = data.get('method', 'ensemble')
                
                # Classify genre
                genre, confidence, additional_info = self.genre_classification_engine.classify_genre(
                    self.current_audio, method
                )
                
                # Get model info
                model_info = self.genre_classification_engine.get_model_info()
                
                return jsonify({
                    'success': True,
                    'predicted_genre': genre,
                    'confidence': float(confidence),
                    'method': method,
                    'additional_info': additional_info,
                    'model_info': model_info,
                    'available_genres': self.genre_classification_engine.get_available_genres()
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/genre_classification/classify_best', methods=['POST'])
        def classify_genre_best():
            """🎯 NEW: Classify using 2 BEST methods (Musicnn, Custom ML)"""
            try:
                if self.current_file_path is None:
                    return jsonify({
                        'error': 'No audio file loaded',
                        'message': 'Please upload an audio file first'
                    }), 400
                
                data = request.get_json()
                option = data.get('option', 'both')
                
                print(f"🎵 Using BEST Genre Classification Methods (option: {option})...")
                
                if option == 'option1':
                    # Use Advanced Librosa/Musicnn method
                    result = self.genre_classification_engine.advanced_classifier.option1_musicnn_classify(self.current_file_path)
                    return jsonify({
                        'success': True,
                        'predicted_genre': result.get('predicted_genre', 'unknown'),
                        'confidence': result.get('confidence', 0.0),
                        'method': result.get('method', 'Advanced Analysis'),
                        'additional_info': result
                    })
                elif option == 'option2':
                    # Use Custom ML method
                    result = self.genre_classification_engine.advanced_classifier.option2_custom_ml_classify(self.current_file_path)
                    return jsonify({
                        'success': True,
                        'predicted_genre': result.get('predicted_genre', 'unknown'),
                        'confidence': result.get('confidence', 0.0),
                        'method': result.get('method', 'Custom ML (GTZAN)'),
                        'additional_info': result
                    })
                elif option == 'both':
                    # Run both methods and compare
                    try:
                        result1 = self.genre_classification_engine.advanced_classifier.option1_musicnn_classify(self.current_file_path)
                        result2 = self.genre_classification_engine.advanced_classifier.option2_custom_ml_classify(self.current_file_path)
                        
                        # Choose the result with higher confidence
                        if result1.get('confidence', 0) >= result2.get('confidence', 0):
                            best_result = result1
                            comparison = f"Option1: {result1.get('predicted_genre')} ({result1.get('confidence', 0):.2f}), Option2: {result2.get('predicted_genre')} ({result2.get('confidence', 0):.2f})"
                        else:
                            best_result = result2
                            comparison = f"Option2: {result2.get('predicted_genre')} ({result2.get('confidence', 0):.2f}), Option1: {result1.get('predicted_genre')} ({result1.get('confidence', 0):.2f})"
                        
                        return jsonify({
                            'success': True,
                            'predicted_genre': best_result.get('predicted_genre', 'unknown'),
                            'confidence': best_result.get('confidence', 0.0),
                            'method': 'Combined Analysis (Best Result)',
                            'comparison': comparison,
                            'option1_result': result1,
                            'option2_result': result2
                        })
                        
                    except Exception as e:
                        print(f"❌ Both methods failed: {e}")
                        return jsonify({'error': f'Classification failed: {str(e)}'}), 500
                else:
                    return jsonify({'error': 'Invalid option parameter'}), 400
                
            except Exception as e:
                print(f"❌ Genre classification error: {e}")
                return jsonify({'error': str(e)}), 500
                
            except Exception as e:
                print(f"❌ Genre classification error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/realtime/start', methods=['POST'])
        def start_realtime():
            """Start real-time processing"""
            try:
                data = request.get_json()
                
                # Set processing parameters
                equalizer_params = data.get('equalizer_params', {})
                noise_method = data.get('noise_method', 'spectral')
                noise_level = data.get('noise_reduction_level', 0.5)
                enabled_modules = data.get('enabled_modules', {
                    'equalizer': True,
                    'noise_reduction': True,
                    'genre_classification': True
                })
                
                # Configure real-time engine
                self.realtime_engine.set_equalizer_params(equalizer_params)
                self.realtime_engine.set_noise_reduction_params(noise_method, noise_level)
                self.realtime_engine.enable_processing(**enabled_modules)
                
                # Set callbacks
                self.realtime_engine.set_callbacks(
                    audio_callback=self._realtime_audio_callback,
                    genre_callback=self._realtime_genre_callback
                )
                
                # Start processing
                success = self.realtime_engine.start_processing_sounddevice()
                
                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Real-time processing started'
                    })
                else:
                    return jsonify({'error': 'Failed to start real-time processing'}), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/realtime/stop', methods=['POST'])
        def stop_realtime():
            """Stop real-time processing"""
            try:
                self.realtime_engine.stop_processing()
                return jsonify({
                    'success': True,
                    'message': 'Real-time processing stopped'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/realtime/stats', methods=['GET'])
        def get_realtime_stats():
            """Get real-time processing statistics"""
            try:
                stats = self.realtime_engine.get_processing_stats()
                return jsonify(stats)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/audio_devices', methods=['GET'])
        def get_audio_devices():
            """Get available audio devices"""
            try:
                devices = self.realtime_engine.get_audio_devices()
                return jsonify(devices)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/equalizer/presets', methods=['GET'])
        def get_equalizer_presets():
            """Get available equalizer presets"""
            try:
                presets = {}
                preset_names = self.equalizer_engine.get_available_presets()
                
                for preset_name in preset_names:
                    presets[preset_name] = self.equalizer_engine.get_preset_gains(preset_name)
                
                return jsonify({
                    'presets': presets,
                    'frequency_bands': self.equalizer_engine.frequency_bands
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/noise_reduction/methods', methods=['GET'])
        def get_noise_reduction_methods():
            """Get available noise reduction methods"""
            try:
                methods = self.noise_reduction_engine.get_available_methods()
                return jsonify({'methods': methods})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/genre_classification/info', methods=['GET'])
        def get_genre_classification_info():
            """Get genre classification information"""
            try:
                info = self.genre_classification_engine.get_model_info()
                methods = self.genre_classification_engine.get_available_methods()
                info['available_methods'] = methods
                return jsonify(info)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _setup_socket_events(self):
        """Set up SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            emit('connected', {'message': 'Connected to Advanced Audio Processing'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            # Stop real-time processing if active
            if self.realtime_engine.is_processing:
                self.realtime_engine.stop_processing()
    
    def _realtime_audio_callback(self, audio_chunk):
        """Callback for real-time processed audio"""
        try:
            # Calculate audio metrics
            rms = np.sqrt(np.mean(audio_chunk**2))
            
            # Emit audio data to connected clients
            self.socketio.emit('realtime_audio', {
                'rms_level': float(rms),
                'chunk_size': len(audio_chunk),
                'timestamp': time.time()
            })
        except Exception as e:
            print(f"⚠️ Audio callback error: {e}")
    
    def _realtime_genre_callback(self, genre_result):
        """Callback for real-time genre classification"""
        try:
            # Emit genre classification to connected clients
            self.socketio.emit('realtime_genre', genre_result)
        except Exception as e:
            print(f"⚠️ Genre callback error: {e}")
    
    def _plot_frequency_response(self, freqs, response_db):
        """Generate frequency response plot"""
        try:
            plt.figure(figsize=(12, 6))
            plt.semilogx(freqs, response_db, 'b-', linewidth=2)
            plt.grid(True, alpha=0.3)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Gain (dB)')
            plt.title('Equalizer Frequency Response')
            plt.xlim([20, 20000])
            plt.ylim([-25, 25])
            
            # Mark frequency bands
            for band_name, freq in self.equalizer_engine.frequency_bands.items():
                plt.axvline(x=freq, color='r', linestyle='--', alpha=0.5)
                plt.text(freq, 20, band_name, rotation=90, ha='right', va='bottom', fontsize=8)
            
            plot_path = os.path.join(self.app.config['RESULTS_FOLDER'], 'freq_response.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Error plotting frequency response: {e}")
            return None
    
    def _plot_noise_reduction_comparison(self, original_audio, processed_audio):
        """Generate noise reduction comparison plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Original waveform
            time_axis = np.linspace(0, len(original_audio)/self.sample_rate, len(original_audio))
            axes[0, 0].plot(time_axis, original_audio)
            axes[0, 0].set_title('Original Audio Waveform')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            
            # Processed waveform
            axes[0, 1].plot(time_axis[:len(processed_audio)], processed_audio)
            axes[0, 1].set_title('Processed Audio Waveform')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude')
            
            # Original spectrogram
            D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)))
            librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', 
                                   sr=self.sample_rate, ax=axes[1, 0])
            axes[1, 0].set_title('Original Spectrogram')
            
            # Processed spectrogram
            D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(processed_audio)))
            librosa.display.specshow(D_proc, y_axis='hz', x_axis='time', 
                                   sr=self.sample_rate, ax=axes[1, 1])
            axes[1, 1].set_title('Processed Spectrogram')
            
            plt.tight_layout()
            
            plot_path = os.path.join(self.app.config['RESULTS_FOLDER'], 'noise_reduction_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Error plotting noise reduction comparison: {e}")
            return None
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the application"""
        print(f"🚀 Starting Advanced Audio Processing Application")
        print(f"   URL: http://{host}:{port}")
        print(f"   Debug mode: {debug}")
        
        self.socketio.run(self.app, host=host, port=port, debug=debug)

# Initialize and run application
if __name__ == '__main__':
    app = MainApplication()
    app.run(debug=True)
