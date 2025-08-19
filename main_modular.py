#!/usr/bin/env python3
"""
Main Application Controller
Điều phối tất cả các module và xử lý giao diện web
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, session
from flask_socketio import SocketIO, emit
import os
import numpy as np
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import our engines
from modules.equalizer_engine import EqualizerEngine
from modules.noise_reduction_engine import NoiseReductionEngine
from modules.genre_classification_engine import GenreClassificationEngine
from modules.realtime_processing_engine import RealTimeProcessingEngine
from modules.audio_analysis_engine import AudioAnalysisEngine

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
        self.sample_rate = 44100 # 22050
        self.equalizer_engine = EqualizerEngine(sample_rate=self.sample_rate)
        self.noise_reduction_engine = NoiseReductionEngine(sample_rate=self.sample_rate)
        self.genre_classification_engine = GenreClassificationEngine(sample_rate=self.sample_rate)
        self.realtime_engine = RealTimeProcessingEngine(sample_rate=self.sample_rate)
        self.analysis_engine = AudioAnalysisEngine()
        
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
                    total_samples = len(audio)
                    channels = 1 if len(audio.shape) == 1 else audio.shape[1]
                    
                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'filepath': filepath,
                        'duration': float(duration),
                        'rms_level': float(rms),
                        'sample_rate': int(sr),
                        'total_samples': int(total_samples),
                        'channels': int(channels)
                    })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/uploads/<path:filename>')
        def serve_upload(filename):
            """Serve uploaded files for the audio player."""
            return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)
        
        @self.app.route('/api/equalizer/visualize', methods=['POST'])
        def visualize_equalizer():
            try:
                data = request.get_json()
                gains = data.get('gains')
                plot_options = data.get('plot_options')

                if self.current_audio is None:
                    return jsonify({'error': 'No audio file loaded'}), 400

                processed_audio = self.equalizer_engine.apply_equalizer(self.current_audio, gains, filter_type='iir')

                plot_paths = self.equalizer_engine.generate_comparison_plots(
                    original_audio=self.current_audio,
                    processed_audio=processed_audio,
                    options=plot_options,
                    output_dir='static/results'
                )

                return jsonify({
                    'success': True,
                    'plot_paths': plot_paths
                })
            
            except Exception as e:
                app.logger.error(f"Error in visualize_equalizer: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/equalizer/process', methods=['POST'])
        def process_equalizer():
            """Process audio with equalizer for saving."""
            try:
                if self.current_audio is None:
                    return jsonify({'error': 'No audio file loaded'}), 400
                
                data = request.get_json()
                gains = data.get('gains', {})
                filter_type = data.get('filter_type', 'iir') # Default to iir if not provided
                
                # Apply equalizer using the main method
                processed_audio = self.equalizer_engine.apply_equalizer(
                    self.current_audio, gains, filter_type=filter_type
                )
                
                # Save processed audio
                output_path = os.path.join(
                    self.app.config['UPLOAD_FOLDER'], 
                    'processed_eq.wav'
                )
                sf.write(output_path, processed_audio, self.sample_rate)
                
                return jsonify({
                    'success': True,
                    'output_path': output_path,
                    'filter_type': filter_type,
                    'gains_used': gains
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/noise_reduction/process', methods=['POST'])
        def process_noise_reduction():
            """Process audio with noise reduction and create detailed comparison"""
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
                
                # Save both original and processed audio files
                timestamp = int(time.time())
                original_filename = f'original_audio_{timestamp}.wav'
                processed_filename = f'processed_nr_{method}_{timestamp}.wav'
                
                original_path = os.path.join(self.app.config['UPLOAD_FOLDER'], original_filename)
                processed_path = os.path.join(self.app.config['UPLOAD_FOLDER'], processed_filename)
                
                # Save audio files
                sf.write(original_path, self.current_audio, self.sample_rate)
                sf.write(processed_path, processed_audio, self.sample_rate)
                
                # Create detailed comparison analysis
                comparison_analysis = self.noise_reduction_engine.create_comparison_analysis(
                    self.current_audio, processed_audio, method, reduction_level
                )
                
                # Add file paths to analysis
                if 'error' in comparison_analysis:
                    return jsonify({'error': comparison_analysis['error']}), 400
                comparison_analysis['audio_files'] = {
                    'original': original_filename,
                    'processed': processed_filename,
                    'original_path': original_path,
                    'processed_path': processed_path
                }
                return jsonify({
                    'success': True,
                    'comparison_analysis': comparison_analysis,
                    'audio_files': comparison_analysis['audio_files'],
                    'method': method,
                    'reduction_level': reduction_level
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
                    result = self.genre_classification_engine.advanced_classifier.option1_librosa_classify(self.current_file_path)
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
                        result1 = self.genre_classification_engine.advanced_classifier.option1_librosa_classify(self.current_file_path)
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
        
        @self.app.route('/api/realtime/start_recording', methods=['POST'])
        def start_recording():
            """Start recording processed audio"""
            try:
                data = request.get_json()
                filename = data.get('filename', f'recorded_{int(time.time())}.wav')
                duration = data.get('duration')  # None for continuous
                
                # Ensure uploads directory exists
                os.makedirs('uploads', exist_ok=True)
                filepath = os.path.join('uploads', filename)
                
                success = self.realtime_engine.start_recording_to_file(filepath, duration)
                
                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Recording started',
                        'filename': filename,
                        'filepath': filepath,
                        'duration': duration
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to start recording'
                    }), 400
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/realtime/stop_recording', methods=['POST'])
        def stop_recording():
            """Stop recording and save file"""
            try:
                success = self.realtime_engine.stop_recording()
                
                if success:
                    # Get recording info
                    stats = self.realtime_engine.get_processing_stats()
                    filename = getattr(self.realtime_engine, 'recording_filename', 'unknown.wav')
                    
                    return jsonify({
                        'success': True,
                        'message': 'Recording saved',
                        'filename': os.path.basename(filename),
                        'filepath': filename,
                        'duration': getattr(self.realtime_engine, 'recording_duration', 0)
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'No recording in progress or failed to save'
                    }), 400
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e)
                }), 500
        
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
        
        @self.app.route('/api/audio/download/<filename>')
        def download_audio_file(filename):
            """Download audio file from uploads folder"""
            try:
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(file_path):
                    return send_file(file_path, as_attachment=True)
                else:
                    return jsonify({'error': 'File not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/equalizer/download_processed_audio', methods=['POST'])
        def download_processed_audio():
            """Download processed audio after equalizer application."""
            try:
                if self.current_audio is None:
                    return jsonify({'error': 'No audio file loaded'}), 400

                data = request.get_json()
                gains = data.get('gains', {})
                filter_type = data.get('filter_type', 'iir')
                download_format = data.get('format', 'wav')

                processed_audio = self.equalizer_engine.apply_equalizer(
                    self.current_audio, gains, filter_type=filter_type
                )

                timestamp = int(time.time())
                output_filename = f'processed_eq_{timestamp}.{download_format}'
                output_path = os.path.join(self.app.config['UPLOAD_FOLDER'], output_filename)

                # Ensure the format is supported by soundfile
                if download_format not in ['wav', 'flac', 'ogg']:
                    return jsonify({'error': f'Unsupported download format: {download_format}'}), 400

                sf.write(output_path, processed_audio, self.sample_rate, format=download_format.upper())

                return send_file(output_path, as_attachment=True, download_name=output_filename)

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
        
        @self.app.route('/api/analysis/analyze', methods=['POST'])
        def analyze_audio():
            """Comprehensive audio analysis"""
            try:
                data = request.get_json()
                
                if not self.current_file_path:
                    return jsonify({'error': 'No audio file uploaded'}), 400
                
                # Get analysis options from request
                analysis_options = data.get('options', {
                    'waveform': True,
                    'spectrogram': True,
                    'frequency': True,
                    'mfcc': True,
                    'chroma': True,
                    'tempo': True
                })
                
                # Perform comprehensive analysis
                result = self.analysis_engine.analyze_audio_comprehensive(
                    self.current_file_path, 
                    analysis_options
                )
                
                if result['success']:
                    # Store results for potential export
                    self.processing_results['analysis'] = result['results']
                    
                    return jsonify({
                        'success': True,
                        'results': result['results'],
                        'message': result['message']
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result['error']
                    }), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analysis/export', methods=['POST'])
        def export_analysis():
            """Export analysis results"""
            try:
                result = self.analysis_engine.export_analysis_report()
                
                if result['success']:
                    return jsonify({
                        'success': True,
                        'report_file': result['report_file'],
                        'plot_files': result['plot_files'],
                        'message': result['message']
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result.get('error', 'Export failed')
                    }), 500
                    
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
