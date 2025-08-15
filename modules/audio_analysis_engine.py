"""
Audio Analysis Engine
Comprehensive audio analysis with visual feedback for assignment requirements
Includes: Waveform, Spectrogram, Frequency Analysis, MFCC, Chroma, Tempo analysis
"""

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import io
import base64
from datetime import datetime
import os

class AudioAnalysisEngine:
    def __init__(self):
        self.analysis_results = {}
        self.plots_generated = []
        
        # Set plotting style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
    def analyze_audio_comprehensive(self, file_path, analysis_options=None):
        """
        Comprehensive audio analysis with all requested features
        """
        if analysis_options is None:
            analysis_options = {
                'waveform': True,
                'spectrogram': True, 
                'frequency': True,
                'mfcc': True,
                'chroma': True,
                'tempo': True
            }
            
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            # Initialize results
            results = {
                'file_info': {
                    'duration': duration,
                    'sample_rate': sr,
                    'total_samples': len(y),
                    'channels': 1 if len(y.shape) == 1 else y.shape[1]
                },
                'analysis_plots': {},
                'features': {},
                'insights': []
            }
            
            # 1. Waveform Analysis
            if analysis_options.get('waveform', True):
                waveform_plot = self._analyze_waveform(y, sr)
                results['analysis_plots']['waveform'] = waveform_plot
                results['features']['amplitude_stats'] = {
                    'max': float(np.max(np.abs(y))),
                    'mean': float(np.mean(np.abs(y))),
                    'std': float(np.std(y)),
                    'dynamic_range': float(np.max(y) - np.min(y))
                }
                
            # 2. Spectrogram Analysis  
            if analysis_options.get('spectrogram', True):
                spectrogram_plot = self._analyze_spectrogram(y, sr)
                results['analysis_plots']['spectrogram'] = spectrogram_plot
                
            # 3. Frequency Analysis
            if analysis_options.get('frequency', True):
                frequency_plot, freq_stats = self._analyze_frequency(y, sr)
                results['analysis_plots']['frequency'] = frequency_plot
                results['features']['frequency_stats'] = freq_stats
                
            # 4. MFCC Features
            if analysis_options.get('mfcc', True):
                mfcc_plot, mfcc_features = self._analyze_mfcc(y, sr)
                results['analysis_plots']['mfcc'] = mfcc_plot
                results['features']['mfcc'] = mfcc_features
                
            # 5. Chroma Features
            if analysis_options.get('chroma', True):
                chroma_plot, chroma_features = self._analyze_chroma(y, sr)
                results['analysis_plots']['chroma'] = chroma_plot
                results['features']['chroma'] = chroma_features
                
            # 6. Tempo & Beat Analysis
            if analysis_options.get('tempo', True):
                tempo_plot, tempo_features = self._analyze_tempo(y, sr)
                results['analysis_plots']['tempo'] = tempo_plot
                results['features']['tempo'] = tempo_features
                
            # Generate insights based on analysis
            results['insights'] = self._generate_insights(results['features'])
            
            # Store results
            self.analysis_results = results
            
            return {
                'success': True,
                'results': results,
                'message': f'Successfully analyzed {duration:.2f}s audio file'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Audio analysis failed'
            }
    
    def _analyze_waveform(self, y, sr):
        """Analyze and plot waveform"""
        plt.figure(figsize=(12, 6))
        
        # Time axis
        time = np.linspace(0, len(y)/sr, len(y))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(time, y, color='cyan', alpha=0.7, linewidth=0.5)
        plt.title('🎵 Audio Waveform Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot envelope
        plt.subplot(2, 1, 2)
        envelope = np.abs(signal.hilbert(y))
        plt.plot(time, envelope, color='orange', linewidth=1)
        plt.fill_between(time, envelope, alpha=0.3, color='orange')
        plt.title('📊 Amplitude Envelope', fontsize=12)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Envelope')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._plot_to_base64()
        
    def _analyze_spectrogram(self, y, sr):
        """Analyze and plot spectrogram"""
        plt.figure(figsize=(12, 8))
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        plt.subplot(2, 1, 1)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='plasma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('🎼 Spectrogram - Full Frequency Range', fontsize=14, fontweight='bold')
        
        # Mel spectrogram
        plt.subplot(2, 1, 2)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('🎵 Mel Spectrogram', fontsize=12)
        
        plt.tight_layout()
        return self._plot_to_base64()
        
    def _analyze_frequency(self, y, sr):
        """Analyze frequency content"""
        plt.figure(figsize=(12, 8))
        
        # FFT
        fft = np.fft.fft(y)
        magnitude = np.abs(fft)
        frequency = np.fft.fftfreq(len(fft), 1/sr)
        
        # Plot frequency spectrum
        plt.subplot(2, 2, 1)
        plt.plot(frequency[:len(frequency)//2], magnitude[:len(magnitude)//2], 
                color='lime', alpha=0.8)
        plt.title('📊 Frequency Spectrum', fontsize=12, fontweight='bold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
        
        # Spectral centroid
        plt.subplot(2, 2, 2)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        time_frames = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr)
        plt.plot(time_frames, spectral_centroids, color='yellow')
        plt.title('🎯 Spectral Centroid', fontsize=12)
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        plt.grid(True, alpha=0.3)
        
        # Spectral rolloff
        plt.subplot(2, 2, 3)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        plt.plot(time_frames, spectral_rolloff, color='magenta')
        plt.title('📈 Spectral Rolloff', fontsize=12)
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        plt.grid(True, alpha=0.3)
        
        # Zero crossing rate
        plt.subplot(2, 2, 4)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        time_zcr = librosa.frames_to_time(np.arange(len(zcr)), sr=sr)
        plt.plot(time_zcr, zcr, color='red')
        plt.title('⚡ Zero Crossing Rate', fontsize=12)
        plt.xlabel('Time (s)')
        plt.ylabel('ZCR')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate frequency statistics
        freq_stats = {
            'dominant_frequency': float(frequency[np.argmax(magnitude[:len(magnitude)//2])]),
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zcr))
        }
        
        return self._plot_to_base64(), freq_stats
        
    def _analyze_mfcc(self, y, sr):
        """Analyze MFCC features"""
        plt.figure(figsize=(12, 8))
        
        # Compute MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        plt.subplot(2, 1, 1)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')
        plt.colorbar()
        plt.title('🎼 MFCC Features (Mel-frequency Cepstral Coefficients)', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('MFCC Coefficients')
        
        # MFCC statistics plot
        plt.subplot(2, 1, 2)
        mfcc_means = np.mean(mfccs, axis=1)
        plt.bar(range(1, 14), mfcc_means, color='skyblue', alpha=0.7)
        plt.title('📊 Average MFCC Values', fontsize=12)
        plt.xlabel('MFCC Coefficient')
        plt.ylabel('Mean Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        mfcc_features = {
            'mfcc_means': mfcc_means.tolist(),
            'mfcc_stds': np.std(mfccs, axis=1).tolist()
        }
        
        return self._plot_to_base64(), mfcc_features
        
    def _analyze_chroma(self, y, sr):
        """Analyze chroma features"""
        plt.figure(figsize=(12, 6))
        
        # Compute chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        plt.subplot(1, 2, 1)
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap='plasma')
        plt.colorbar()
        plt.title('🎹 Chroma Features', fontsize=14, fontweight='bold')
        
        # Chroma summary
        plt.subplot(1, 2, 2)
        chroma_means = np.mean(chroma, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        plt.bar(notes, chroma_means, color='lightgreen', alpha=0.7)
        plt.title('🎵 Average Chroma Values', fontsize=12)
        plt.xlabel('Note')
        plt.ylabel('Chroma Strength')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Detect dominant key
        dominant_note = notes[np.argmax(chroma_means)]
        
        chroma_features = {
            'chroma_means': chroma_means.tolist(),
            'dominant_note': dominant_note,
            'key_strength': float(np.max(chroma_means))
        }
        
        return self._plot_to_base64(), chroma_features
        
    def _analyze_tempo(self, y, sr):
        """Analyze tempo and beat"""
        plt.figure(figsize=(12, 8))
        
        # Tempo and beat detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Plot beat tracking
        plt.subplot(2, 1, 1)
        time = np.linspace(0, len(y)/sr, len(y))
        plt.plot(time, y, alpha=0.6, color='lightblue')
        plt.vlines(beat_times, -1, 1, color='red', alpha=0.8, linewidth=2)
        plt.title(f'🥁 Beat Tracking - Estimated Tempo: {tempo:.1f} BPM', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Onset detection
        plt.subplot(2, 1, 2)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        plt.plot(time, y, alpha=0.6, color='lightblue')
        plt.vlines(onset_times, -1, 1, color='orange', alpha=0.8, linewidth=1)
        plt.title(f'🎼 Onset Detection - Found {len(onset_times)} onsets', fontsize=12)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        tempo_features = {
            'tempo_bpm': float(tempo),
            'num_beats': len(beat_times),
            'num_onsets': len(onset_times),
            'beat_consistency': float(np.std(np.diff(beat_times))),
            'average_beat_interval': float(np.mean(np.diff(beat_times))) if len(beat_times) > 1 else 0
        }
        
        return self._plot_to_base64(), tempo_features
        
    def _generate_insights(self, features):
        """Generate insights from analysis results"""
        insights = []
        
        # Tempo insights
        if 'tempo' in features:
            tempo = features['tempo']['tempo_bpm']
            if tempo < 60:
                insights.append(f"🐌 Very slow tempo ({tempo:.1f} BPM) - likely ballad or ambient music")
            elif tempo < 90:
                insights.append(f"🚶 Slow tempo ({tempo:.1f} BPM) - possibly ballad, blues, or downtempo")
            elif tempo < 120:
                insights.append(f"🎵 Moderate tempo ({tempo:.1f} BPM) - suitable for pop, rock, or folk")
            elif tempo < 140:
                insights.append(f"🏃 Fast tempo ({tempo:.1f} BPM) - likely dance, rock, or upbeat pop")
            else:
                insights.append(f"⚡ Very fast tempo ({tempo:.1f} BPM) - possibly electronic, metal, or punk")
                
        # Frequency insights
        if 'frequency_stats' in features:
            centroid = features['frequency_stats']['spectral_centroid_mean']
            if centroid < 2000:
                insights.append(f"🎵 Low spectral centroid ({centroid:.0f} Hz) - warm, bass-heavy sound")
            elif centroid > 4000:
                insights.append(f"✨ High spectral centroid ({centroid:.0f} Hz) - bright, treble-heavy sound")
            else:
                insights.append(f"🎶 Balanced spectral centroid ({centroid:.0f} Hz) - well-balanced frequency content")
                
        # Chroma insights
        if 'chroma' in features:
            dominant_note = features['chroma']['dominant_note']
            key_strength = features['chroma']['key_strength']
            insights.append(f"🎹 Dominant note: {dominant_note} (strength: {key_strength:.2f})")
            
        # Dynamic range insights
        if 'amplitude_stats' in features:
            dynamic_range = features['amplitude_stats']['dynamic_range']
            if dynamic_range > 1.5:
                insights.append("📊 High dynamic range - good variation in loudness")
            elif dynamic_range < 0.5:
                insights.append("📊 Low dynamic range - compressed audio")
                
        return insights
        
    def _plot_to_base64(self):
        """Convert current matplotlib plot to base64 string"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor='#2b2b2b', edgecolor='none', dpi=100)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
        
    def export_analysis_report(self, output_dir="static/analysis_reports"):
        """Export analysis results to files"""
        if not self.analysis_results:
            return {'success': False, 'message': 'No analysis results to export'}
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save plots as PNG files
            plot_files = []
            for plot_type, plot_data in self.analysis_results['results']['analysis_plots'].items():
                filename = f"analysis_{plot_type}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(plot_data))
                plot_files.append(filename)
                
            # Create analysis report JSON
            report_file = f"analysis_report_{timestamp}.json"
            report_path = os.path.join(output_dir, report_file)
            
            import json
            with open(report_path, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'features': self.analysis_results['results']['features'],
                    'insights': self.analysis_results['results']['insights'],
                    'file_info': self.analysis_results['results']['file_info'],
                    'plot_files': plot_files
                }, f, indent=2)
                
            return {
                'success': True,
                'report_file': report_file,
                'plot_files': plot_files,
                'message': f'Analysis exported successfully - {len(plot_files)} plots saved'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to export analysis'
            }
