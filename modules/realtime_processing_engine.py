#!/usr/bin/env python3
"""
Real-time Processing Engine Module
Xử lý âm thanh thời gian thực với độ trễ thấp
"""

import numpy as np
import pyaudio
import sounddevice as sd
import threading
import time
import queue
from typing import Dict, Callable, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class RealTimeProcessingEngine:
    def __init__(self, sample_rate: int = 22050, 
                 chunk_size: int = 1024,
                 channels: int = 1):
        """
        Initialize Real-time Processing Engine
        
        Args:
            sample_rate: Sample rate for audio processing
            chunk_size: Size of audio chunks for processing
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paFloat32
        
        # Processing state
        self.is_processing = False
        self.is_recording = False
        
        # Audio buffers
        self.input_buffer = queue.Queue()
        self.output_buffer = queue.Queue()
        
        # Processing modules (injected dependencies)
        self.equalizer_engine = None
        self.noise_reduction_engine = None
        self.genre_classification_engine = None
        
        # Processing parameters
        self.equalizer_params = {
            'sub_bass': 0.0, 'bass': 0.0, 'low_mid': 0.0, 'mid': 0.0,
            'high_mid': 0.0, 'presence': 0.0, 'brilliance': 0.0, 
            'air': 0.0, 'ultra_high': 0.0, 'extreme': 0.0
        }
        self.noise_reduction_params = {
            'method': 'spectral',
            'reduction_level': 0.5
        }
        self.processing_enabled = {
            'equalizer': True,
            'noise_reduction': True,
            'genre_classification': True
        }
        
        # PyAudio instance
        self.p = None
        self.stream = None
        
        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'total_latency': 0.0,
            'avg_latency': 0.0,
            'max_latency': 0.0,
            'processing_errors': 0
        }
        
        # Callback for processed audio
        self.audio_callback = None
        self.genre_callback = None
    
    def set_processing_modules(self, 
                             equalizer_engine=None,
                             noise_reduction_engine=None, 
                             genre_classification_engine=None):
        """
        Set processing modules for real-time processing
        
        Args:
            equalizer_engine: Equalizer engine instance
            noise_reduction_engine: Noise reduction engine instance
            genre_classification_engine: Genre classification engine instance
        """
        if equalizer_engine:
            self.equalizer_engine = equalizer_engine
        if noise_reduction_engine:
            self.noise_reduction_engine = noise_reduction_engine
        if genre_classification_engine:
            self.genre_classification_engine = genre_classification_engine
    
    def set_equalizer_params(self, params: Dict[str, float]):
        """Set equalizer parameters for real-time processing"""
        self.equalizer_params.update(params)
    
    def set_noise_reduction_params(self, method: str = 'spectral', 
                                  reduction_level: float = 0.5):
        """Set noise reduction parameters for real-time processing"""
        self.noise_reduction_params = {
            'method': method,
            'reduction_level': reduction_level
        }
    
    def enable_processing(self, equalizer: bool = True, 
                         noise_reduction: bool = True,
                         genre_classification: bool = True):
        """Enable/disable specific processing modules"""
        self.processing_enabled = {
            'equalizer': equalizer,
            'noise_reduction': noise_reduction,
            'genre_classification': genre_classification
        }
    
    def set_callbacks(self, audio_callback: Callable = None, 
                     genre_callback: Callable = None):
        """
        Set callback functions for processed audio
        
        Args:
            audio_callback: Function to call with processed audio chunks
            genre_callback: Function to call with genre classification results
        """
        self.audio_callback = audio_callback
        self.genre_callback = genre_callback
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process single audio chunk with enabled modules
        
        Args:
            audio_chunk: Input audio chunk
            
        Returns:
            Processed audio chunk
        """
        start_time = time.time()
        processed_chunk = audio_chunk.copy()
        
        try:
            # Apply equalizer
            if (self.processing_enabled['equalizer'] and 
                self.equalizer_engine is not None):
                processed_chunk = self.equalizer_engine.apply_equalizer_fft(
                    processed_chunk, self.equalizer_params
                )
            
            # Apply noise reduction  
            if (self.processing_enabled['noise_reduction'] and 
                self.noise_reduction_engine is not None):
                processed_chunk = self.noise_reduction_engine.reduce_noise(
                    processed_chunk, 
                    self.noise_reduction_params['method'],
                    self.noise_reduction_params['reduction_level']
                )
            
            # Genre classification (runs in background, doesn't affect audio)
            if (self.processing_enabled['genre_classification'] and 
                self.genre_classification_engine is not None and
                self.genre_callback is not None):
                
                # Only classify every 10th chunk to reduce CPU load
                if self.stats['chunks_processed'] % 10 == 0:
                    try:
                        genre, confidence, info = self.genre_classification_engine.classify_genre(
                            processed_chunk, method='rule_based'  # Fast method for real-time
                        )
                        self.genre_callback({
                            'genre': genre,
                            'confidence': confidence,
                            'chunk_number': self.stats['chunks_processed']
                        })
                    except Exception as e:
                        print(f"⚠️ Genre classification error: {e}")
            
            # Update statistics
            latency = time.time() - start_time
            self.stats['total_latency'] += latency
            self.stats['chunks_processed'] += 1
            self.stats['avg_latency'] = self.stats['total_latency'] / self.stats['chunks_processed']
            self.stats['max_latency'] = max(self.stats['max_latency'], latency)
            
        except Exception as e:
            print(f"⚠️ Audio processing error: {e}")
            self.stats['processing_errors'] += 1
            # Return original audio on error
            processed_chunk = audio_chunk
        
        return processed_chunk
    
    def _audio_callback_pyaudio(self, in_data, frame_count, time_info, status):
        """PyAudio callback function"""
        if status:
            print(f"PyAudio status: {status}")
        
        try:
            # Convert input data to numpy array
            audio_chunk = np.frombuffer(in_data, dtype=np.float32)
            
            # Process audio chunk
            processed_chunk = self._process_audio_chunk(audio_chunk)
            
            # Call user callback if provided
            if self.audio_callback:
                self.audio_callback(processed_chunk)
            
            # Return processed audio for output
            return (processed_chunk.tobytes(), pyaudio.paContinue)
            
        except Exception as e:
            print(f"⚠️ PyAudio callback error: {e}")
            return (in_data, pyaudio.paContinue)
    
    def _audio_callback_sounddevice(self, indata, outdata, frames, time, status):
        """SoundDevice callback function"""
        if status:
            print(f"SoundDevice status: {status}")
        
        try:
            # Get input audio
            audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
            
            # Process audio chunk
            processed_chunk = self._process_audio_chunk(audio_chunk)
            
            # Call user callback if provided
            if self.audio_callback:
                self.audio_callback(processed_chunk)
            
            # Output processed audio
            if outdata.ndim > 1:
                outdata[:, 0] = processed_chunk
                if outdata.shape[1] > 1:  # Stereo output
                    outdata[:, 1] = processed_chunk
            else:
                outdata[:] = processed_chunk
                
        except Exception as e:
            print(f"⚠️ SoundDevice callback error: {e}")
            # Pass through original audio on error
            if outdata.ndim > 1:
                outdata[:, 0] = indata[:, 0] if indata.ndim > 1 else indata
                if outdata.shape[1] > 1:
                    outdata[:, 1] = indata[:, 0] if indata.ndim > 1 else indata
            else:
                outdata[:] = indata[:, 0] if indata.ndim > 1 else indata
    
    def start_processing_pyaudio(self, input_device=None, output_device=None):
        """
        Start real-time processing using PyAudio
        
        Args:
            input_device: Input device index (None for default)
            output_device: Output device index (None for default)
        """
        if self.is_processing:
            print("⚠️ Processing already started")
            return False
        
        try:
            self.p = pyaudio.PyAudio()
            
            # Reset statistics
            self.stats = {
                'chunks_processed': 0,
                'total_latency': 0.0,
                'avg_latency': 0.0,
                'max_latency': 0.0,
                'processing_errors': 0
            }
            
            # Open stream
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                output=True,
                input_device_index=input_device,
                output_device_index=output_device,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback_pyaudio
            )
            
            self.stream.start_stream()
            self.is_processing = True
            
            print(f"✓ Real-time processing started with PyAudio")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Chunk size: {self.chunk_size}")
            print(f"  Channels: {self.channels}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error starting PyAudio processing: {e}")
            self.stop_processing()
            return False
    
    def start_processing_sounddevice(self, input_device=None, output_device=None):
        """
        Start real-time processing using SoundDevice
        
        Args:
            input_device: Input device index (None for default)
            output_device: Output device index (None for default)
        """
        if self.is_processing:
            print("⚠️ Processing already started")
            return False
        
        try:
            # Reset statistics
            self.stats = {
                'chunks_processed': 0,
                'total_latency': 0.0,
                'avg_latency': 0.0,
                'max_latency': 0.0,
                'processing_errors': 0
            }
            
            # Start stream
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback_sounddevice,
                blocksize=self.chunk_size,
                device=(input_device, output_device),
                dtype='float32',
                latency='low'
            )
            
            self.stream.start()
            self.is_processing = True
            
            print(f"✓ Real-time processing started with SoundDevice")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Chunk size: {self.chunk_size}")
            print(f"  Channels: {self.channels}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error starting SoundDevice processing: {e}")
            self.stop_processing()
            return False
    
    def stop_processing(self):
        """Stop real-time processing"""
        try:
            self.is_processing = False
            
            if self.stream:
                if hasattr(self.stream, 'stop_stream'):  # PyAudio
                    self.stream.stop_stream()
                    self.stream.close()
                else:  # SoundDevice
                    self.stream.stop()
                    self.stream.close()
                self.stream = None
            
            if self.p:  # PyAudio
                self.p.terminate()
                self.p = None
            
            print("✓ Real-time processing stopped")
            
        except Exception as e:
            print(f"⚠️ Error stopping processing: {e}")
    
    def start_recording_to_file(self, filename: str, duration: float = None):
        """
        Start recording processed audio to file
        
        Args:
            filename: Output filename
            duration: Recording duration in seconds (None for continuous)
        """
        if self.is_recording:
            print("⚠️ Recording already started")
            return False
        
        try:
            import soundfile as sf
            
            self.is_recording = True
            self.recorded_audio = []
            self.recording_start_time = time.time()
            self.recording_duration = duration
            self.recording_filename = filename
            
            # Set callback to collect audio
            def record_callback(audio_chunk):
                if self.is_recording:
                    self.recorded_audio.append(audio_chunk.copy())
                    
                    # Stop if duration reached
                    if (self.recording_duration and 
                        time.time() - self.recording_start_time >= self.recording_duration):
                        self.stop_recording()
            
            self.set_callbacks(audio_callback=record_callback)
            
            print(f"✓ Recording started to {filename}")
            if duration:
                print(f"  Duration: {duration} seconds")
            else:
                print("  Duration: continuous (call stop_recording() to stop)")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error starting recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording and save audio to file"""
        if not self.is_recording:
            print("⚠️ No recording in progress")
            return False
        
        try:
            import soundfile as sf
            
            self.is_recording = False
            
            if self.recorded_audio:
                # Concatenate all recorded chunks
                full_audio = np.concatenate(self.recorded_audio)
                
                # Save to file
                sf.write(self.recording_filename, full_audio, self.sample_rate)
                
                duration = len(full_audio) / self.sample_rate
                print(f"✓ Recording saved to {self.recording_filename}")
                print(f"  Duration: {duration:.2f} seconds")
                print(f"  Chunks: {len(self.recorded_audio)}")
                
                return True
            else:
                print("⚠️ No audio recorded")
                return False
                
        except Exception as e:
            print(f"⚠️ Error saving recording: {e}")
            return False
    
    def get_processing_stats(self) -> Dict:
        """Get real-time processing statistics"""
        stats = self.stats.copy()
        stats.update({
            'is_processing': self.is_processing,
            'is_recording': self.is_recording,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'enabled_modules': self.processing_enabled,
            'target_latency_ms': (self.chunk_size / self.sample_rate) * 1000,
            'avg_latency_ms': self.stats['avg_latency'] * 1000,
            'max_latency_ms': self.stats['max_latency'] * 1000
        })
        return stats
    
    def get_audio_devices(self) -> Dict:
        """Get available audio input/output devices"""
        devices = {'input': [], 'output': []}
        
        try:
            # Try SoundDevice first (more reliable)
            sd_devices = sd.query_devices()
            for i, device in enumerate(sd_devices):
                device_info = {
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'] + device['max_output_channels'],
                    'sample_rate': device['default_samplerate']
                }
                
                if device['max_input_channels'] > 0:
                    devices['input'].append(device_info)
                if device['max_output_channels'] > 0:
                    devices['output'].append(device_info)
        except:
            pass
        
        try:
            # Fallback to PyAudio
            if not devices['input'] and not devices['output']:
                p = pyaudio.PyAudio()
                
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    device_info = {
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'] + info['maxOutputChannels'],
                        'sample_rate': info['defaultSampleRate']
                    }
                    
                    if info['maxInputChannels'] > 0:
                        devices['input'].append(device_info)
                    if info['maxOutputChannels'] > 0:
                        devices['output'].append(device_info)
                
                p.terminate()
        except:
            pass
        
        return devices
    
    def test_latency(self, duration: float = 5.0) -> Dict:
        """
        Test processing latency
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            Latency test results
        """
        print(f"Testing latency for {duration} seconds...")
        
        # Reset stats
        self.stats = {
            'chunks_processed': 0,
            'total_latency': 0.0,
            'avg_latency': 0.0,
            'max_latency': 0.0,
            'processing_errors': 0
        }
        
        # Start processing
        if not self.start_processing_sounddevice():
            return {'error': 'Failed to start processing'}
        
        # Wait for test duration
        time.sleep(duration)
        
        # Stop processing
        self.stop_processing()
        
        # Calculate results
        results = {
            'test_duration': duration,
            'chunks_processed': self.stats['chunks_processed'],
            'avg_latency_ms': self.stats['avg_latency'] * 1000,
            'max_latency_ms': self.stats['max_latency'] * 1000,
            'target_latency_ms': (self.chunk_size / self.sample_rate) * 1000,
            'processing_errors': self.stats['processing_errors'],
            'chunks_per_second': self.stats['chunks_processed'] / duration,
            'real_time_factor': (self.stats['chunks_processed'] * self.chunk_size / self.sample_rate) / duration
        }
        
        print(f"✓ Latency test completed:")
        print(f"  Average latency: {results['avg_latency_ms']:.1f} ms")
        print(f"  Maximum latency: {results['max_latency_ms']:.1f} ms")
        print(f"  Target latency: {results['target_latency_ms']:.1f} ms")
        print(f"  Processing errors: {results['processing_errors']}")
        
        return results
