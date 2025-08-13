#!/usr/bin/env python3
"""
Equalizer Engine Module
Bộ cân bằng âm 10 băng tần với các dải tần số cụ thể và preset
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict, List, Tuple, Optional
import json
import os

class EqualizerEngine:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize Equalizer Engine
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
        # 10-band equalizer frequency bands (Hz)
        self.frequency_bands = {
            'sub_bass': 60,      # 60Hz
            'bass': 170,         # 170Hz  
            'low_mid': 310,      # 310Hz
            'mid': 600,          # 600Hz
            'high_mid': 1000,    # 1kHz
            'presence': 3000,    # 3kHz
            'brilliance': 6000,  # 6kHz
            'air': 12000,        # 12kHz
            'ultra_high': 14000, # 14kHz
            'extreme': 16000     # 16kHz
        }
        
        # Load presets
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, Dict[str, float]]:
        """Load equalizer presets"""
        presets = {
            'flat': {
                'sub_bass': 0.0, 'bass': 0.0, 'low_mid': 0.0, 'mid': 0.0,
                'high_mid': 0.0, 'presence': 0.0, 'brilliance': 0.0, 
                'air': 0.0, 'ultra_high': 0.0, 'extreme': 0.0
            },
            'rock': {
                'sub_bass': 5.0, 'bass': 3.0, 'low_mid': -2.0, 'mid': 0.0,
                'high_mid': 2.0, 'presence': 4.0, 'brilliance': 6.0, 
                'air': 3.0, 'ultra_high': 2.0, 'extreme': 1.0
            },
            'pop': {
                'sub_bass': 2.0, 'bass': 1.0, 'low_mid': 0.0, 'mid': 1.0,
                'high_mid': 2.0, 'presence': 3.0, 'brilliance': 4.0, 
                'air': 3.0, 'ultra_high': 2.0, 'extreme': 1.0
            },
            'classical': {
                'sub_bass': 0.0, 'bass': 0.0, 'low_mid': 0.0, 'mid': 0.0,
                'high_mid': 0.0, 'presence': 1.0, 'brilliance': 2.0, 
                'air': 3.0, 'ultra_high': 2.0, 'extreme': 1.0
            },
            'jazz': {
                'sub_bass': 1.0, 'bass': 2.0, 'low_mid': 1.0, 'mid': 1.0,
                'high_mid': 0.0, 'presence': 1.0, 'brilliance': 2.0, 
                'air': 2.0, 'ultra_high': 1.0, 'extreme': 0.0
            },
            'bass_boost': {
                'sub_bass': 8.0, 'bass': 6.0, 'low_mid': 4.0, 'mid': 2.0,
                'high_mid': 0.0, 'presence': -1.0, 'brilliance': -1.0, 
                'air': 0.0, 'ultra_high': 0.0, 'extreme': 0.0
            },
            'vocal': {
                'sub_bass': -2.0, 'bass': -1.0, 'low_mid': 1.0, 'mid': 3.0,
                'high_mid': 4.0, 'presence': 5.0, 'brilliance': 3.0, 
                'air': 2.0, 'ultra_high': 1.0, 'extreme': 0.0
            },
            'dance': {
                'sub_bass': 6.0, 'bass': 4.0, 'low_mid': 1.0, 'mid': 0.0,
                'high_mid': 1.0, 'presence': 2.0, 'brilliance': 4.0, 
                'air': 4.0, 'ultra_high': 3.0, 'extreme': 2.0
            }
        }
        return presets
    
    def db_to_linear(self, db_gain: float) -> float:
        """Convert dB gain to linear gain"""
        return 10.0 ** (db_gain / 20.0)
    
    def linear_to_db(self, linear_gain: float) -> float:
        """Convert linear gain to dB gain"""
        return 20.0 * np.log10(max(linear_gain, 1e-10))
    
    def create_frequency_response(self, audio_length: int, gains: Dict[str, float]) -> np.ndarray:
        """
        Create frequency response curve for equalizer
        
        Args:
            audio_length: Length of audio signal
            gains: Dictionary of frequency band gains in dB
            
        Returns:
            Complex frequency response array
        """
        freqs = np.fft.fftfreq(audio_length, 1/self.sample_rate)
        freq_response = np.ones_like(freqs, dtype=complex)
        
        # Apply gains for each frequency band
        for band_name, center_freq in self.frequency_bands.items():
            if band_name in gains:
                gain_db = gains[band_name]
                gain_linear = self.db_to_linear(gain_db)
                
                # Define bandwidth based on frequency
                if center_freq <= 100:
                    bandwidth = center_freq * 1.5
                elif center_freq <= 1000:
                    bandwidth = center_freq * 0.8
                else:
                    bandwidth = center_freq * 0.6
                
                # Create bell filter response
                freq_mask = (np.abs(freqs) >= center_freq - bandwidth/2) & \
                           (np.abs(freqs) <= center_freq + bandwidth/2)
                
                # Apply smooth transition using Gaussian-like curve
                for i, freq in enumerate(freqs):
                    if freq_mask[i]:
                        distance = abs(abs(freq) - center_freq)
                        weight = np.exp(-2 * (distance / bandwidth) ** 2)
                        freq_response[i] *= (1 + (gain_linear - 1) * weight)
        
        return freq_response
    
    def apply_equalizer_fft(self, audio: np.ndarray, gains: Dict[str, float]) -> np.ndarray:
        """
        Apply equalizer using FFT method
        
        Args:
            audio: Input audio signal
            gains: Dictionary of frequency band gains in dB (-20 to +20)
            
        Returns:
            Processed audio signal
        """
        if len(audio) == 0:
            return audio
        
        # Apply FFT
        fft_audio = np.fft.fft(audio)
        
        # Create frequency response
        freq_response = self.create_frequency_response(len(audio), gains)
        
        # Apply frequency response
        processed_fft = fft_audio * freq_response
        
        # Convert back to time domain
        processed_audio = np.real(np.fft.ifft(processed_fft))
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed_audio))
        if max_val > 1.0:
            processed_audio = processed_audio / max_val * 0.95
            
        return processed_audio.astype(np.float32)
    
    def apply_equalizer_filter(self, audio: np.ndarray, gains: Dict[str, float]) -> np.ndarray:
        """
        Apply equalizer using cascaded IIR filters (more CPU intensive but higher quality)
        
        Args:
            audio: Input audio signal
            gains: Dictionary of frequency band gains in dB
            
        Returns:
            Processed audio signal
        """
        processed_audio = audio.copy()
        
        for band_name, center_freq in self.frequency_bands.items():
            if band_name in gains and gains[band_name] != 0.0:
                gain_db = gains[band_name]
                gain_linear = self.db_to_linear(gain_db)
                
                # Calculate Q factor based on frequency
                if center_freq <= 100:
                    Q = 0.7
                elif center_freq <= 1000:
                    Q = 1.0
                else:
                    Q = 1.2
                
                # Design peaking filter
                processed_audio = self._apply_peaking_filter(
                    processed_audio, center_freq, Q, gain_linear
                )
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed_audio))
        if max_val > 1.0:
            processed_audio = processed_audio / max_val * 0.95
            
        return processed_audio.astype(np.float32)
    
    def _apply_peaking_filter(self, audio: np.ndarray, center_freq: float, 
                             Q: float, gain: float) -> np.ndarray:
        """Apply peaking filter for specific frequency band"""
        nyquist = self.sample_rate / 2
        w0 = 2 * np.pi * center_freq / self.sample_rate
        
        if gain > 1.0:  # Boost
            A = np.sqrt(gain)
            alpha = np.sin(w0) / (2 * Q)
            
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
        elif gain < 1.0:  # Cut
            A = np.sqrt(1/gain)
            alpha = np.sin(w0) / (2 * Q)
            
            b0 = 1 + alpha / A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha / A
            a0 = 1 + alpha * A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha * A
        else:  # No change
            return audio
        
        # Normalize coefficients
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1/a0, a2/a0])
        
        # Apply filter
        return filtfilt(b, a, audio)
    
    def apply_preset(self, audio: np.ndarray, preset_name: str, 
                    method: str = 'fft') -> np.ndarray:
        """
        Apply equalizer preset to audio
        
        Args:
            audio: Input audio signal
            preset_name: Name of preset to apply
            method: 'fft' or 'filter' processing method
            
        Returns:
            Processed audio signal
        """
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        gains = self.presets[preset_name]
        
        if method == 'fft':
            return self.apply_equalizer_fft(audio, gains)
        elif method == 'filter':
            return self.apply_equalizer_filter(audio, gains)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def process_audio_file(self, input_path: str, output_path: str,
                          gains: Dict[str, float] = None, 
                          preset_name: str = None,
                          method: str = 'fft') -> Dict:
        """
        Process audio file with equalizer
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            gains: Custom equalizer gains (dB)
            preset_name: Name of preset to use
            method: Processing method ('fft' or 'filter')
            
        Returns:
            Dictionary with processing results
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate)
        original_rms = np.sqrt(np.mean(audio ** 2))
        
        # Apply equalizer
        if preset_name:
            processed_audio = self.apply_preset(audio, preset_name, method)
            used_gains = self.presets[preset_name]
        elif gains:
            if method == 'fft':
                processed_audio = self.apply_equalizer_fft(audio, gains)
            else:
                processed_audio = self.apply_equalizer_filter(audio, gains)
            used_gains = gains
        else:
            processed_audio = audio  # No processing
            used_gains = self.presets['flat']
        
        processed_rms = np.sqrt(np.mean(processed_audio ** 2))
        
        # Save processed audio
        sf.write(output_path, processed_audio, self.sample_rate)
        
        return {
            'input_path': input_path,
            'output_path': output_path,
            'original_rms': float(original_rms),
            'processed_rms': float(processed_rms),
            'gain_change_db': float(self.linear_to_db(processed_rms / max(original_rms, 1e-10))),
            'equalizer_gains': used_gains,
            'method': method,
            'preset_used': preset_name if preset_name else 'custom',
            'sample_rate': self.sample_rate,
            'duration': len(processed_audio) / self.sample_rate
        }
    
    def get_frequency_response(self, gains: Dict[str, float], 
                              num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency response curve for visualization
        
        Args:
            gains: Equalizer gains in dB
            num_points: Number of frequency points to calculate
            
        Returns:
            Tuple of (frequencies, response_db)
        """
        # Create frequency array (log scale)
        freqs = np.logspace(1, np.log10(self.sample_rate/2), num_points)
        response = np.ones_like(freqs)
        
        for band_name, center_freq in self.frequency_bands.items():
            if band_name in gains:
                gain_db = gains[band_name]
                gain_linear = self.db_to_linear(gain_db)
                
                # Define bandwidth
                if center_freq <= 100:
                    bandwidth = center_freq * 1.5
                elif center_freq <= 1000:
                    bandwidth = center_freq * 0.8
                else:
                    bandwidth = center_freq * 0.6
                
                # Apply bell filter response
                for i, freq in enumerate(freqs):
                    distance = abs(freq - center_freq)
                    if distance <= bandwidth:
                        weight = np.exp(-2 * (distance / bandwidth) ** 2)
                        response[i] *= (1 + (gain_linear - 1) * weight)
        
        response_db = self.linear_to_db(response)
        return freqs, response_db
    
    def get_available_presets(self) -> List[str]:
        """Get list of available preset names"""
        return list(self.presets.keys())
    
    def get_preset_gains(self, preset_name: str) -> Dict[str, float]:
        """Get gains for a specific preset"""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self.presets[preset_name].copy()
    
    def save_custom_preset(self, name: str, gains: Dict[str, float]) -> bool:
        """Save custom preset to file"""
        try:
            # Add to current presets
            self.presets[name] = gains.copy()
            
            # Save to file (if desired)
            presets_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'custom_presets.json')
            os.makedirs(os.path.dirname(presets_file), exist_ok=True)
            
            with open(presets_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving preset: {e}")
            return False
