#!/usr/bin/env python3
"""
Equalizer Engine Module
Bộ cân bằng âm 10 băng tần với các dải tần số cụ thể và preset
"""

import numpy as np
from scipy.signal import butter, sosfilt, firwin, lfilter
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time
import os

class EqualizerEngine:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize Equalizer Engine.
        
        Args:
            sample_rate: Sample rate for audio processing.
        """
        self.sample_rate = sample_rate
        
        # 10-band equalizer frequency bands (Hz)
        self.frequency_bands = {
            'band_31_hz': 31,
            'band_62_hz': 62,
            'band_125_hz': 125,
            'band_250_hz': 250,
            'band_500_hz': 500,
            'band_1k_hz': 1000,
            'band_2k_hz': 2000,
            'band_4k_hz': 4000,
            'band_8k_hz': 8000,
            'band_16k_hz': 16000
        }
        
        self.presets = self._load_presets()

    def _load_presets(self) -> Dict[str, Dict[str, float]]:
        """Load equalizer presets for the 10 bands."""
        return {
            'flat': {band: 0.0 for band in self.frequency_bands},
            'rock': {'band_31_hz': 4, 'band_62_hz': 2, 'band_125_hz': 1, 'band_250_hz': -2, 'band_500_hz': -3, 'band_1k_hz': -1, 'band_2k_hz': 2, 'band_4k_hz': 5, 'band_8k_hz': 6, 'band_16k_hz': 7},
            'pop': {'band_31_hz': -1, 'band_62_hz': 2, 'band_125_hz': 4, 'band_250_hz': 5, 'band_500_hz': 2, 'band_1k_hz': -1, 'band_2k_hz': -2, 'band_4k_hz': -1, 'band_8k_hz': 1, 'band_16k_hz': 2},
            'classical': {'band_31_hz': 0, 'band_62_hz': 0, 'band_125_hz': 0, 'band_250_hz': 0, 'band_500_hz': 0, 'band_1k_hz': 0, 'band_2k_hz': 0, 'band_4k_hz': 0, 'band_8k_hz': 5, 'band_16k_hz': 5},
            'jazz': {'band_31_hz': 0, 'band_62_hz': 2, 'band_125_hz': 2, 'band_250_hz': -2, 'band_500_hz': -2, 'band_1k_hz': 0, 'band_2k_hz': 3, 'band_4k_hz': 2, 'band_8k_hz': 2, 'band_16k_hz': 1},
            'bass_boost': {'band_31_hz': 9, 'band_62_hz': 7, 'band_125_hz': 5, 'band_250_hz': 3, 'band_500_hz': 1, 'band_1k_hz': 0, 'band_2k_hz': 0, 'band_4k_hz': 0, 'band_8k_hz': 0, 'band_16k_hz': 0},
            'treble_boost': {'band_31_hz': 0, 'band_62_hz': 0, 'band_125_hz': 0, 'band_250_hz': 0, 'band_500_hz': 0, 'band_1k_hz': 1, 'band_2k_hz': 3, 'band_4k_hz': 5, 'band_8k_hz': 7, 'band_16k_hz': 9},
            'vocal_boost': {'band_31_hz': -3, 'band_62_hz': -3, 'band_125_hz': -3, 'band_250_hz': 0, 'band_500_hz': 3, 'band_1k_hz': 5, 'band_2k_hz': 3, 'band_4k_hz': 0, 'band_8k_hz': -2, 'band_16k_hz': -3},
            'electronic': {'band_31_hz': 6, 'band_62_hz': 4, 'band_125_hz': 2, 'band_250_hz': 0, 'band_500_hz': -2, 'band_1k_hz': 0, 'band_2k_hz': 2, 'band_4k_hz': 4, 'band_8k_hz': 6, 'band_16k_hz': 8}
        }

    def _create_peaking_filter(self, center_freq: float, gain_db: float, q_factor: float = 1.41):
        """Creates a peaking (bell) filter for a given frequency band."""
        w0 = 2 * np.pi * center_freq / self.sample_rate
        A = 10**(gain_db / 40.0)
        alpha = np.sin(w0) / (2 * q_factor)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        # Return SOS (Second-Order Sections) format for stability
        return np.array([[b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]])

    def _create_fir_filter(self, band_name: str, center_freq: float, gain_db: float, num_taps: int = 101):
        """
        Creates an FIR filter (low-pass, band-pass, or high-pass) for a given frequency band.
        Gain is applied by scaling the filter coefficients.
        """
        nyquist = 0.5 * self.sample_rate
        freqs = list(self.frequency_bands.values())
        freq_idx = freqs.index(center_freq)
        
        linear_gain = 10**(gain_db / 20.0) # Convert dB to linear gain

        if band_name == 'band_31_hz': # Lowest band: Low-pass
            cutoff_freq = np.sqrt(freqs[freq_idx] * freqs[freq_idx + 1]) / nyquist
            fir_coeffs = firwin(num_taps, cutoff_freq, pass_zero=True)
        elif band_name == 'band_16k_hz': # Highest band: High-pass
            cutoff_freq = np.sqrt(freqs[freq_idx - 1] * freqs[freq_idx]) / nyquist
            fir_coeffs = firwin(num_taps, cutoff_freq, pass_zero=False)
        else: # Middle bands: Band-pass
            low_cutoff = np.sqrt(freqs[freq_idx - 1] * freqs[freq_idx]) / nyquist
            high_cutoff = np.sqrt(freqs[freq_idx] * freqs[freq_idx + 1]) / nyquist
            fir_coeffs = firwin(num_taps, [low_cutoff, high_cutoff], pass_zero=False)
        
        return fir_coeffs * linear_gain # Apply gain by scaling coefficients

    def apply_equalizer(self, audio: np.ndarray, gains: Dict[str, float], filter_type: str = 'iir') -> np.ndarray:
        """
        Apply equalizer using a cascade of IIR peaking filters or FIR filters.
        
        Args:
            audio: Input audio signal.
            gains: Dictionary of frequency band gains in dB.
            filter_type: Type of filter to apply ('iir' for peaking, 'fir' for FIR).
            
        Returns:
            Processed audio signal.
        """
        if not isinstance(audio, np.ndarray):
            raise TypeError("Input audio must be a numpy array.")
            
        processed_audio = audio.copy()

        if filter_type == 'iir':
            for band_name, center_freq in self.frequency_bands.items():
                gain_db = gains.get(band_name, 0.0)
                if gain_db != 0.0:
                    sos_filter = self._create_peaking_filter(center_freq, gain_db)
                    processed_audio = sosfilt(sos_filter, processed_audio)
        elif filter_type == 'fir':
            # For FIR, we create a composite filter by summing the individual band filters
            num_taps = 101
            composite_fir_coeffs = np.zeros(num_taps)
            for band_name, center_freq in self.frequency_bands.items():
                gain_db = gains.get(band_name, 0.0)
                fir_coeffs = self._create_fir_filter(band_name, center_freq, gain_db)
                composite_fir_coeffs += fir_coeffs
            processed_audio = lfilter(composite_fir_coeffs, 1.0, processed_audio)
        else:
            raise ValueError("Invalid filter_type. Must be 'iir' or 'fir'.")
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed_audio))
        if max_val > 1.0:
            processed_audio /= max_val
            
        return processed_audio.astype(np.float32)

    def generate_comparison_plots(self, original_audio: np.ndarray, processed_audio: np.ndarray, options: Dict[str, bool], output_dir: str) -> Dict[str, str]:
        """
        Generate comparison plots (waveform and spectrogram) for original vs. processed audio.

        Args:
            original_audio: The original audio signal.
            processed_audio: The processed audio signal.
            options: A dictionary indicating which plots to generate. e.g., {'displayWaveformPlot': True, 'displaySpectrogramPlot': True}
            output_dir: The directory to save the plot images.

        Returns:
            A dictionary containing the paths to the generated plots.
        """
        plot_paths = {}
        timestamp = int(time.time())
        
        os.makedirs(output_dir, exist_ok=True)

        if options.get('displayWaveformPlot'):
            plt.figure(figsize=(12, 6))
            ax1 = plt.subplot(2, 1, 1)
            librosa.display.waveshow(original_audio, sr=self.sample_rate, alpha=0.8)
            plt.title('Original Waveform')
            plt.xlabel(None)
            plt.ylabel('Amplitude')
            
            plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
            librosa.display.waveshow(processed_audio, sr=self.sample_rate, alpha=0.8, color='r')
            plt.title('Processed Waveform')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            
            plt.tight_layout()
            path = os.path.join(output_dir, f"eq_waveform_{timestamp}.png")
            plt.savefig(path)
            plt.close()
            plot_paths['waveform'] = path.replace(os.path.sep, '/')

        if options.get('displaySpectrogramPlot'):
            plt.figure(figsize=(12, 8))
            
            D_original = librosa.stft(original_audio)
            S_db_original = librosa.amplitude_to_db(np.abs(D_original), ref=np.max)
            ax1 = plt.subplot(2, 1, 1)
            img = librosa.display.specshow(S_db_original, sr=self.sample_rate, x_axis='time', y_axis='log', ax=ax1)
            fig = plt.gcf()
            fig.colorbar(img, ax=ax1, format='%+2.0f dB')
            plt.title('Original Spectrogram')
            
            D_processed = librosa.stft(processed_audio)
            S_db_processed = librosa.amplitude_to_db(np.abs(D_processed), ref=np.max)
            ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
            img2 = librosa.display.specshow(S_db_processed, sr=self.sample_rate, x_axis='time', y_axis='log', ax=ax2)
            fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
            plt.title('Processed Spectrogram')
            
            plt.tight_layout()
            path = os.path.join(output_dir, f"eq_spectrogram_{timestamp}.png")
            plt.savefig(path)
            plt.close()
            plot_paths['spectrogram'] = path.replace(os.path.sep, '/')
            
        return plot_paths

    def apply_preset(self, audio: np.ndarray, preset_name: str, filter_type: str = 'iir') -> np.ndarray:
        """
        Apply an equalizer preset to the audio.
        
        Args:
            audio: Input audio signal.
            preset_name: Name of the preset to apply.
            filter_type: Type of filter to apply ('iir' for peaking, 'fir' for FIR).
            
        Returns:
            Processed audio signal.
        """
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        gains = self.presets[preset_name]
        return self.apply_equalizer(audio, gains, filter_type)

    def get_frequency_response(self, gains: Dict[str, float], num_points: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the frequency response of the current EQ settings.
        NOTE: This method currently only supports IIR (peaking) filters.
        
        Args:
            gains: Equalizer gains in dB.
            num_points: Number of frequency points to calculate.
            
        Returns:
            Tuple of (frequencies, response_in_db).
        """
        from scipy.signal import sosfreqz
        
        total_response = np.ones(num_points, dtype=np.complex128)
        
        for band_name, center_freq in self.frequency_bands.items():
            gain_db = gains.get(band_name, 0.0)
            if gain_db != 0.0:
                sos_filter = self._create_peaking_filter(center_freq, gain_db)
                w, h = sosfreqz(sos_filter, worN=num_points, fs=self.sample_rate)
                total_response *= h

        freqs = w
        response_db = 20 * np.log10(np.abs(total_response) + 1e-9)
        
        return freqs, response_db

    def get_available_presets(self) -> List[str]:
        """Get a list of available preset names."""
        return list(self.presets.keys())

    def get_preset_gains(self, preset_name: str) -> Dict[str, float]:
        """Get the gain values for a specific preset."""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self.presets[preset_name].copy()