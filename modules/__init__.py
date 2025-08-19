"""
Advanced Audio Processing Modules
Modular audio processing engines for equalizer, noise reduction, genre classification, and real-time processing
"""

__version__ = "1.0.0"
__author__ = "Advanced Audio Processing Team"

from .equalizer_engine import EqualizerEngine
from .noise_reduction_engine import NoiseReductionEngine  
from .genre_classification_engine import GenreClassificationEngine
from .realtime_processing_engine import RealTimeProcessingEngine

__all__ = [
    'EqualizerEngine',
    'NoiseReductionEngine', 
    'GenreClassificationEngine',
    'RealTimeProcessingEngine'
]
