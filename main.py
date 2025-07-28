#!/usr/bin/env python3
"""
Audio Processing Application
Main entry point for the application
"""

import sys
import os
import argparse
from audio_processor import AudioProcessor
from models.train_models import ModelTrainer

def train_models():
    """Train all ML models"""
    print("Training models...")
    trainer = ModelTrainer()
    trainer.train_all_models()
    print("Model training completed!")

def process_single_file(input_file, output_file, bass_gain=1.0, mid_gain=1.0, 
                       treble_gain=1.0, denoise=True):
    """Process a single audio file"""
    print(f"Processing {input_file}...")
    
    processor = AudioProcessor()
    processed_audio, genre, confidence = processor.process_audio_file(
        input_file, bass_gain, mid_gain, treble_gain, denoise
    )
    
    processor.save_audio(processed_audio, output_file)
    
    print(f"Genre: {genre} (Confidence: {confidence:.1%})")
    print(f"Processed audio saved to: {output_file}")

def run_gui():
    """Run the GUI application"""
    try:
        from gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"GUI not available: {e}")
        print("Please install required GUI dependencies")

def main():
    parser = argparse.ArgumentParser(description="Audio Processing Application")
    parser.add_argument("--train", action="store_true", 
                       help="Train ML models")
    parser.add_argument("--gui", action="store_true", 
                       help="Run GUI application")
    parser.add_argument("--input", type=str, 
                       help="Input audio file")
    parser.add_argument("--output", type=str, 
                       help="Output audio file")
    parser.add_argument("--bass", type=float, default=1.0,
                       help="Bass gain (default: 1.0)")
    parser.add_argument("--mid", type=float, default=1.0,
                       help="Mid gain (default: 1.0)")
    parser.add_argument("--treble", type=float, default=1.0,
                       help="Treble gain (default: 1.0)")
    parser.add_argument("--no-denoise", action="store_true",
                       help="Disable noise reduction")
    
    args = parser.parse_args()
    
    if args.train:
        train_models()
    elif args.gui:
        run_gui()
    elif args.input and args.output:
        process_single_file(
            args.input, args.output, 
            args.bass, args.mid, args.treble, 
            not args.no_denoise
        )
    else:
        # Default to GUI if no arguments provided
        print("Starting GUI application...")
        run_gui()

if __name__ == "__main__":
    main() 