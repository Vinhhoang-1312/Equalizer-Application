import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import soundfile as sf
from audio_processor import AudioProcessor
import os
import time
import librosa

class AudioProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Processing Application")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.bass_gain = tk.DoubleVar(value=1.0)
        self.mid_gain = tk.DoubleVar(value=1.0)
        self.treble_gain = tk.DoubleVar(value=1.0)
        self.denoise_var = tk.BooleanVar(value=True)
        self.real_time_var = tk.BooleanVar(value=False)
        self.genre_result = tk.StringVar(value="Unknown")
        self.confidence_result = tk.StringVar(value="0.0%")
        
        # Audio data
        self.original_audio = None
        self.processed_audio = None
        self.is_playing = False
        self.is_recording = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Audio Processing Application", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # File processing tab
        file_frame = ttk.Frame(notebook)
        notebook.add(file_frame, text="File Processing")
        self.setup_file_processing_tab(file_frame)
        
        # Real-time processing tab
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="Real-time Processing")
        self.setup_realtime_tab(realtime_frame)
        
        # Visualization tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Visualization")
        self.setup_visualization_tab(viz_frame)
    
    def setup_file_processing_tab(self, parent):
        """Setup file processing tab"""
        # File selection frame
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Input file
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.input_file, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Output file
        ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_output_file).grid(row=1, column=2, padx=5, pady=5)
        
        # Equalizer frame
        eq_frame = ttk.LabelFrame(parent, text="Equalizer Settings", padding=10)
        eq_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Bass control
        ttk.Label(eq_frame, text="Bass Gain:").grid(row=0, column=0, sticky=tk.W, pady=5)
        bass_scale = ttk.Scale(eq_frame, from_=0.0, to=3.0, variable=self.bass_gain, 
                              orient=tk.HORIZONTAL, length=200)
        bass_scale.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(eq_frame, textvariable=tk.StringVar(value="1.0")).grid(row=0, column=2, padx=5, pady=5)
        
        # Mid control
        ttk.Label(eq_frame, text="Mid Gain:").grid(row=1, column=0, sticky=tk.W, pady=5)
        mid_scale = ttk.Scale(eq_frame, from_=0.0, to=3.0, variable=self.mid_gain, 
                             orient=tk.HORIZONTAL, length=200)
        mid_scale.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(eq_frame, textvariable=tk.StringVar(value="1.0")).grid(row=1, column=2, padx=5, pady=5)
        
        # Treble control
        ttk.Label(eq_frame, text="Treble Gain:").grid(row=2, column=0, sticky=tk.W, pady=5)
        treble_scale = ttk.Scale(eq_frame, from_=0.0, to=3.0, variable=self.treble_gain, 
                                orient=tk.HORIZONTAL, length=200)
        treble_scale.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(eq_frame, textvariable=tk.StringVar(value="1.0")).grid(row=2, column=2, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(parent, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(options_frame, text="Apply Noise Reduction", 
                       variable=self.denoise_var).pack(anchor=tk.W, pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Process Audio", 
                  command=self.process_audio_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Play Original", 
                  command=self.play_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Play Processed", 
                  command=self.play_processed).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", 
                  command=self.stop_audio).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Results", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(results_frame, text="Genre:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(results_frame, textvariable=self.genre_result, 
                 font=('Arial', 12, 'bold')).grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        ttk.Label(results_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(results_frame, textvariable=self.confidence_result, 
                 font=('Arial', 12, 'bold')).grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
    
    def setup_realtime_tab(self, parent):
        """Setup real-time processing tab"""
        # Real-time controls
        control_frame = ttk.LabelFrame(parent, text="Real-time Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Start Recording", 
                  command=self.start_realtime_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Recording", 
                  command=self.stop_realtime_processing).pack(side=tk.LEFT, padx=5)
        
        # Real-time equalizer
        eq_frame = ttk.LabelFrame(parent, text="Real-time Equalizer", padding=10)
        eq_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Bass control
        ttk.Label(eq_frame, text="Bass Gain:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Scale(eq_frame, from_=0.0, to=3.0, variable=self.bass_gain, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=5)
        
        # Mid control
        ttk.Label(eq_frame, text="Mid Gain:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(eq_frame, from_=0.0, to=3.0, variable=self.mid_gain, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5, pady=5)
        
        # Treble control
        ttk.Label(eq_frame, text="Treble Gain:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Scale(eq_frame, from_=0.0, to=3.0, variable=self.treble_gain, 
                 orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, padx=5, pady=5)
        
        # Real-time results
        results_frame = ttk.LabelFrame(parent, text="Real-time Results", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.realtime_genre = tk.StringVar(value="Unknown")
        self.realtime_confidence = tk.StringVar(value="0.0%")
        
        ttk.Label(results_frame, text="Current Genre:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(results_frame, textvariable=self.realtime_genre, 
                 font=('Arial', 14, 'bold')).grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        ttk.Label(results_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(results_frame, textvariable=self.realtime_confidence, 
                 font=('Arial', 14, 'bold')).grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
    
    def setup_visualization_tab(self, parent):
        """Setup visualization tab"""
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Update Visualization", 
                  command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear Plots", 
                  command=self.clear_plots).pack(side=tk.LEFT, padx=5)
    
    def browse_input_file(self):
        """Browse for input audio file"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            # Auto-set output filename
            base_name = os.path.splitext(filename)[0]
            self.output_file.set(f"{base_name}_processed.wav")
    
    def browse_output_file(self):
        """Browse for output audio file"""
        filename = filedialog.asksaveasfilename(
            title="Save Processed Audio",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
    
    def process_audio_file(self):
        """Process audio file with all features"""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        
        if not self.output_file.get():
            messagebox.showerror("Error", "Please select an output file")
            return
        
        try:
            # Process audio in separate thread
            def process_thread():
                try:
                    # Process audio
                    processed_audio, genre, confidence = self.audio_processor.process_audio_file(
                        self.input_file.get(),
                        self.bass_gain.get(),
                        self.mid_gain.get(),
                        self.treble_gain.get(),
                        self.denoise_var.get()
                    )
                    
                    # Save processed audio
                    self.audio_processor.save_audio(processed_audio, self.output_file.get())
                    
                    # Update results
                    self.genre_result.set(genre.title())
                    self.confidence_result.set(f"{confidence:.1%}")
                    
                    # Store audio data for playback
                    self.original_audio, _ = librosa.load(self.input_file.get(), sr=22050)
                    self.processed_audio = processed_audio
                    
                    messagebox.showinfo("Success", "Audio processing completed!")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Processing failed: {str(e)}")
            
            threading.Thread(target=process_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
    
    def play_original(self):
        """Play original audio"""
        if self.original_audio is not None:
            self.stop_audio()
            self.is_playing = True
            sd.play(self.original_audio, 22050)
    
    def play_processed(self):
        """Play processed audio"""
        if self.processed_audio is not None:
            self.stop_audio()
            self.is_playing = True
            sd.play(self.processed_audio, 22050)
    
    def stop_audio(self):
        """Stop audio playback"""
        sd.stop()
        self.is_playing = False
    
    def start_realtime_processing(self):
        """Start real-time audio processing"""
        if self.is_recording:
            return
        
        self.is_recording = True
        
        def realtime_callback(audio_chunk):
            if not self.is_recording:
                return
            
            # Classify genre
            genre, confidence = self.audio_processor.classify_genre(audio_chunk)
            
            # Update UI
            self.realtime_genre.set(genre.title())
            self.realtime_confidence.set(f"{confidence:.1%}")
        
        # Start real-time processing in separate thread
        def realtime_thread():
            self.audio_processor.start_real_time_processing(realtime_callback)
        
        threading.Thread(target=realtime_thread, daemon=True).start()
    
    def stop_realtime_processing(self):
        """Stop real-time audio processing"""
        self.is_recording = False
        self.audio_processor.stop_real_time_processing()
    
    def update_visualization(self):
        """Update audio visualization"""
        if self.original_audio is None or self.processed_audio is None:
            messagebox.showwarning("Warning", "No audio data to visualize")
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot original audio
        self.ax1.plot(self.original_audio[:22050], label='Original', alpha=0.7)
        self.ax1.set_title('Original Audio')
        self.ax1.set_xlabel('Sample')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Plot processed audio
        self.ax2.plot(self.processed_audio[:22050], label='Processed', alpha=0.7, color='orange')
        self.ax2.set_title('Processed Audio')
        self.ax2.set_xlabel('Sample')
        self.ax2.set_ylabel('Amplitude')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def clear_plots(self):
        """Clear visualization plots"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title('Original Audio')
        self.ax2.set_title('Processed Audio')
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = AudioProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 