/**
 * Modern Audio Visualizer using WaveSurfer.js
 * Professional waveform visualization with multiple styles
 * 
 * Features:
 * - Real-time microphone visualization
 * - Multiple visualization styles (Wave, Bars, Spectrogram, Oscilloscope)
 * - Professional audio controls
 * - Beautiful gradients and animations
 */

class WaveSurferAudioVisualizer {
    constructor() {
        this.wavesurfer = null;
        this.microphone = null;
        this.audioContext = null;
        this.analyser = null;
        this.mediaStream = null;
        this.isRecording = false;
        this.currentStyle = 'wave';
        
        this.init();
    }

    async init() {
        console.log('🎵 Initializing WaveSurfer Audio Visualizer...');
        
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initializeWaveSurfer());
        } else {
            this.initializeWaveSurfer();
        }
    }

    initializeWaveSurfer() {
        const waveformContainer = document.getElementById('waveform');
        if (!waveformContainer) {
            console.log('⏳ Waveform container not found, retrying...');
            setTimeout(() => this.initializeWaveSurfer(), 500);
            return;
        }

        try {
            // Create WaveSurfer instance with beautiful gradient
            this.wavesurfer = WaveSurfer.create({
                container: '#waveform',
                waveColor: this.createGradient(),
                progressColor: 'rgba(255, 255, 255, 0.8)',
                cursorColor: '#ffffff',
                barWidth: 3,
                barRadius: 3,
                barGap: 2,
                height: 120,
                normalize: true,
                interact: true,
                hideScrollbar: true,
                cursorWidth: 2
            });

            console.log('✅ WaveSurfer initialized successfully!');
            this.setupEventListeners();
            this.setupMicrophone();
            
        } catch (error) {
            console.error('❌ Error initializing WaveSurfer:', error);
        }
    }

    createGradient() {
        // Create beautiful gradient for waveform
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 120);
        
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(0.5, '#764ba2');
        gradient.addColorStop(1, '#f093fb');
        
        return gradient;
    }

    setupEventListeners() {
        // Style switching buttons
        document.querySelectorAll('.viz-style').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const style = e.currentTarget.dataset.style;
                this.switchVisualizationStyle(style);
                
                // Update button states
                document.querySelectorAll('.viz-style').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
            });
        });

        // Audio controls
        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const stopBtn = document.getElementById('stopBtn');
        const volumeSlider = document.getElementById('volumeSlider');

        if (playBtn) {
            playBtn.addEventListener('click', () => this.play());
        }
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => this.pause());
        }
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stop());
        }
        if (volumeSlider) {
            volumeSlider.addEventListener('input', (e) => {
                const volume = e.target.value / 100;
                this.setVolume(volume);
            });
        }

        // Set default active style
        document.querySelector('.viz-style[data-style="wave"]')?.classList.add('active');
    }

    async setupMicrophone() {
        try {
            console.log('🎤 Setting up microphone access...');
            
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create analyser for real-time visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;

            // Connect microphone to analyser
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            source.connect(this.analyser);

            console.log('✅ Microphone setup complete!');
            this.startRealtimeVisualization();
            
        } catch (error) {
            console.error('❌ Microphone access denied:', error);
            this.showMicrophoneError();
        }
    }

    startRealtimeVisualization() {
        if (!this.analyser) return;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            if (!this.analyser) return;
            
            this.analyser.getByteTimeDomainData(dataArray);
            
            // Convert to Float32Array for WaveSurfer
            const floatArray = new Float32Array(bufferLength);
            for (let i = 0; i < bufferLength; i++) {
                floatArray[i] = (dataArray[i] - 128) / 128.0;
            }

            // Update waveform visualization
            if (this.wavesurfer) {
                try {
                    // Generate fake audio buffer for real-time visualization
                    const audioBuffer = this.audioContext.createBuffer(1, bufferLength, 44100);
                    audioBuffer.getChannelData(0).set(floatArray);
                    
                    // Load the buffer into WaveSurfer
                    this.wavesurfer.loadDecodedBuffer(audioBuffer);
                } catch (error) {
                    // Silently handle errors to avoid spam
                }
            }
            
            requestAnimationFrame(draw);
        };
        
        draw();
        console.log('🎵 Real-time visualization started!');
    }

    switchVisualizationStyle(style) {
        this.currentStyle = style;
        
        if (!this.wavesurfer) return;

        // Apply different styles
        switch (style) {
            case 'wave':
                this.wavesurfer.setOptions({
                    waveColor: this.createGradient(),
                    progressColor: 'rgba(255, 255, 255, 0.8)',
                    barWidth: 0, // Disable bars for wave mode
                    cursorColor: '#ffffff'
                });
                break;
                
            case 'bars':
                this.wavesurfer.setOptions({
                    waveColor: '#3498db',
                    progressColor: '#e74c3c',
                    barWidth: 4,
                    barGap: 2,
                    barRadius: 4,
                    cursorColor: '#f39c12'
                });
                break;
                
            case 'spectrogram':
                this.wavesurfer.setOptions({
                    waveColor: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'],
                    progressColor: 'rgba(255, 255, 255, 0.6)',
                    barWidth: 2,
                    barGap: 1,
                    cursorColor: '#ffffff'
                });
                break;
                
            case 'oscilloscope':
                this.wavesurfer.setOptions({
                    waveColor: '#00ff88',
                    progressColor: '#ffff00',
                    barWidth: 0,
                    cursorColor: '#00ff88',
                    normalize: false
                });
                break;
        }
        
        console.log(`🎨 Switched to ${style} visualization style`);
    }

    play() {
        if (this.wavesurfer && this.wavesurfer.isPlaying()) {
            return;
        }
        
        if (this.wavesurfer) {
            this.wavesurfer.play();
        }
        console.log('▶️ Playing audio');
    }

    pause() {
        if (this.wavesurfer) {
            this.wavesurfer.pause();
        }
        console.log('⏸️ Audio paused');
    }

    stop() {
        if (this.wavesurfer) {
            this.wavesurfer.stop();
        }
        console.log('⏹️ Audio stopped');
    }

    setVolume(volume) {
        if (this.wavesurfer) {
            this.wavesurfer.setVolume(volume);
        }
        console.log(`🔊 Volume set to ${Math.round(volume * 100)}%`);
    }

    showMicrophoneError() {
        const waveformContainer = document.getElementById('waveform');
        if (waveformContainer) {
            waveformContainer.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 120px; color: #ffffff; text-align: center;">
                    <div>
                        <i class="fas fa-microphone-slash fa-2x mb-2"></i>
                        <p>Microphone access required for real-time visualization</p>
                        <small>Please allow microphone permissions and refresh</small>
                    </div>
                </div>
            `;
        }
    }

    // Load audio file for demonstration
    async loadDemoAudio() {
        try {
            // You can load any demo audio file here
            await this.wavesurfer.load('/static/demo-audio.mp3');
            console.log('🎵 Demo audio loaded');
        } catch (error) {
            console.log('ℹ️ No demo audio available');
        }
    }

    destroy() {
        if (this.wavesurfer) {
            this.wavesurfer.destroy();
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        console.log('🗑️ WaveSurfer visualizer destroyed');
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.waveSurferVisualizer = new WaveSurferAudioVisualizer();
    });
} else {
    window.waveSurferVisualizer = new WaveSurferAudioVisualizer();
}

// Export for use in other modules
window.WaveSurferAudioVisualizer = WaveSurferAudioVisualizer;
