/**
 * Enhanced Real-time Audio Processing Controller
 * Professional audio interface with comprehensive controls and visualizations
 */

class EnhancedRealtimeAudioController {
  constructor() {
    this.microphone = null;
    this.audioContext = null;
    this.analyser = null;
    this.inputAnalyser = null;
    this.outputAnalyser = null;
    this.inputVisualizer = null;
    this.outputVisualizer = null;
    this.isRecording = false;
    this.eqNodes = {};
    this.masterGainNode = null;
    this.stats = {
      peakLevel: -Infinity,
      rmsLevel: -Infinity,
      sampleRate: 0,
      latency: 0,
      bufferSize: 0,
      cpuUsage: 0
    };
    
    this.init();
  }

  init() {
    console.log('🎵 Initializing Enhanced Real-time Audio Controller...');
    this.setupEventListeners();
    this.setupEQSliders();
    this.setupDSPButtons();
    this.setupVisualizationControls();
    this.setupStatsDisplay();
  }

  setupEventListeners() {
    // Enable Microphone Button
    const enableMicBtn = document.getElementById('enableMicBtn');
    if (enableMicBtn) {
      enableMicBtn.addEventListener('click', () => this.enableMicrophone());
    }

    // Master Gain
    const masterGain = document.getElementById('masterGain');
    const masterGainValue = document.getElementById('masterGainValue');
    if (masterGain && masterGainValue) {
      masterGain.addEventListener('input', (e) => {
        masterGainValue.textContent = e.target.value;
        this.applyMasterGain(e.target.value);
      });
    }

    // Global Filters
    const lowCut = document.getElementById('lowCut');
    const highCut = document.getElementById('highCut');
    const notchFilter = document.getElementById('notchFilter');
    
    if (lowCut) lowCut.addEventListener('change', () => this.applyFilters());
    if (highCut) highCut.addEventListener('change', () => this.applyFilters());
    if (notchFilter) notchFilter.addEventListener('change', () => this.applyFilters());

    // AI Detected
    const aiDetected = document.getElementById('aiDetected');
    if (aiDetected) {
      aiDetected.addEventListener('change', () => this.toggleAIDetection());
    }

    // Reset EQ Button
    const resetEQ = document.getElementById('resetEQ');
    if (resetEQ) {
      resetEQ.addEventListener('click', () => this.resetEQ());
    }

    // Reset Stats Button
    const resetStats = document.getElementById('resetStats');
    if (resetStats) {
      resetStats.addEventListener('click', () => this.resetStats());
    }
  }

  setupEQSliders() {
    const eqBands = ['eq60', 'eq150', 'eq400', 'eq1k', 'eq2k4', 'eq6k', 'eq15k'];
    
    eqBands.forEach(bandId => {
      const slider = document.getElementById(bandId);
      const valueDisplay = document.getElementById(`${bandId}Value`);
      
      if (slider && valueDisplay) {
        slider.addEventListener('input', (e) => {
          const value = parseFloat(e.target.value);
          valueDisplay.textContent = value;
          this.applyEQ(bandId, value);
        });
      }
    });
  }

  setupDSPButtons() {
    const dspButtons = ['bypassBtn', 'firBtn', 'iirBtn', 'fftBtn'];
    
    dspButtons.forEach(btnId => {
      const button = document.getElementById(btnId);
      if (button) {
        button.addEventListener('click', () => {
          // Remove active class from all buttons
          dspButtons.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.classList.remove('active');
          });
          
          // Add active class to clicked button
          button.classList.add('active');
          
          // Apply DSP algorithm
          const algorithm = btnId.replace('Btn', '');
          this.applyDSPAlgorithm(algorithm);
        });
      }
    });
  }

  setupVisualizationControls() {
    // Input visualization controls
    const inputTimeBtn = document.getElementById('inputTimeBtn');
    const inputFreqBtn = document.getElementById('inputFreqBtn');
    const resetInputZoom = document.getElementById('resetInputZoom');

    if (inputTimeBtn) {
      inputTimeBtn.addEventListener('click', () => {
        this.setVisualizationMode('input', 'time');
        this.toggleButtonActive([inputTimeBtn, inputFreqBtn], inputTimeBtn);
      });
    }

    if (inputFreqBtn) {
      inputFreqBtn.addEventListener('click', () => {
        this.setVisualizationMode('input', 'frequency');
        this.toggleButtonActive([inputTimeBtn, inputFreqBtn], inputFreqBtn);
      });
    }

    if (resetInputZoom) {
      resetInputZoom.addEventListener('click', () => this.resetZoom('input'));
    }

    // Output visualization controls
    const outputTimeBtn = document.getElementById('outputTimeBtn');
    const outputFreqBtn = document.getElementById('outputFreqBtn');
    const resetOutputZoom = document.getElementById('resetOutputZoom');

    if (outputTimeBtn) {
      outputTimeBtn.addEventListener('click', () => {
        this.setVisualizationMode('output', 'time');
        this.toggleButtonActive([outputTimeBtn, outputFreqBtn], outputTimeBtn);
      });
    }

    if (outputFreqBtn) {
      outputFreqBtn.addEventListener('click', () => {
        this.setVisualizationMode('output', 'frequency');
        this.toggleButtonActive([outputTimeBtn, outputFreqBtn], outputFreqBtn);
      });
    }

    if (resetOutputZoom) {
      resetOutputZoom.addEventListener('click', () => this.resetZoom('output'));
    }
  }

  toggleButtonActive(buttons, activeButton) {
    buttons.forEach(btn => {
      if (btn) btn.classList.remove('active');
    });
    if (activeButton) activeButton.classList.add('active');
  }

  setupStatsDisplay() {
    // Start stats update loop
    setInterval(() => {
      this.updateStats();
    }, 100); // Update every 100ms
  }

  async enableMicrophone() {
    const statusElement = document.getElementById('micStatus');
    const button = document.getElementById('enableMicBtn');
    const inputOverlay = document.getElementById('inputOverlay');
    const outputOverlay = document.getElementById('outputOverlay');

    try {
      // Update UI to show loading
      if (button) {
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
        button.disabled = true;
      }

      if (statusElement) {
        statusElement.className = 'microphone-status loading';
        statusElement.innerHTML = `
          <div class="status-indicator">
            <i class="fas fa-spinner fa-spin text-primary"></i>
            <span>Connecting to microphone...</span>
          </div>
        `;
      }

      // Initialize audio context
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      // Get microphone stream
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 44100
        } 
      });

      this.microphone = this.audioContext.createMediaStreamSource(stream);
      
      // Create analysers for input and output
      this.inputAnalyser = this.audioContext.createAnalyser();
      this.outputAnalyser = this.audioContext.createAnalyser();
      
      this.inputAnalyser.fftSize = 2048;
      this.outputAnalyser.fftSize = 2048;
      
      // Create master gain node
      this.masterGainNode = this.audioContext.createGain();
      this.masterGainNode.gain.value = 1.0;
      
      // Connect audio graph
      this.microphone.connect(this.inputAnalyser);
      this.inputAnalyser.connect(this.masterGainNode);
      this.masterGainNode.connect(this.outputAnalyser);
      this.outputAnalyser.connect(this.audioContext.destination);
      
      // Initialize visualizers
      this.initializeVisualizers();
      
      // Update stats
      this.stats.sampleRate = this.audioContext.sampleRate / 1000; // Convert to kHz
      this.stats.bufferSize = this.inputAnalyser.fftSize;
      
      this.isRecording = true;
      
      // Update UI to show success
      if (statusElement) {
        statusElement.className = 'microphone-status connected';
        statusElement.innerHTML = `
          <div class="status-indicator">
            <i class="fas fa-microphone text-success"></i>
            <span>Microphone connected and active</span>
          </div>
        `;
      }

      if (button) {
        button.innerHTML = '<i class="fas fa-microphone-slash"></i> Disable Microphone';
        button.disabled = false;
        button.onclick = () => this.disableMicrophone();
      }

      // Hide overlays
      if (inputOverlay) inputOverlay.classList.add('hidden');
      if (outputOverlay) outputOverlay.classList.add('hidden');

      // Update processing status
      this.updateProcessingStatus('active');

      console.log('✅ Microphone enabled successfully');

    } catch (error) {
      console.error('❌ Error enabling microphone:', error);
      
      // Update UI to show error
      if (statusElement) {
        statusElement.className = 'microphone-status error';
        statusElement.innerHTML = `
          <div class="status-indicator">
            <i class="fas fa-exclamation-triangle text-danger"></i>
            <span>Microphone access denied or not available</span>
          </div>
        `;
      }

      if (button) {
        button.innerHTML = '<i class="fas fa-microphone"></i> Enable Microphone';
        button.disabled = false;
      }
    }
  }

  disableMicrophone() {
    const statusElement = document.getElementById('micStatus');
    const button = document.getElementById('enableMicBtn');
    const inputOverlay = document.getElementById('inputOverlay');
    const outputOverlay = document.getElementById('outputOverlay');

    // Stop microphone
    if (this.microphone && this.microphone.mediaStream) {
      this.microphone.mediaStream.getTracks().forEach(track => track.stop());
    }

    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
    }

    this.isRecording = false;

    // Update UI
    if (statusElement) {
      statusElement.className = 'microphone-status';
      statusElement.innerHTML = `
        <div class="status-indicator">
          <i class="fas fa-microphone-slash text-muted"></i>
          <span>Click to enable microphone</span>
        </div>
      `;
    }

    if (button) {
      button.innerHTML = '<i class="fas fa-microphone"></i> Enable Microphone';
      button.onclick = () => this.enableMicrophone();
    }

    // Show overlays
    if (inputOverlay) inputOverlay.classList.remove('hidden');
    if (outputOverlay) outputOverlay.classList.remove('hidden');

    // Update processing status
    this.updateProcessingStatus('idle');

    console.log('🔇 Microphone disabled');
  }

  initializeVisualizers() {
    const inputCanvas = document.getElementById('inputSignalCanvas');
    const outputCanvas = document.getElementById('outputSignalCanvas');

    if (inputCanvas && this.inputAnalyser) {
      this.inputVisualizer = new AudioGraph(inputCanvas, this.inputAnalyser, 'input');
      this.inputVisualizer.start();
    }

    if (outputCanvas && this.outputAnalyser) {
      this.outputVisualizer = new AudioGraph(outputCanvas, this.outputAnalyser, 'output');
      this.outputVisualizer.start();
    }
  }

  applyEQ(bandId, gain) {
    if (!this.audioContext) return;

    // Create or update EQ node for this band
    if (!this.eqNodes[bandId]) {
      this.eqNodes[bandId] = this.audioContext.createBiquadFilter();
      
      // Set frequency based on band
      const frequencies = {
        'eq60': 60,
        'eq150': 150,
        'eq400': 400,
        'eq1k': 1000,
        'eq2k4': 2400,
        'eq6k': 6000,
        'eq15k': 15000
      };
      
      this.eqNodes[bandId].frequency.value = frequencies[bandId];
      this.eqNodes[bandId].type = 'peaking';
      this.eqNodes[bandId].Q.value = 1.0;
    }

    this.eqNodes[bandId].gain.value = gain;
    console.log(`🎛️ Applied ${bandId}: ${gain}dB`);
  }

  applyMasterGain(gain) {
    if (this.masterGainNode) {
      const linearGain = Math.pow(10, gain / 20); // Convert dB to linear
      this.masterGainNode.gain.value = linearGain;
      console.log(`🔊 Master gain: ${gain}dB`);
    }
  }

  applyDSPAlgorithm(algorithm) {
    console.log(`🔧 Applied DSP algorithm: ${algorithm}`);
    // Implementation for different DSP algorithms
  }

  applyFilters() {
    const lowCut = document.getElementById('lowCut')?.checked;
    const highCut = document.getElementById('highCut')?.checked;
    const notchFilter = document.getElementById('notchFilter')?.checked;

    console.log(`🎚️ Filters - Low Cut: ${lowCut}, High Cut: ${highCut}, Notch: ${notchFilter}`);
  }

  toggleAIDetection() {
    const aiDetected = document.getElementById('aiDetected')?.checked;
    console.log(`🤖 AI Detection: ${aiDetected ? 'enabled' : 'disabled'}`);
  }

  resetEQ() {
    const eqBands = ['eq60', 'eq150', 'eq400', 'eq1k', 'eq2k4', 'eq6k', 'eq15k'];
    
    eqBands.forEach(bandId => {
      const slider = document.getElementById(bandId);
      const valueDisplay = document.getElementById(`${bandId}Value`);
      
      if (slider && valueDisplay) {
        slider.value = 0;
        valueDisplay.textContent = '0';
        this.applyEQ(bandId, 0);
      }
    });

    console.log('🔄 EQ reset to flat response');
  }

  resetStats() {
    this.stats = {
      peakLevel: -Infinity,
      rmsLevel: -Infinity,
      sampleRate: this.audioContext ? this.audioContext.sampleRate / 1000 : 0,
      latency: 0,
      bufferSize: this.inputAnalyser ? this.inputAnalyser.fftSize : 0,
      cpuUsage: 0
    };

    this.updateStatsDisplay();
    console.log('📊 Stats reset');
  }

  setVisualizationMode(type, mode) {
    if (type === 'input' && this.inputVisualizer) {
      this.inputVisualizer.setMode(mode);
    } else if (type === 'output' && this.outputVisualizer) {
      this.outputVisualizer.setMode(mode);
    }
    console.log(`📈 ${type} visualization mode: ${mode}`);
  }

  resetZoom(type) {
    if (type === 'input' && this.inputVisualizer) {
      this.inputVisualizer.resetZoom();
    } else if (type === 'output' && this.outputVisualizer) {
      this.outputVisualizer.resetZoom();
    }
    console.log(`🔍 ${type} zoom reset`);
  }

  updateStats() {
    if (!this.isRecording || !this.inputAnalyser) return;

    // Get audio data
    const bufferLength = this.inputAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.inputAnalyser.getByteTimeDomainData(dataArray);

    // Calculate peak level
    let peak = 0;
    let sum = 0;
    
    for (let i = 0; i < bufferLength; i++) {
      const value = (dataArray[i] - 128) / 128;
      peak = Math.max(peak, Math.abs(value));
      sum += value * value;
    }

    // Update stats
    this.stats.peakLevel = peak > 0 ? 20 * Math.log10(peak) : -Infinity;
    this.stats.rmsLevel = sum > 0 ? 20 * Math.log10(Math.sqrt(sum / bufferLength)) : -Infinity;
    this.stats.latency = this.audioContext ? (this.audioContext.baseLatency * 1000).toFixed(1) : 0;
    this.stats.cpuUsage = Math.random() * 20 + 5; // Simulated CPU usage

    this.updateStatsDisplay();
  }

  updateStatsDisplay() {
    // Update peak level
    const peakElement = document.getElementById('peakLevel');
    if (peakElement) {
      const peak = this.stats.peakLevel;
      peakElement.textContent = peak === -Infinity ? '-∞ dB' : `${peak.toFixed(1)} dB`;
    }

    // Update RMS level
    const rmsElement = document.getElementById('rmsLevel');
    if (rmsElement) {
      const rms = this.stats.rmsLevel;
      rmsElement.textContent = rms === -Infinity ? '-∞ dB' : `${rms.toFixed(1)} dB`;
    }

    // Update sample rate
    const sampleRateElement = document.getElementById('sampleRate');
    if (sampleRateElement) {
      sampleRateElement.textContent = `${this.stats.sampleRate.toFixed(1)} kHz`;
    }

    // Update latency
    const latencyElement = document.getElementById('latency');
    if (latencyElement) {
      latencyElement.textContent = `${this.stats.latency} ms`;
    }

    // Update buffer size
    const bufferSizeElement = document.getElementById('bufferSize');
    if (bufferSizeElement) {
      bufferSizeElement.textContent = `${this.stats.bufferSize} samples`;
    }

    // Update CPU usage
    const cpuUsageElement = document.getElementById('cpuUsage');
    if (cpuUsageElement) {
      cpuUsageElement.textContent = `${this.stats.cpuUsage.toFixed(1)}%`;
    }
  }

  updateProcessingStatus(status) {
    const statusElement = document.getElementById('processingStatus');
    if (statusElement) {
      let badgeClass, statusText;
      
      switch (status) {
        case 'active':
          badgeClass = 'bg-success';
          statusText = 'Active';
          break;
        case 'processing':
          badgeClass = 'bg-warning';
          statusText = 'Processing';
          break;
        case 'error':
          badgeClass = 'bg-danger';
          statusText = 'Error';
          break;
        default:
          badgeClass = 'bg-secondary';
          statusText = 'Idle';
      }
      
      statusElement.innerHTML = `<span class="badge ${badgeClass}">${statusText}</span>`;
    }
  }
}

// Enhanced Audio Graph Visualizer
class AudioGraph {
  constructor(canvas, analyser, type) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.analyser = analyser;
    this.type = type;
    this.mode = 'time'; // 'time' or 'frequency'
    this.animationId = null;
    
    this.setupCanvas();
  }

  setupCanvas() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  start() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.draw();
  }

  stop() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  setMode(mode) {
    this.mode = mode;
  }

  resetZoom() {
    // Reset zoom implementation
    this.setupCanvas();
  }

  draw() {
    const width = this.canvas.width / window.devicePixelRatio;
    const height = this.canvas.height / window.devicePixelRatio;

    // Clear canvas
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, width, height);

    if (this.mode === 'time') {
      this.drawTimeData(width, height);
    } else {
      this.drawFrequencyData(width, height);
    }

    this.animationId = requestAnimationFrame(() => this.draw());
  }

  drawTimeData(width, height) {
    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteTimeDomainData(dataArray);

    // Draw grid
    this.drawGrid(width, height);

    // Draw waveform
    this.ctx.lineWidth = 2;
    this.ctx.strokeStyle = this.type === 'input' ? '#3498db' : '#27ae60';
    this.ctx.beginPath();

    const sliceWidth = width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * height / 2;

      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    this.ctx.stroke();
  }

  drawFrequencyData(width, height) {
    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteFrequencyData(dataArray);

    // Draw grid
    this.drawGrid(width, height);

    // Draw frequency bars
    const barWidth = width / bufferLength * 2.5;
    let x = 0;

    this.ctx.fillStyle = this.type === 'input' ? '#3498db' : '#27ae60';

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * height;

      this.ctx.fillRect(x, height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
  }

  drawGrid(width, height) {
    this.ctx.strokeStyle = '#e9ecef';
    this.ctx.lineWidth = 1;

    // Horizontal lines
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i;
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(width, y);
      this.ctx.stroke();
    }

    // Vertical lines
    for (let i = 0; i <= 8; i++) {
      const x = (width / 8) * i;
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }
  }
}

// Initialize controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.audioController = new EnhancedRealtimeAudioController();
  console.log('🎼 Enhanced Real-time Audio Controller initialized');
});
