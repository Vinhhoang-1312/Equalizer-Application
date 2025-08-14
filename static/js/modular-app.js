/**
 * Advanced Audio Processing - Modular JavaScript Application
 * Handles all client-side interactions for the 6 module tabs
 */

class AdvancedAudioApp {
    constructor() {
        this.socket = io();
        this.currentFile = null;
        this.currentAudio = null;
        this.isProcessing = false;
        this.isRealtimeActive = false;
        this.charts = {};
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        console.log('🚀 Advanced Audio Processing App initializing...');
        
        // Set up all event listeners
        this.setupFileUpload();
        this.setupEqualizer();
        this.setupNoiseReduction();
        this.setupGenreClassification();
        this.setupRealtimeProcessing();
        this.setupAnalysis();
        this.setupSocketEvents();
        
        // Load initial data
        this.loadAudioDevices();
        this.loadEqualizerPresets();
        this.loadModelInfo();
        
        console.log('✓ Advanced Audio Processing App initialized');
    }

    // File Upload Module
    setupFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('audioFile');

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.uploadFile(e.target.files[0]);
            }
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                this.uploadFile(e.dataTransfer.files[0]);
            }
        });
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showProcessingStatus('Uploading file...');
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.currentFile = result;
                this.displayFileInfo(result);
                this.hideProcessingStatus();
                this.showSuccess('File uploaded successfully!');
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.hideProcessingStatus();
            this.showError('Upload failed: ' + error.message);
        }
    }

    displayFileInfo(fileInfo) {
        document.getElementById('fileName').textContent = fileInfo.filename;
        document.getElementById('fileDuration').textContent = fileInfo.duration.toFixed(2);
        document.getElementById('fileSampleRate').textContent = fileInfo.sample_rate;
        document.getElementById('fileRMS').textContent = fileInfo.rms_level.toFixed(4);
        
        document.getElementById('fileInfo').style.display = 'block';
    }

    // Equalizer Module
    setupEqualizer() {
        // Slider value updates
        const sliders = ['subBass', 'bass', 'lowMid', 'mid', 'highMid', 
                        'presence', 'brilliance', 'air', 'ultraHigh', 'extreme'];
        
        sliders.forEach(slider => {
            const element = document.getElementById(slider);
            const valueElement = document.getElementById(slider + 'Value');
            
            element.addEventListener('input', (e) => {
                valueElement.textContent = e.target.value + ' dB';
                this.updateFrequencyResponse();
            });
        });

        // Preset loading
        document.getElementById('loadPreset').addEventListener('click', () => {
            this.loadEqualizerPreset();
        });

        // Processing
        document.getElementById('processEqualizer').addEventListener('click', () => {
            this.processEqualizer();
        });

        // Reset
        document.getElementById('resetEqualizer').addEventListener('click', () => {
            this.resetEqualizer();
        });

        // Initialize frequency response chart
        this.initFrequencyResponseChart();
    }

    async loadEqualizerPreset() {
        const presetName = document.getElementById('eqPreset').value;
        if (!presetName) return;

        try {
            const response = await fetch('/api/equalizer/presets');
            const data = await response.json();
            
            if (data.presets[presetName]) {
                const gains = data.presets[presetName];
                
                // Map preset gains to sliders
                const mapping = {
                    'sub_bass': 'subBass',
                    'bass': 'bass',
                    'low_mid': 'lowMid',
                    'mid': 'mid',
                    'high_mid': 'highMid',
                    'presence': 'presence',
                    'brilliance': 'brilliance',
                    'air': 'air',
                    'ultra_high': 'ultraHigh',
                    'extreme': 'extreme'
                };

                Object.entries(mapping).forEach(([key, sliderId]) => {
                    const slider = document.getElementById(sliderId);
                    const valueElement = document.getElementById(sliderId + 'Value');
                    
                    if (gains[key] !== undefined) {
                        slider.value = gains[key];
                        valueElement.textContent = gains[key] + ' dB';
                    }
                });

                this.updateFrequencyResponse();
                this.showSuccess(`Loaded preset: ${presetName}`);
            }
        } catch (error) {
            this.showError('Failed to load preset: ' + error.message);
        }
    }

    async processEqualizer() {
        if (!this.currentFile) {
            this.showError('Please upload an audio file first');
            return;
        }

        const gains = this.getEqualizerGains();
        const method = document.getElementById('eqMethod').value;
        const preset = document.getElementById('eqPreset').value;

        try {
            this.showProcessingStatus('Applying equalizer...');

            const response = await fetch('/api/equalizer/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gains: gains,
                    method: method,
                    preset: preset || null
                })
            });

            const result = await response.json();

            if (result.success) {
                this.hideProcessingStatus();
                this.showSuccess(`Equalizer applied! RMS change: ${result.rms_change_db.toFixed(2)} dB`);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.hideProcessingStatus();
            this.showError('Equalizer processing failed: ' + error.message);
        }
    }

    getEqualizerGains() {
        return {
            sub_bass: parseFloat(document.getElementById('subBass').value),
            bass: parseFloat(document.getElementById('bass').value),
            low_mid: parseFloat(document.getElementById('lowMid').value),
            mid: parseFloat(document.getElementById('mid').value),
            high_mid: parseFloat(document.getElementById('highMid').value),
            presence: parseFloat(document.getElementById('presence').value),
            brilliance: parseFloat(document.getElementById('brilliance').value),
            air: parseFloat(document.getElementById('air').value),
            ultra_high: parseFloat(document.getElementById('ultraHigh').value),
            extreme: parseFloat(document.getElementById('extreme').value)
        };
    }

    resetEqualizer() {
        const sliders = ['subBass', 'bass', 'lowMid', 'mid', 'highMid', 
                        'presence', 'brilliance', 'air', 'ultraHigh', 'extreme'];
        
        sliders.forEach(slider => {
            document.getElementById(slider).value = 0;
            document.getElementById(slider + 'Value').textContent = '0 dB';
        });

        document.getElementById('eqPreset').value = '';
        this.updateFrequencyResponse();
    }

    initFrequencyResponseChart() {
        const ctx = document.getElementById('frequencyResponseChart').getContext('2d');
        
        this.charts.frequencyResponse = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Frequency Response',
                    data: [],
                    borderColor: 'rgb(102, 126, 234)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: { display: true, text: 'Frequency (Hz)' }
                    },
                    y: {
                        title: { display: true, text: 'Gain (dB)' },
                        min: -25,
                        max: 25
                    }
                }
            }
        });

        this.updateFrequencyResponse();
    }

    updateFrequencyResponse() {
        // Generate frequency response curve
        const frequencies = [];
        const response = [];
        
        // Log scale from 20Hz to 20kHz
        for (let i = 0; i < 100; i++) {
            const freq = 20 * Math.pow(1000, i / 99); // 20Hz to 20kHz
            frequencies.push(freq);
            
            // Calculate response based on current gains
            let gain = 0;
            const gains = this.getEqualizerGains();
            const bands = {
                60: gains.sub_bass,
                170: gains.bass,
                310: gains.low_mid,
                600: gains.mid,
                1000: gains.high_mid,
                3000: gains.presence,
                6000: gains.brilliance,
                12000: gains.air,
                14000: gains.ultra_high,
                16000: gains.extreme
            };

            // Simple bell curve approximation
            Object.entries(bands).forEach(([centerFreq, bandGain]) => {
                const center = parseFloat(centerFreq);
                const distance = Math.abs(Math.log10(freq) - Math.log10(center));
                const influence = Math.exp(-distance * 2); // Gaussian-like curve
                gain += bandGain * influence;
            });

            response.push(gain);
        }

        // Update chart
        this.charts.frequencyResponse.data.labels = frequencies;
        this.charts.frequencyResponse.data.datasets[0].data = response;
        this.charts.frequencyResponse.update();
    }

    // Noise Reduction Module
    setupNoiseReduction() {
        // Reduction level slider
        document.getElementById('reductionLevel').addEventListener('input', (e) => {
            document.getElementById('reductionValue').textContent = e.target.value;
        });

        // New simplified buttons
        document.getElementById('processNoiseML').addEventListener('click', () => {
            this.processNoiseWithMethod('ml');
        });

        document.getElementById('processNoiseLibrary').addEventListener('click', () => {
            this.processNoiseWithMethod('library');
        });
    }

    async processNoiseReduction() {
        if (!this.currentFile) {
            this.showError('Please upload an audio file first');
            return;
        }

        const method = document.getElementById('noiseMethod').value;
        const reductionLevel = parseFloat(document.getElementById('reductionLevel').value);

        try {
            this.showProcessingStatus('Reducing noise...');

            const response = await fetch('/api/noise_reduction/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    method: method,
                    reduction_level: reductionLevel
                })
            });

            const result = await response.json();

            if (result.success) {
                this.hideProcessingStatus();
                this.displayNoiseResults(result);
                this.showSuccess(`Noise reduced! SNR improvement: ${result.snr_improvement.toFixed(2)} dB`);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.hideProcessingStatus();
            this.showError('Noise reduction failed: ' + error.message);
        }
    }

    async analyzeNoise() {
        if (!this.currentFile) {
            this.showError('Please upload an audio file first');
            return;
        }

        // This would typically call a specific analyze endpoint
        // For now, we'll show the analysis from the last processing
        document.getElementById('noiseAnalysis').style.display = 'block';
    }

    displayNoiseResults(result) {
        const analysis = result.original_analysis;
        
        document.getElementById('snrEstimate').textContent = analysis.snr_estimate?.toFixed(1) || 'N/A';
        document.getElementById('rmsLevel').textContent = analysis.rms_level?.toFixed(4) || 'N/A';
        document.getElementById('dynamicRange').textContent = analysis.dynamic_range?.toFixed(1) || 'N/A';
        document.getElementById('recommendedMethod').textContent = analysis.recommended_method || 'N/A';
        document.getElementById('recommendedLevel').textContent = analysis.recommended_reduction?.toFixed(1) || 'N/A';
        document.getElementById('spectralCentroid').textContent = analysis.spectral_centroid?.toFixed(0) || 'N/A';
        
        document.getElementById('noiseAnalysis').style.display = 'block';
    }

    async processNoiseWithMethod(method) {
        if (!this.currentFile) {
            this.showError('Vui lòng upload file audio trước');
            return;
        }

        const reductionLevel = parseFloat(document.getElementById('reductionLevel').value);

        try {
            let statusMessage = method === 'ml' ? 
                'Đang giảm nhiễu với AI/ML hệ thống...' : 
                'Đang giảm nhiễu với thư viện tốt nhất...';
            
            this.showProcessingStatus(statusMessage);

            // Choose method based on type
            const noiseMethod = method === 'ml' ? 'autoencoder' : 'noisereduce';

            const response = await fetch('/api/noise_reduction/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    method: noiseMethod,
                    reduction_level: reductionLevel
                })
            });

            const result = await response.json();

            if (result.success) {
                this.hideProcessingStatus();
                this.displayNoiseResults(result);
                
                const methodName = method === 'ml' ? 'AI/ML Hệ Thống' : 'Thư Viện Tốt Nhất';
                this.showSuccess(`Giảm nhiễu thành công! SNR cải thiện: ${result.snr_improvement.toFixed(2)} dB - ${methodName}`);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.hideProcessingStatus();
            this.showError('Giảm nhiễu thất bại: ' + error.message);
        }
    }

    // Genre Classification Module
    setupGenreClassification() {
        // New simplified buttons
        document.getElementById('classifyML').addEventListener('click', () => {
            this.classifyWithMethod('ml');
        });

        document.getElementById('classifyLibrary').addEventListener('click', () => {
            this.classifyWithMethod('library');
        });

        // Load model status
        this.loadModelInfo();
    }

    async classifyGenre() {
        if (!this.currentFile) {
            this.showError('Please upload an audio file first');
            return;
        }

        const method = document.getElementById('genreMethod').value;

        try {
            this.showProcessingStatus('Classifying genre...');

            const response = await fetch('/api/genre_classification/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ method: method })
            });

            const result = await response.json();

            if (result.success) {
                this.hideProcessingStatus();
                this.displayGenreResults(result);
                this.showSuccess(`Genre classified: ${result.predicted_genre} (${(result.confidence * 100).toFixed(1)}%)`);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.hideProcessingStatus();
            this.showError('Genre classification failed: ' + error.message);
        }
    }

    displayGenreResults(result) {
        document.getElementById('predictedGenre').textContent = result.predicted_genre;
        document.getElementById('genreConfidence').textContent = (result.confidence * 100).toFixed(1);
        document.getElementById('methodUsed').textContent = result.method;
        
        const confidenceBar = document.getElementById('confidenceBar');
        confidenceBar.style.width = (result.confidence * 100) + '%';
        
        document.getElementById('genreResults').style.display = 'block';

        // Show detailed analysis if available
        if (result.additional_info && result.additional_info.ensemble_probabilities) {
            this.displayProbabilityTable(result.additional_info.ensemble_probabilities);
            document.getElementById('detailedAnalysis').style.display = 'block';
        }
    }

    displayProbabilityTable(probabilities) {
        const tbody = document.getElementById('probabilityTable');
        tbody.innerHTML = '';

        // Sort by probability
        const sortedProbs = Object.entries(probabilities)
            .sort(([,a], [,b]) => b - a);

        sortedProbs.forEach(([genre, prob]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${genre}</td>
                <td>${(prob * 100).toFixed(1)}%</td>
                <td>
                    <div class="progress" style="height: 20px;">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${prob * 100}%"></div>
                    </div>
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    async loadModelInfo() {
        try {
            const response = await fetch('/api/genre_classification/info');
            const info = await response.json();

            // Update model status indicators
            const models = ['rf', 'svm', 'nn', 'lstm', 'cnn'];
            const modelNames = ['random_forest', 'svm', 'neural_network', 'lstm', 'cnn'];

            models.forEach((model, index) => {
                const statusElement = document.getElementById(model + 'ModelStatus');
                if (statusElement) {
                    const isLoaded = info.models_loaded && info.models_loaded[modelNames[index]];
                    statusElement.className = 'status-indicator ' + (isLoaded ? 'status-active' : 'status-inactive');
                }
            });
        } catch (error) {
            console.warn('Could not load model info:', error);
        }
    }

    async classifyWithMethod(method) {
        if (!this.currentFile) {
            this.showError('Vui lòng upload file audio trước');
            return;
        }

        try {
            let statusMessage = method === 'ml' ? 
                'Đang phân tích với AI/ML hệ thống...' : 
                'Đang phân tích với thư viện tốt nhất...';
            
            this.showProcessingStatus(statusMessage);

            const endpoint = '/api/genre_classification/classify_best';
                
            const body = method === 'ml' ? 
                JSON.stringify({ option: 'option2' }) :  // AI/ML Hệ Thống = Custom ML
                JSON.stringify({ option: 'option1' });  // Thư Viện Tốt Nhất = Advanced Librosa

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: body
            });

            const result = await response.json();

            if (result.success) {
                this.hideProcessingStatus();
                this.displayGenreResults(result);
                
                const methodName = method === 'ml' ? 'AI/ML Hệ Thống' : 'Thư Viện Tốt Nhất';
                this.showSuccess(`Kết quả: ${result.predicted_genre} (${(result.confidence * 100).toFixed(1)}%) - ${methodName}`);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.hideProcessingStatus();
            this.showError('Phân tích thất bại: ' + error.message);
        }
    }

    displayBestClassificationResults(result) {
        // Update basic results
        document.getElementById('predictedGenre').textContent = result.predicted_genre;
        document.getElementById('genreConfidence').textContent = (result.confidence * 100).toFixed(1);
        document.getElementById('methodUsed').textContent = result.method || 'Advanced Classification';
        
        const confidenceBar = document.getElementById('confidenceBar');
        confidenceBar.style.width = (result.confidence * 100) + '%';
        
        document.getElementById('genreResults').style.display = 'block';

        // Show comparison results if both methods were tested
        if (result.comparison) {
            this.displayComparisonResults(result.comparison);
        }

        // Show detailed analysis if available
        if (result.additional_info) {
            if (result.additional_info.ensemble_probabilities) {
                this.displayProbabilityTable(result.additional_info.ensemble_probabilities);
                document.getElementById('detailedAnalysis').style.display = 'block';
            }
        }
    }

    displayComparisonResults(comparison) {
        // Create comparison display if it doesn't exist
        let comparisonDiv = document.getElementById('comparisonResults');
        if (!comparisonDiv) {
            comparisonDiv = document.createElement('div');
            comparisonDiv.id = 'comparisonResults';
            comparisonDiv.className = 'mt-3';
            document.getElementById('genreResults').appendChild(comparisonDiv);
        }

        comparisonDiv.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h6 class="mb-0"><i class="fas fa-chart-bar"></i> Method Comparison</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="comparison-item ${comparison.best_method === 'Musicnn' ? 'winner' : ''}">
                                <strong>🤖 Musicnn Deep Learning</strong><br>
                                <span class="text-info">${comparison.option1_result.predicted_genre}</span>
                                <span class="confidence-badge">${(comparison.option1_result.confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="comparison-item ${comparison.best_method === 'Custom ML' ? 'winner' : ''}">
                                <strong>🎯 Custom ML</strong><br>
                                <span class="text-success">${comparison.option2_result.predicted_genre}</span>
                                <span class="confidence-badge">${(comparison.option2_result.confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                    <div class="text-center mt-2">
                        <span class="badge bg-primary">Winner: ${comparison.best_method}</span>
                    </div>
                </div>
            </div>
        `;
        comparisonDiv.style.display = 'block';
    }

    // Real-time Processing Module
    setupRealtimeProcessing() {
        // Device management
        document.getElementById('refreshDevices').addEventListener('click', () => {
            this.loadAudioDevices();
        });

        document.getElementById('testLatency').addEventListener('click', () => {
            this.testLatency();
        });

        // Real-time control
        document.getElementById('startRealtime').addEventListener('click', () => {
            this.startRealtimeProcessing();
        });

        document.getElementById('stopRealtime').addEventListener('click', () => {
            this.stopRealtimeProcessing();
        });

        // Recording control
        document.getElementById('startRecording').addEventListener('click', () => {
            this.startRecording();
        });

        document.getElementById('stopRecording').addEventListener('click', () => {
            this.stopRecording();
        });

        // Initialize audio visualizer
        this.initAudioVisualizer();
    }

    async loadAudioDevices() {
        try {
            const response = await fetch('/api/audio_devices');
            const devices = await response.json();

            // Populate input devices
            const inputSelect = document.getElementById('inputDevice');
            inputSelect.innerHTML = '<option value="">Default Input Device</option>';
            devices.input?.forEach(device => {
                const option = document.createElement('option');
                option.value = device.index;
                option.textContent = `${device.name} (${device.channels} ch)`;
                inputSelect.appendChild(option);
            });

            // Populate output devices
            const outputSelect = document.getElementById('outputDevice');
            outputSelect.innerHTML = '<option value="">Default Output Device</option>';
            devices.output?.forEach(device => {
                const option = document.createElement('option');
                option.value = device.index;
                option.textContent = `${device.name} (${device.channels} ch)`;
                outputSelect.appendChild(option);
            });

        } catch (error) {
            this.showError('Failed to load audio devices: ' + error.message);
        }
    }

    async startRealtimeProcessing() {
        const equalizerParams = this.getEqualizerGains();
        const noiseMethod = document.getElementById('noiseMethod').value;
        const noiseLevel = parseFloat(document.getElementById('reductionLevel').value);
        const enabledModules = {
            equalizer: document.getElementById('enableEqualizer').checked,
            noise_reduction: document.getElementById('enableNoiseReduction').checked,
            genre_classification: document.getElementById('enableGenreClassification').checked
        };

        try {
            const response = await fetch('/api/realtime/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    equalizer_params: equalizerParams,
                    noise_method: noiseMethod,
                    noise_reduction_level: noiseLevel,
                    enabled_modules: enabledModules
                })
            });

            const result = await response.json();

            if (result.success) {
                this.isRealtimeActive = true;
                document.getElementById('startRealtime').disabled = true;
                document.getElementById('stopRealtime').disabled = false;
                document.getElementById('realtimeStats').style.display = 'block';
                
                this.showSuccess('Real-time processing started!');
                this.startStatsUpdater();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showError('Failed to start real-time processing: ' + error.message);
        }
    }

    async stopRealtimeProcessing() {
        try {
            const response = await fetch('/api/realtime/stop', { method: 'POST' });
            const result = await response.json();

            if (result.success) {
                this.isRealtimeActive = false;
                document.getElementById('startRealtime').disabled = false;
                document.getElementById('stopRealtime').disabled = true;
                
                this.showSuccess('Real-time processing stopped');
                this.stopStatsUpdater();
            }
        } catch (error) {
            this.showError('Failed to stop real-time processing: ' + error.message);
        }
    }

    startStatsUpdater() {
        if (this.statsInterval) clearInterval(this.statsInterval);
        
        this.statsInterval = setInterval(async () => {
            if (!this.isRealtimeActive) return;

            try {
                const response = await fetch('/api/realtime/stats');
                const stats = await response.json();

                document.getElementById('avgLatency').textContent = stats.avg_latency_ms?.toFixed(1) + ' ms' || '0 ms';
                document.getElementById('chunksProcessed').textContent = stats.chunks_processed || '0';
                document.getElementById('processingErrors').textContent = stats.processing_errors || '0';
            } catch (error) {
                console.warn('Failed to update stats:', error);
            }
        }, 1000);
    }

    stopStatsUpdater() {
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }
    }

    initAudioVisualizer() {
        const canvas = document.getElementById('audioVisualizer');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        const resizeCanvas = () => {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        };
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Simple audio visualization
        this.visualizerData = new Array(128).fill(0);
        
        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const barWidth = canvas.width / this.visualizerData.length;
            const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
            gradient.addColorStop(0, 'rgb(102, 126, 234)');
            gradient.addColorStop(1, 'rgb(118, 75, 162)');
            
            ctx.fillStyle = gradient;
            
            this.visualizerData.forEach((value, index) => {
                const barHeight = (value * canvas.height) / 100;
                const x = index * barWidth;
                const y = canvas.height - barHeight;
                
                ctx.fillRect(x, y, barWidth - 1, barHeight);
            });
            
            requestAnimationFrame(draw);
        };
        
        draw();
    }

    // Analysis Module
    setupAnalysis() {
        document.getElementById('runAnalysis').addEventListener('click', () => {
            this.runAnalysis();
        });

        document.getElementById('exportResults').addEventListener('click', () => {
            this.exportResults();
        });
    }

    async runAnalysis() {
        if (!this.currentFile) {
            this.showError('Please upload an audio file first');
            return;
        }

        this.showSuccess('Analysis feature coming soon!');
        document.getElementById('analysisResults').style.display = 'block';
        document.getElementById('processingSummary').style.display = 'block';
    }

    // Socket Events
    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('✓ Connected to server');
        });

        this.socket.on('realtime_audio', (data) => {
            // Update audio visualizer
            if (this.visualizerData) {
                // Simulate audio data visualization
                this.visualizerData = this.visualizerData.map(() => Math.random() * 100);
            }
        });

        this.socket.on('realtime_genre', (data) => {
            // Update current genre display
            document.getElementById('currentGenre').textContent = data.genre || '-';
        });

        this.socket.on('disconnect', () => {
            console.log('⚠️ Disconnected from server');
        });
    }

    // Utility Methods
    showProcessingStatus(message) {
        const statusDiv = document.getElementById('processingStatus');
        statusDiv.querySelector('span').textContent = message;
        statusDiv.style.display = 'block';
        this.isProcessing = true;
    }

    hideProcessingStatus() {
        document.getElementById('processingStatus').style.display = 'none';
        this.isProcessing = false;
    }

    showSuccess(message) {
        this.showAlert(message, 'success');
    }

    showError(message) {
        this.showAlert(message, 'danger');
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts
        document.querySelectorAll('.alert-custom').forEach(alert => alert.remove());

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-custom alert-dismissible fade show`;
        alertDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;
        
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    async loadEqualizerPresets() {
        // This is already implemented in setupEqualizer
    }

    async testLatency() {
        this.showSuccess('Latency test feature coming soon!');
    }

    startRecording() {
        this.showSuccess('Recording feature coming soon!');
    }

    stopRecording() {
        this.showSuccess('Stop recording feature coming soon!');
    }

    compareNoiseReduction() {
        this.showSuccess('Noise comparison feature coming soon!');
    }

    exportResults() {
        this.showSuccess('Export results feature coming soon!');
    }
}

// Initialize the application
const app = new AdvancedAudioApp();
