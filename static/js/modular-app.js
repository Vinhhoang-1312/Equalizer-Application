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
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => this.init());
    } else {
      this.init();
    }
  }

  init() {
    console.log("🚀 Advanced Audio Processing App initializing...");

    // Set up all event listeners
    this.setupFileUpload();
    this.setupEqualizer();
    this.setupNoiseReduction();
    this.setupGenreClassification();
    // Chỉ khởi tạo khi tab Real-time được chọn
    this.setupTabSwitching();
    this.setupAnalysis();
    this.setupSocketEvents();

    // Setup realtime tab since it's active by default
    this.setupRealtimeProcessing();

    // Load initial data
    this.loadAudioDevices();
    this.loadEqualizerPresets();
    this.loadModelInfo();

    console.log("✓ Advanced Audio Processing App initialized");
  }

  setupTabSwitching() {
    const mainTabs = document.querySelectorAll('button[data-bs-toggle="pill"]');
    mainTabs.forEach((tabBtn) => {
      tabBtn.addEventListener("shown.bs.tab", (e) => {
        const target = e.target.getAttribute("data-bs-target");
        if (target === "#realtime") {
          this.setupRealtimeProcessing();
        } else {
          this.resetRealtimeRecordingUI();
        }
      });
    });
  }

  resetRealtimeRecordingUI() {
    // Reset trạng thái nút ghi âm khi chuyển tab
    const startBtn = document.getElementById("startRecording");
    const stopBtn = document.getElementById("stopRecording");
    const testBtn = document.getElementById("testMicrophone");

    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    // Reset test microphone button
    if (testBtn) {
      testBtn.innerHTML = '<i class="fas fa-microphone"></i> Test Mic';
      testBtn.classList.remove("btn-success");
      testBtn.classList.add("btn-info-custom");
    }

    // Stop microphone capture if active
    if (this.isRecordingActive) {
      this.stopMicrophoneCapture();
    }
  }
  // File Upload Module
  setupFileUpload() {
    const uploadArea = document.getElementById("uploadArea");
    const fileInput = document.getElementById("audioFile");

    // Click to upload
    uploadArea.addEventListener("click", () => fileInput.click());

    // File selection
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        this.uploadFile(e.target.files[0]);
      }
    });

    // Drag and drop
    uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
      uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadArea.classList.remove("dragover");

      if (e.dataTransfer.files.length > 0) {
        this.uploadFile(e.dataTransfer.files[0]);
      }
    });
  }

  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      this.showProcessingStatus("Uploading file...");

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        this.currentFile = result;
        this.displayUploadedFileInfo(result);
        this.hideProcessingStatus();
        // Remove the success message - just show file info
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.hideProcessingStatus();
      this.showError("Upload failed: " + error.message);
    }
  }

  displayUploadedFileInfo(fileInfo) {
    // Update the upload area to show uploaded file info
    const uploadArea = document.getElementById("uploadArea");
    uploadArea.innerHTML = `
            <div class="alert alert-success">
                <h5><i class="fas fa-file-audio"></i> File đã upload: ${
                  fileInfo.filename
                }</h5>
                <div class="row mt-3">
                    <div class="col-md-3">
                        <strong>Thời lượng:</strong><br>
                        <span class="text-primary">${(
                          fileInfo.duration || 0
                        ).toFixed(2)}s</span>
                    </div>
                    <div class="col-md-3">
                        <strong>Sample Rate:</strong><br>
                        <span class="text-primary">${
                          fileInfo.sample_rate || "N/A"
                        } Hz</span>
                    </div>
                    <div class="col-md-3">
                        <strong>Samples:</strong><br>
                        <span class="text-primary">${(
                          fileInfo.total_samples || 0
                        ).toLocaleString()}</span>
                    </div>
                    <div class="col-md-3">
                        <strong>Channels:</strong><br>
                        <span class="text-primary">${
                          fileInfo.channels || 1
                        }</span>
                    </div>
                </div>
                <div class="mt-3">
                    <button class="btn btn-sm btn-outline-primary" onclick="location.reload()">
                        <i class="fas fa-upload"></i> Upload file khác
                    </button>
                </div>
            </div>
        `;

    // CLEAR previous genre classification results when new file is uploaded
    this.clearGenreResults();
  }

  displayFileInfo(fileInfo) {
    document.getElementById("fileName").textContent = fileInfo.filename;
    document.getElementById("fileDuration").textContent =
      fileInfo.duration.toFixed(2);
    document.getElementById("fileSampleRate").textContent =
      fileInfo.sample_rate;
    document.getElementById("fileRMS").textContent =
      fileInfo.rms_level.toFixed(4);

    document.getElementById("fileInfo").style.display = "block";

    // CLEAR previous genre classification results when new file is uploaded
    this.clearGenreResults();
  }

  clearGenreResults() {
    // Hide genre results
    document.getElementById("genreResults").style.display = "none";

    // Clear result values
    document.getElementById("predictedGenre").textContent = "";
    document.getElementById("genreConfidence").textContent = "";
    document.getElementById("methodUsed").textContent = "";

    // Reset confidence bar
    const confidenceBar = document.getElementById("confidenceBar");
    if (confidenceBar) {
      confidenceBar.style.width = "0%";
    }

    const confidenceText = document.getElementById("confidenceText");
    if (confidenceText) {
      confidenceText.textContent = "";
    }

    // ALSO clear noise reduction results when new file is uploaded
    this.clearNoiseResults();
  }

  clearNoiseResults() {
    // Hide noise analysis and comparison
    const noiseAnalysis = document.getElementById("noiseAnalysis");
    const noiseComparison = document.getElementById("noiseComparisonChart");

    if (noiseAnalysis) {
      noiseAnalysis.style.display = "none";
    }
    if (noiseComparison) {
      noiseComparison.style.display = "none";
      noiseComparison.innerHTML = ""; // Clear comparison content
    }
  }

  // Equalizer Module
  setupEqualizer() {
    // Slider value updates
    const sliders = [
      "subBass",
      "bass",
      "lowMid",
      "mid",
      "highMid",
      "presence",
      "brilliance",
      "air",
      "ultraHigh",
      "extreme",
    ];

    sliders.forEach((slider) => {
      const element = document.getElementById(slider);
      const valueElement = document.getElementById(slider + "Value");

      element.addEventListener("input", (e) => {
        valueElement.textContent = e.target.value + " dB";
        this.updateFrequencyResponse();
      });
    });

    // Preset loading with buttons
    const presetButtons = document.querySelectorAll(".eq-presets-buttons .btn");
    presetButtons.forEach((button) => {
      button.addEventListener("click", (e) => {
        e.preventDefault();

        // Remove active class from all buttons
        presetButtons.forEach((btn) => btn.classList.remove("active"));

        // Add active class to clicked button
        button.classList.add("active");

        // Load the preset
        this.loadEqualizerPresetByName(button.dataset.preset);
      });
    });

    // Processing
    document
      .getElementById("processEqualizer")
      .addEventListener("click", () => {
        this.processEqualizer();
      });

    // Reset
    document.getElementById("resetEqualizer").addEventListener("click", () => {
      this.resetEqualizer();
    });

    // Initialize frequency response chart
    this.initFrequencyResponseChart();
  }

  async loadEqualizerPreset() {
    const presetName = document.getElementById("eqPreset").value;
    if (!presetName) return;

    try {
      const response = await fetch("/api/equalizer/presets");
      const data = await response.json();

      if (data.presets[presetName]) {
        const gains = data.presets[presetName];

        // Map preset gains to sliders
        const mapping = {
          sub_bass: "subBass",
          bass: "bass",
          low_mid: "lowMid",
          mid: "mid",
          high_mid: "highMid",
          presence: "presence",
          brilliance: "brilliance",
          air: "air",
          ultra_high: "ultraHigh",
          extreme: "extreme",
        };

        Object.entries(mapping).forEach(([key, sliderId]) => {
          const slider = document.getElementById(sliderId);
          const valueElement = document.getElementById(sliderId + "Value");

          if (gains[key] !== undefined) {
            slider.value = gains[key];
            valueElement.textContent = gains[key] + " dB";
          }
        });

        this.updateFrequencyResponse();
        this.showSuccess(`Loaded preset: ${presetName}`);
      }
    } catch (error) {
      this.showError("Failed to load preset: " + error.message);
    }
  }

  async loadEqualizerPresetByName(presetName) {
    if (!presetName) return;

    try {
      const response = await fetch("/api/equalizer/presets");
      const data = await response.json();

      if (data.presets[presetName]) {
        const gains = data.presets[presetName];

        // Map preset gains to sliders
        const mapping = {
          sub_bass: "subBass",
          bass: "bass",
          low_mid: "lowMid",
          mid: "mid",
          high_mid: "highMid",
          presence: "presence",
          brilliance: "brilliance",
          air: "air",
          ultra_high: "ultraHigh",
          extreme: "extreme",
        };

        Object.entries(mapping).forEach(([key, sliderId]) => {
          const slider = document.getElementById(sliderId);
          const valueElement = document.getElementById(sliderId + "Value");

          if (gains[key] !== undefined) {
            slider.value = gains[key];
            valueElement.textContent = gains[key] + " dB";
          }
        });

        this.updateFrequencyResponse();
        this.showSuccess(`Loaded ${presetName} preset successfully!`);
      }
    } catch (error) {
      console.error("Error loading equalizer preset:", error);
      this.showError("Error loading equalizer preset");
    }
  }

  async processEqualizer() {
    if (!this.currentFile) {
      this.showError("Please upload an audio file first");
      return;
    }

    const gains = this.getEqualizerGains();
    const method = document.getElementById("eqMethod").value;
    const activePresetBtn = document.querySelector(
      ".eq-presets-buttons .btn.active"
    );
    const preset = activePresetBtn ? activePresetBtn.dataset.preset : null;

    try {
      this.showProcessingStatus("Applying equalizer...");

      const response = await fetch("/api/equalizer/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gains: gains,
          method: method,
          preset: preset || null,
        }),
      });

      const result = await response.json();

      if (result.success) {
        this.hideProcessingStatus();
        
        // Show results section
        document.getElementById("equalizerResults").style.display = "block";
        
        // Update audio players
        if (result.audio_files) {
          const originalAudio = document.getElementById("originalEqAudio");
          const processedAudio = document.getElementById("processedEqAudio");
          
          originalAudio.src = result.audio_files.original_url;
          processedAudio.src = result.audio_files.processed_url;
          
          // Update download links
          const downloadOriginal = document.getElementById("downloadOriginalEq");
          const downloadProcessed = document.getElementById("downloadProcessedEq");
          
          downloadOriginal.href = result.audio_files.original_url;
          downloadOriginal.download = result.audio_files.original;
          downloadOriginal.style.display = "block";
          
          downloadProcessed.href = result.audio_files.processed_url;
          downloadProcessed.download = result.audio_files.processed;
          downloadProcessed.style.display = "block";
        }
        
        // Generate enhanced visualizations
        this.generateEqualizerVisualizations(gains);
        
        // Update processing message
        document.getElementById("eqProcessingMessage").innerHTML = result.message || 'Equalizer processing complete!';
        
        const rmsChange = result.rms_change || "N/A";
        this.showSuccess(`Equalizer applied! RMS change: ${rmsChange} dB. Files saved to static/results/`);
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.hideProcessingStatus();
      this.showError("Equalizer processing failed: " + error.message);
    }
  }

  getEqualizerGains() {
    return {
      band_31_hz: parseFloat(document.getElementById("subBass").value),    // Sub-bass 31Hz
      band_62_hz: parseFloat(document.getElementById("bass").value),       // Bass 62Hz
      band_125_hz: parseFloat(document.getElementById("lowMid").value),    // Low-mid 125Hz
      band_250_hz: parseFloat(document.getElementById("mid").value),       // Mid 250Hz
      band_500_hz: parseFloat(document.getElementById("highMid").value),   // High-mid 500Hz
      band_1k_hz: parseFloat(document.getElementById("presence").value),   // Presence 1kHz
      band_2k_hz: parseFloat(document.getElementById("brilliance").value), // Brilliance 2kHz
      band_4k_hz: parseFloat(document.getElementById("air").value),        // Air 4kHz
      band_8k_hz: parseFloat(document.getElementById("ultraHigh").value),  // Ultra-high 8kHz
      band_16k_hz: parseFloat(document.getElementById("extreme").value),   // Extreme 16kHz
    };
  }

  resetEqualizer() {
    const sliders = [
      "subBass",
      "bass",
      "lowMid",
      "mid",
      "highMid",
      "presence",
      "brilliance",
      "air",
      "ultraHigh",
      "extreme",
    ];

    sliders.forEach((slider) => {
      document.getElementById(slider).value = 0;
      document.getElementById(slider + "Value").textContent = "0 dB";
    });

    // Reset preset buttons - activate "Flat" button
    const presetButtons = document.querySelectorAll(".eq-presets-buttons .btn");
    presetButtons.forEach((btn) => {
      btn.classList.remove("active");
      if (btn.dataset.preset === "flat") {
        btn.classList.add("active");
      }
    });

    this.updateFrequencyResponse();
  }

  initFrequencyResponseChart() {
    const ctx = document
      .getElementById("frequencyResponseChart")
      .getContext("2d");

    this.charts.frequencyResponse = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Frequency Response",
            data: [],
            borderColor: "rgb(102, 126, 234)",
            backgroundColor: "rgba(102, 126, 234, 0.1)",
            borderWidth: 2,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: "logarithmic",
            title: { display: true, text: "Frequency (Hz)" },
          },
          y: {
            title: { display: true, text: "Gain (dB)" },
            min: -25,
            max: 25,
          },
        },
      },
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
        16000: gains.extreme,
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
    document.getElementById("reductionLevel").addEventListener("input", (e) => {
      document.getElementById("reductionValue").textContent = e.target.value;
    });

    // New simplified buttons
    document.getElementById("processNoiseML").addEventListener("click", () => {
      this.processNoiseWithMethod("ml");
    });

    document
      .getElementById("processNoiseLibrary")
      .addEventListener("click", () => {
        this.processNoiseWithMethod("library");
      });
  }

  async processNoiseReduction() {
    if (!this.currentFile) {
      this.showError("Please upload an audio file first");
      return;
    }

    const method = document.getElementById("noiseMethod").value;
    const reductionLevel = parseFloat(
      document.getElementById("reductionLevel").value
    );

    try {
      this.showProcessingStatus("Reducing noise và tạo phân tích so sánh...");

      const response = await fetch("/api/noise_reduction/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method: method,
          reduction_level: reductionLevel,
        }),
      });

      const result = await response.json();

      if (result.success) {
        this.hideProcessingStatus();
        this.displayAdvancedNoiseResults(result);

        const snrImprovement =
          result.comparison_analysis?.comparison_metrics?.snr_improvement_db ||
          0;
        this.showSuccess(
          `✓ Noise reduction hoàn thành! SNR cải thiện: ${snrImprovement.toFixed(
            2
          )} dB`
        );
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.hideProcessingStatus();
      this.showError("Noise reduction failed: " + error.message);
    }
  }

  async analyzeNoise() {
    if (!this.currentFile) {
      this.showError("Please upload an audio file first");
      return;
    }

    // This would typically call a specific analyze endpoint
    // For now, we'll show the analysis from the last processing
    document.getElementById("noiseAnalysis").style.display = "block";
  }

  displayAdvancedNoiseResults(result) {
    const comparisonAnalysis = result.comparison_analysis;
    const audioFiles = result.audio_files;

    // Check if comparison_analysis exists and has required properties
    if (!comparisonAnalysis) {
      this.showError("Phân tích so sánh không thành công. Vui lòng thử lại.");
      return;
    }

    // Validate that we have metrics data with fallback handling
    const originalMetrics = comparisonAnalysis.original_metrics || {};
    const processedMetrics = comparisonAnalysis.processed_metrics || {};

    // Check if analysis failed but we still have fallback data
    if (originalMetrics.analysis_failed || processedMetrics.analysis_failed) {
      console.warn(
        "Audio analysis had issues but continuing with available data"
      );
      this.showWarning(
        "Phân tích audio gặp một số vấn đề nhưng vẫn hiển thị kết quả có sẵn."
      );
    }

    // Hiển thị 2 file audio để người dùng có thể nghe so sánh
    if (audioFiles) {
      this.displayAudioComparisonPlayer(audioFiles);
    }

    // Hiển thị metrics so sánh
    this.displayComparisonMetrics(comparisonAnalysis);

    // Hiển thị giải thích kỹ thuật chi tiết
    if (comparisonAnalysis.technical_explanation) {
      this.displayTechnicalExplanation(
        comparisonAnalysis.technical_explanation
      );
    }

    // Hiển thị biểu đồ so sánh
    if (comparisonAnalysis.comparison_chart_path) {
      this.displayAdvancedComparisonChart(
        comparisonAnalysis.comparison_chart_path
      );
    }

    document.getElementById("noiseAnalysis").style.display = "block";
  }

  displayAudioComparisonPlayer(audioFiles) {
    const container =
      document.getElementById("audioComparisonPlayer") ||
      this.createAudioComparisonContainer();

    container.innerHTML = `
      <div class="row">
        <div class="col-md-6">
          <div class="card">
            <div class="card-header bg-danger text-white">
              <h6 class="mb-0">🔊 Audio Gốc (Có Nhiễu)</h6>
            </div>
            <div class="card-body">
              <audio controls class="w-100" preload="metadata">
                <source src="/api/audio/download/${audioFiles.original}" type="audio/wav">
                Trình duyệt không hỗ trợ audio player
              </audio>
              <small class="text-muted d-block mt-2">
                File: ${audioFiles.original}<br>
                Đây là sample âm thanh gốc chưa được xử lý noise reduction
              </small>
            </div>
          </div>
        </div>
        <div class="col-md-6">
          <div class="card">
            <div class="card-header bg-success text-white">
              <h6 class="mb-0">🎵 Audio Đã Xử Lý (Giảm Nhiễu)</h6>
            </div>
            <div class="card-body">
              <audio controls class="w-100" preload="metadata">
                <source src="/api/audio/download/${audioFiles.processed}" type="audio/wav">
                Trình duyệt không hỗ trợ audio player
              </audio>
              <small class="text-muted d-block mt-2">
                File: ${audioFiles.processed}<br>
                Sample đã được xử lý bằng thuật toán noise reduction
              </small>
            </div>
          </div>
        </div>
      </div>
      <div class="alert alert-info mt-3">
        <strong>💡 Hướng dẫn:</strong> Hãy nghe cả 2 file để cảm nhận sự khác biệt bằng tai. 
        Audio bên trái là gốc (có nhiễu), audio bên phải đã được xử lý giảm nhiễu.
      </div>
    `;
  }

  displayComparisonMetrics(analysis) {
    const container =
      document.getElementById("comparisonMetrics") ||
      this.createComparisonMetricsContainer();

    const original = analysis.original_metrics || {};
    const processed = analysis.processed_metrics || {};
    const comparison = analysis.comparison_metrics || {};

    // Helper function to safely get numeric values with fallbacks
    const safeValue = (value, fallback = 0, decimals = 2) => {
      return value != null && !isNaN(value)
        ? Number(value).toFixed(decimals)
        : fallback;
    };

    container.innerHTML = `
      <div class="row">
        <div class="col-md-4">
          <div class="card border-danger">
            <div class="card-header bg-danger text-white">
              <h6 class="mb-0">📊 Audio Gốc</h6>
            </div>
            <div class="card-body">
              <div class="metric-item">
                <strong>SNR:</strong> ${safeValue(
                  original.snr_estimate,
                  "N/A",
                  1
                )} dB
              </div>
              <div class="metric-item">
                <strong>RMS Level:</strong> ${safeValue(
                  original.rms_level,
                  "N/A",
                  4
                )}
              </div>
              <div class="metric-item">
                <strong>Dynamic Range:</strong> ${safeValue(
                  original.dynamic_range,
                  "N/A",
                  1
                )} dB
              </div>
              <div class="metric-item">
                <strong>Noise Floor:</strong> ${safeValue(
                  original.noise_floor,
                  "N/A",
                  4
                )}
              </div>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card border-success">
            <div class="card-header bg-success text-white">
              <h6 class="mb-0">📈 Audio Đã Xử Lý</h6>
            </div>
            <div class="card-body">
              <div class="metric-item">
                <strong>SNR:</strong> ${safeValue(
                  processed.snr_estimate,
                  "N/A",
                  1
                )} dB
              </div>
              <div class="metric-item">
                <strong>RMS Level:</strong> ${safeValue(
                  processed.rms_level,
                  "N/A",
                  4
                )}
              </div>
              <div class="metric-item">
                <strong>Dynamic Range:</strong> ${safeValue(
                  processed.dynamic_range,
                  "N/A",
                  1
                )} dB
              </div>
              <div class="metric-item">
                <strong>Noise Floor:</strong> ${safeValue(
                  processed.noise_floor,
                  "N/A",
                  4
                )}
              </div>
              </div>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card border-primary">
            <div class="card-header bg-primary text-white">
              <h6 class="mb-0">🔄 Cải Thiện</h6>
            </div>
            <div class="card-body">
              <div class="metric-item ${
                (comparison.snr_improvement_db || 0) > 0
                  ? "text-success"
                  : "text-warning"
              }">
                <strong>SNR Cải Thiện:</strong> ${
                  (comparison.snr_improvement_db || 0) > 0 ? "+" : ""
                }${safeValue(comparison.snr_improvement_db, "0.0", 1)} dB
              </div>
              <div class="metric-item ${
                (comparison.rms_reduction_percent || 0) > 0
                  ? "text-success"
                  : "text-warning"
              }">
                <strong>RMS Giảm:</strong> ${safeValue(
                  comparison.rms_reduction_percent,
                  "0.0",
                  1
                )}%
              </div>
              <div class="metric-item">
                <strong>Noise Floor Giảm:</strong> ${safeValue(
                  comparison.noise_floor_reduction,
                  "0.0",
                  4
                )}
              </div>
              <div class="metric-item">
                <strong>Dynamic Range:</strong> ${
                  (comparison.dynamic_range_change || 0) > 0 ? "+" : ""
                }${safeValue(comparison.dynamic_range_change, "0.0", 1)} dB
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  displayTechnicalExplanation(technicalExplanation) {
    const container =
      document.getElementById("technicalExplanation") ||
      this.createTechnicalExplanationContainer();

    const method = technicalExplanation.method_description;
    const steps = technicalExplanation.processing_steps;
    const params = technicalExplanation.parameter_explanation;
    const results = technicalExplanation.results_interpretation;

    container.innerHTML = `
      <div class="accordion" id="technicalAccordion">
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#methodCollapse">
              🔬 Phương Pháp Sử Dụng: ${method.name}
            </button>
          </h2>
          <div id="methodCollapse" class="accordion-collapse collapse show">
            <div class="accordion-body">
              <p><strong>Mô tả:</strong> ${method.description}</p>
              <p><strong>Ưu điểm:</strong> ${method.advantages}</p>
              <p><strong>Phù hợp cho:</strong> ${method.suitable_for}</p>
            </div>
          </div>
        </div>
        
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#stepsCollapse">
              ⚙️ Các Bước Xử Lý (${steps.length} bước)
            </button>
          </h2>
          <div id="stepsCollapse" class="accordion-collapse collapse">
            <div class="accordion-body">
              <ol class="list-group list-group-numbered">
                ${steps
                  .map((step) => `<li class="list-group-item">${step}</li>`)
                  .join("")}
              </ol>
            </div>
          </div>
        </div>
        
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#paramsCollapse">
              📋 Parameters & Cấu Hình
            </button>
          </h2>
          <div id="paramsCollapse" class="accordion-collapse collapse">
            <div class="accordion-body">
              ${Object.entries(params)
                .map(
                  ([key, value]) =>
                    `<div class="mb-2"><strong>${key}:</strong> ${value}</div>`
                )
                .join("")}
            </div>
          </div>
        </div>
        
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#resultsCollapse">
              📊 Đánh Giá Kết Quả - ${results.quality_assessment}
            </button>
          </h2>
          <div id="resultsCollapse" class="accordion-collapse collapse">
            <div class="accordion-body">
              <div class="alert alert-${
                results.quality_assessment === "Excellent"
                  ? "success"
                  : results.quality_assessment === "Good"
                  ? "info"
                  : "warning"
              }">
                <strong>Chất lượng:</strong> ${results.quality_assessment}
              </div>
              <p><strong>SNR:</strong> ${results.snr_explanation}</p>
              <p><strong>RMS:</strong> ${results.rms_explanation}</p>
              <div class="alert alert-info">
                <strong>💡 Khuyến nghị:</strong> ${results.recommendation}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  displayAdvancedComparisonChart(chartPath) {
    const container =
      document.getElementById("advancedComparisonChart") ||
      this.createAdvancedChartContainer();

    // Extract filename from full path
    const filename = chartPath.split("/").pop().split("\\").pop();

    container.innerHTML = `
      <div class="card">
        <div class="card-header bg-primary text-white">
          <h6 class="mb-0">📈 Biểu Đồ Phân Tích Chi Tiết - Advanced Visualization</h6>
        </div>
        <div class="card-body">
          <img src="/${chartPath}?${new Date().getTime()}" 
               class="img-fluid w-100" 
               alt="Advanced Noise Reduction Analysis"
               style="max-height: 1000px; object-fit: contain; border: 2px solid #dee2e6; border-radius: 8px;">
          <div class="mt-4">
            <div class="alert alert-info">
              <h6><i class="fas fa-chart-line"></i> Chi Tiết Các Biểu Đồ:</h6>
              <div class="row">
                <div class="col-md-6">
                  <strong>📊 Waveform & Spectrum:</strong><br>
                  • So sánh dạng sóng âm thanh gốc và đã xử lý<br>
                  • Phân tích phổ tần số để thấy nhiễu bị loại bỏ<br><br>
                  
                  <strong>🌈 Spectrograms:</strong><br>
                  • Hình ảnh 2D của âm thanh theo thời gian<br>
                  • Màu sáng = cường độ cao, tối = ít năng lượng<br>
                  • So sánh trực quan nhiễu trước/sau
                </div>
                <div class="col-md-6">
                  <strong>📈 Metrics & Analysis:</strong><br>
                  • SNR, RMS, Dynamic Range comparison<br>
                  • Hiệu quả giảm nhiễu theo thời gian<br><br>
                  
                  <strong>🎯 Quality Assessment:</strong><br>
                  • Đánh giá chất lượng tổng thể<br>
                  • Thông số kỹ thuật chi tiết<br>
                  • Khuyến nghị cải thiện
                </div>
              </div>
            </div>
            <div class="text-center mt-3">
              <a href="/${chartPath}" target="_blank" class="btn btn-outline-primary">
                <i class="fas fa-external-link-alt"></i> Xem Ảnh Kích Thước Đầy Đủ
              </a>
              <a href="/${chartPath}" download="${filename}" class="btn btn-outline-success ms-2">
                <i class="fas fa-download"></i> Tải Về Biểu Đồ
              </a>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  createAudioComparisonContainer() {
    const container = document.createElement("div");
    container.id = "audioComparisonPlayer";
    container.className = "mt-4";

    const noiseAnalysis = document.getElementById("noiseAnalysis");
    noiseAnalysis.appendChild(container);

    return container;
  }

  createComparisonMetricsContainer() {
    const container = document.createElement("div");
    container.id = "comparisonMetrics";
    container.className = "mt-4";

    const noiseAnalysis = document.getElementById("noiseAnalysis");
    noiseAnalysis.appendChild(container);

    return container;
  }

  createTechnicalExplanationContainer() {
    const container = document.createElement("div");
    container.id = "technicalExplanation";
    container.className = "mt-4";

    const noiseAnalysis = document.getElementById("noiseAnalysis");
    noiseAnalysis.appendChild(container);

    return container;
  }

  createAdvancedChartContainer() {
    const container = document.createElement("div");
    container.id = "advancedComparisonChart";
    container.className = "mt-4";

    const noiseAnalysis = document.getElementById("noiseAnalysis");
    noiseAnalysis.appendChild(container);

    return container;
  }

  displayNoiseResults(result) {
    const analysis = result.original_analysis || {};

    // Populate simplified noise results with safe access
    document.getElementById("snrEstimate").textContent =
      analysis.snr_estimate?.toFixed(1) || "N/A";
    document.getElementById("dynamicRange").textContent =
      analysis.dynamic_range?.toFixed(1) || "N/A";

    // Show SNR improvement (key metric for đề bài) with safe access
    const snrImprovementElement = document.getElementById("snrImprovement");
    if (snrImprovementElement && result.snr_improvement != null) {
      snrImprovementElement.textContent =
        "+" + Number(result.snr_improvement).toFixed(1);
    }

    // Show method used
    const methodElement = document.getElementById("methodUsedNoise");
    if (methodElement) {
      const methodName =
        result.method === "autoencoder"
          ? "AI/ML Neural Network"
          : "Advanced Library";
      methodElement.textContent = methodName;
    }

    document.getElementById("noiseAnalysis").style.display = "block";

    // Display simple before/after comparison if available
    if (result.comparison_plot) {
      this.displayNoiseComparisonChart(result);
    }
  }

  displayNoiseComparisonChart(result) {
    // Show the comparison image generated by backend
    const comparisonContainer = document.getElementById("noiseComparisonChart");

    // Create or update the comparison image
    let comparisonImg = comparisonContainer.querySelector(".comparison-image");
    if (!comparisonImg) {
      comparisonImg = document.createElement("img");
      comparisonImg.className = "comparison-image img-fluid";
      comparisonImg.style.width = "100%";
      comparisonImg.style.maxHeight = "500px";
      comparisonImg.style.objectFit = "contain";
      comparisonContainer.innerHTML = ""; // Clear canvas
      comparisonContainer.appendChild(comparisonImg);
    }

    // Set the image source to the generated plot
    comparisonImg.src =
      "/static/results/noise_reduction_comparison.png?" + new Date().getTime(); // Add timestamp to prevent caching
    comparisonImg.alt = "Noise Reduction Comparison - Before vs After";

    // Add title
    if (!comparisonContainer.querySelector(".comparison-title")) {
      const title = document.createElement("h5");
      title.className = "comparison-title text-center mt-3";
      title.innerHTML =
        '<i class="fas fa-chart-area"></i> Before vs After Comparison';
      comparisonContainer.insertBefore(title, comparisonImg);
    }

    comparisonContainer.style.display = "block";
  }

  async processNoiseWithMethod(method) {
    if (!this.currentFile) {
      this.showError("Vui lòng upload file audio trước");
      return;
    }

    const reductionLevel = parseFloat(
      document.getElementById("reductionLevel").value
    );

    try {
      let statusMessage =
        method === "ml"
          ? "Đang giảm nhiễu với AI/ML hệ thống..."
          : "Đang giảm nhiễu với thư viện tốt nhất...";

      this.showProcessingStatus(statusMessage);

      // Choose method based on type
      const noiseMethod = method === "ml" ? "autoencoder" : "noisereduce";

      const response = await fetch("/api/noise_reduction/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method: noiseMethod,
          reduction_level: reductionLevel,
        }),
      });

      const result = await response.json();

      if (result.success) {
        this.hideProcessingStatus();
        // Use the advanced display for detailed results
        this.displayAdvancedNoiseResults(result);

        const methodName =
          method === "ml" ? "AI/ML Hệ Thống" : "Thư Viện Tốt Nhất";

        // Safe access to snr_improvement
        const snrImprovement =
          result.comparison_analysis?.comparison_metrics?.snr_improvement_db !=
          null
            ? Number(
                result.comparison_analysis?.comparison_metrics
                  ?.snr_improvement_db
              ).toFixed(2)
            : "N/A";

        this.showSuccess(
          `Giảm nhiễu thành công! SNR cải thiện: ${snrImprovement} dB - ${methodName}`
        );
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.hideProcessingStatus();
      this.showError("Giảm nhiễu thất bại: " + error.message);
    }
  }

  // Genre Classification Module
  setupGenreClassification() {
    // New simplified buttons
    document.getElementById("classifyML").addEventListener("click", () => {
      this.classifyWithMethod("ml");
    });

    document.getElementById("classifyLibrary").addEventListener("click", () => {
      this.classifyWithMethod("library");
    });

    // Load model status
    this.loadModelInfo();
  }

  async classifyGenre() {
    if (!this.currentFile) {
      this.showError("Please upload an audio file first");
      return;
    }

    const method = document.getElementById("genreMethod").value;

    try {
      this.showProcessingStatus("Classifying genre...");

      const response = await fetch("/api/genre_classification/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method: method }),
      });

      const result = await response.json();

      if (result.success) {
        this.hideProcessingStatus();
        this.displayGenreResults(result);
        this.showSuccess(
          `Genre classified: ${result.predicted_genre} (${(
            result.confidence * 100
          ).toFixed(1)}%)`
        );
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.hideProcessingStatus();
      this.showError("Genre classification failed: " + error.message);
    }
  }

  displayGenreResults(result) {
    document.getElementById("predictedGenre").textContent =
      result.predicted_genre;
    document.getElementById("genreConfidence").textContent = (
      result.confidence * 100
    ).toFixed(1);
    document.getElementById("methodUsed").textContent = result.method;

    const confidenceBar = document.getElementById("confidenceBar");
    const confidenceValue = result.confidence * 100;
    confidenceBar.style.width = confidenceValue + "%";

    // Update confidence text in progress bar
    const confidenceText = document.getElementById("confidenceText");
    if (confidenceText) {
      confidenceText.textContent = confidenceValue.toFixed(1) + "%";
    }

    // Hiển thị giải thích confidence
    if (
      result.additional_info &&
      result.additional_info.confidence_explanation
    ) {
      const confidenceExplain = document.getElementById("confidenceExplain");
      if (confidenceExplain) {
        confidenceExplain.innerHTML = `<div class='alert alert-info mt-2'><strong>Giải thích độ tin cậy:</strong> ${result.additional_info.confidence_explanation}</div>`;
      }
    }

    // Hiển thị bảng báo cáo chi tiết
    if (result.additional_info && result.additional_info.detailed_report) {
      this.displayGenreDetailedReport(result.additional_info.detailed_report);
    }

    document.getElementById("genreResults").style.display = "block";
  }

  displayGenreDetailedReport(report) {
    const container = document.getElementById("genreDetailedReport");
    if (!container) return;
    let html = "";
    for (const [section, content] of Object.entries(report)) {
      html += `<div class='card mb-3'><div class='card-header bg-primary text-white'><strong>${section}</strong></div><div class='card-body'>`;
      if (typeof content === "object" && !Array.isArray(content)) {
        html += `<table class='table table-bordered table-sm'>`;
        for (const [key, value] of Object.entries(content)) {
          html += `<tr><td><strong>${key}</strong></td><td>${
            typeof value === "object" ? JSON.stringify(value) : value
          }</td></tr>`;
        }
        html += `</table>`;
      } else if (Array.isArray(content)) {
        html += `<ol class='list-group list-group-numbered'>`;
        for (const step of content) {
          html += `<li class='list-group-item'>${step}</li>`;
        }
        html += `</ol>`;
      } else {
        html += `<div>${content}</div>`;
      }
      html += `</div></div>`;
    }
    container.innerHTML = html;
  }

  displayAudioFeatures(features) {
    const featuresHtml = `
            <div class="alert alert-info mt-3">
                <h6><i class="fas fa-chart-line"></i> Phân Tích Đặc Trưng Âm Thanh Chi Tiết</h6>
                <div class="row">
                    <div class="col-md-4">
                        <h6>🎵 Đặc trưng Spectral</h6>
                        <p><strong>Spectral Centroid:</strong> ${features.spectral_centroid.toFixed(
                          0
                        )} Hz</p>
                        <p><strong>Spectral Rolloff:</strong> ${features.spectral_rolloff.toFixed(
                          0
                        )} Hz</p>
                        <p><strong>RMS Energy:</strong> ${features.rms_energy.toFixed(
                          3
                        )}</p>
                    </div>
                    <div class="col-md-4">
                        <h6>🎼 Đặc trưng Rhythm & Harmonic</h6>
                        <p><strong>Tempo:</strong> ${features.tempo.toFixed(
                          0
                        )} BPM</p>
                        <p><strong>Zero Crossing Rate:</strong> ${features.zero_crossing_rate.toFixed(
                          3
                        )}</p>
                        <p><strong>Harmonic Ratio:</strong> ${features.harmonic_ratio.toFixed(
                          2
                        )}</p>
                    </div>
                    <div class="col-md-4">
                        <h6>🎹 Đặc trưng MFCC & Chroma</h6>
                        ${
                          features.mfcc_characteristics
                            ? `
                            <p><strong>Timbre Brightness:</strong> ${features.mfcc_characteristics.timbre_brightness.toFixed(
                              2
                            )}</p>
                            <p><strong>Spectral Shape:</strong> ${features.mfcc_characteristics.spectral_shape.toFixed(
                              2
                            )}</p>
                        `
                            : ""
                        }
                        ${
                          features.chroma_characteristics
                            ? `
                            <p><strong>Dominant Pitch:</strong> ${this.getPitchClassName(
                              features.chroma_characteristics
                                .dominant_pitch_class
                            )}</p>
                            <p><strong>Harmonic Complexity:</strong> ${features.chroma_characteristics.harmonic_complexity.toFixed(
                              2
                            )}</p>
                        `
                            : ""
                        }
                    </div>
                </div>
                <div class="mt-2">
                    <small class="text-muted">
                        💡 <strong>Giải thích:</strong> Spectral Centroid là "trung tâm khối lượng" của frequency spectrum. 
                        Tempo là nhịp độ (beats per minute). Zero Crossing Rate phản ánh độ "gồ ghề" của âm thanh.
                        Harmonic Ratio cho biết tỷ lệ âm hài hòa so với percussion.
                    </small>
                </div>
            </div>
        `;

    // Insert after genre results
    const resultsDiv = document.getElementById("genreResults");

    // Remove existing features div if present
    const existingFeatures = resultsDiv.querySelector(".audio-features");
    if (existingFeatures) {
      existingFeatures.remove();
    }

    const featuresDiv = document.createElement("div");
    featuresDiv.className = "audio-features";
    featuresDiv.innerHTML = featuresHtml;
    resultsDiv.appendChild(featuresDiv);
  }

  getPitchClassName(pitchClass) {
    const pitchNames = [
      "C",
      "C#",
      "D",
      "D#",
      "E",
      "F",
      "F#",
      "G",
      "G#",
      "A",
      "A#",
      "B",
    ];
    return pitchNames[pitchClass] || "Unknown";
  }

  displayClassificationReasoning(reasoning) {
    if (!reasoning || reasoning.length === 0) return;

    const reasoningHtml = `
            <div class="alert alert-success mt-3">
                <h6><i class="fas fa-brain"></i> Lý Do Phân Loại</h6>
                <ul class="mb-0">
                    ${reasoning.map((reason) => `<li>${reason}</li>`).join("")}
                </ul>
            </div>
        `;

    // Insert after genre results
    const resultsDiv = document.getElementById("genreResults");

    // Remove existing reasoning div if present
    const existingReasoning = resultsDiv.querySelector(
      ".classification-reasoning"
    );
    if (existingReasoning) {
      existingReasoning.remove();
    }

    const reasoningDiv = document.createElement("div");
    reasoningDiv.className = "classification-reasoning";
    reasoningDiv.innerHTML = reasoningHtml;
    resultsDiv.appendChild(reasoningDiv);
  }

  displayMLAnalysis(mlAnalysis) {
    if (!mlAnalysis) return;

    const analysisHtml = `
            <div class="alert alert-primary mt-3">
                <h6><i class="fas fa-robot"></i> Phân Tích Machine Learning Chi Tiết</h6>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>🔬 Tổng số features:</strong> ${
                          mlAnalysis.total_features_extracted
                        }</p>
                        <p><strong>📊 Độ tin cậy model:</strong> ${(
                          mlAnalysis.model_confidence * 100
                        ).toFixed(1)}%</p>
                        <p><strong>🧠 Loại features:</strong></p>
                        <ul class="small">
                            ${mlAnalysis.feature_types
                              .map((type) => `<li>${type}</li>`)
                              .join("")}
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <p><strong>🏆 Top 3 dự đoán:</strong></p>
                        <ol class="small">
                            ${mlAnalysis.top_3_predictions
                              .map(
                                ([genre, prob]) =>
                                  `<li>${genre}: ${(prob * 100).toFixed(
                                    1
                                  )}%</li>`
                              )
                              .join("")}
                        </ol>
                    </div>
                </div>
            </div>
        `;

    // Insert after genre results
    const resultsDiv = document.getElementById("genreResults");

    // Remove existing ML analysis div if present
    const existingAnalysis = resultsDiv.querySelector(".ml-analysis");
    if (existingAnalysis) {
      existingAnalysis.remove();
    }

    const analysisDiv = document.createElement("div");
    analysisDiv.className = "ml-analysis";
    analysisDiv.innerHTML = analysisHtml;
    resultsDiv.appendChild(analysisDiv);
  }

  displayDatasetInfo(datasetInfo) {
    if (!datasetInfo) return;

    const datasetHtml = `
            <div class="alert alert-warning mt-3">
                <h6><i class="fas fa-database"></i> Chi Tiết Dataset Training</h6>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>📂 Dataset:</strong> ${datasetInfo.name}</p>
                        <p><strong>🎵 Tổng samples:</strong> ${
                          datasetInfo.total_samples
                        } bài nhạc</p>
                        <p><strong>⚖️ Phân bố:</strong> ${
                          datasetInfo.samples_per_genre
                        } bài/genre</p>
                        <p><strong>⏱️ Độ dài:</strong> ${
                          datasetInfo.audio_length
                        }</p>
                        <p><strong>🤖 Algorithm:</strong> ${
                          datasetInfo.training_method
                        }</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>🎯 Accuracy:</strong> ${
                          datasetInfo.average_accuracy
                        }</p>
                        <p><strong>✅ Validation:</strong> ${
                          datasetInfo.cross_validation
                        }</p>
                        <p><strong>🏷️ Genres support:</strong></p>
                        <div class="d-flex flex-wrap gap-1">
                            ${datasetInfo.genres
                              .map(
                                (genre) =>
                                  `<span class="badge bg-secondary">${genre}</span>`
                              )
                              .join("")}
                        </div>
                    </div>
                </div>
                
                <div class="mt-3">
                    <h6><strong>🔍 Đặc Trưng Quan Trọng Nhất:</strong></h6>
                    <ul class="small">
                        ${datasetInfo.feature_importance.most_important_features
                          .map((feature) => `<li>${feature}</li>`)
                          .join("")}
                    </ul>
                </div>

                <div class="mt-3">
                    <h6><strong>🎼 Chữ Ký Âm Nhạc Của Từng Genre:</strong></h6>
                    <div class="accordion" id="genreSignatures">
                        ${Object.entries(
                          datasetInfo.feature_importance.genre_signatures
                        )
                          .map(
                            ([genre, signature], index) => `
                            <div class="accordion-item">
                                <h2 class="accordion-header" id="heading${index}">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                                            data-bs-target="#collapse${index}" aria-expanded="false">
                                        <strong>${genre.toUpperCase()}</strong>
                                    </button>
                                </h2>
                                <div id="collapse${index}" class="accordion-collapse collapse" 
                                     data-bs-parent="#genreSignatures">
                                    <div class="accordion-body small">
                                        ${signature}
                                    </div>
                                </div>
                            </div>
                        `
                          )
                          .join("")}
                    </div>
                </div>

                <div class="mt-3">
                    <h6><strong>💡 Training Insights:</strong></h6>
                    <ul class="small">
                        ${datasetInfo.feature_importance.training_insights
                          .map((insight) => `<li>${insight}</li>`)
                          .join("")}
                    </ul>
                </div>
            </div>
        `;

    const resultsDiv = document.getElementById("genreResults");
    const existingDataset = resultsDiv.querySelector(".dataset-info");
    if (existingDataset) {
      existingDataset.remove();
    }

    const datasetDiv = document.createElement("div");
    datasetDiv.className = "dataset-info";
    datasetDiv.innerHTML = datasetHtml;
    resultsDiv.appendChild(datasetDiv);
  }

  displayTechnicalDetails(technicalDetails) {
    if (!technicalDetails) return;

    const technicalHtml = `
            <div class="alert alert-dark mt-3">
                <h6><i class="fas fa-cogs"></i> Chi Tiết Kỹ Thuật ML Pipeline</h6>
                
                <div class="row">
                    <div class="col-md-4">
                        <h6><strong>🔧 Preprocessing:</strong></h6>
                        <ul class="small">
                            ${technicalDetails.preprocessing
                              .map((step) => `<li>${step}</li>`)
                              .join("")}
                        </ul>
                    </div>
                    <div class="col-md-4">
                        <h6><strong>📊 Feature Extraction:</strong></h6>
                        <ul class="small">
                            ${technicalDetails.feature_extraction
                              .map((feature) => `<li>${feature}</li>`)
                              .join("")}
                        </ul>
                    </div>
                    <div class="col-md-4">
                        <h6><strong>🤖 Model Details:</strong></h6>
                        <ul class="small">
                            <li><strong>Algorithm:</strong> ${
                              technicalDetails.model_details.algorithm
                            }</li>
                            <li><strong>Trees:</strong> ${
                              technicalDetails.model_details.n_estimators
                            }</li>
                            <li><strong>Max Depth:</strong> ${
                              technicalDetails.model_details.max_depth ||
                              "Unlimited"
                            }</li>
                            <li><strong>Scaling:</strong> ${
                              technicalDetails.model_details.feature_scaling
                            }</li>
                            <li><strong>Prediction:</strong> ${
                              technicalDetails.model_details.prediction_method
                            }</li>
                        </ul>
                    </div>
                </div>
                
                <div class="alert alert-info mt-3">
                    <small>
                        <strong>🧠 Giải thích:</strong> Random Forest sử dụng 100 decision trees, mỗi tree 
                        vote cho 1 genre. Genre có nhiều vote nhất sẽ được chọn. Feature scaling đảm bảo 
                        tất cả features có cùng tầm quan trọng trong quá trình training.
                    </small>
                </div>
            </div>
        `;

    const resultsDiv = document.getElementById("genreResults");
    const existingTechnical = resultsDiv.querySelector(".technical-details");
    if (existingTechnical) {
      existingTechnical.remove();
    }

    const technicalDiv = document.createElement("div");
    technicalDiv.className = "technical-details";
    technicalDiv.innerHTML = technicalHtml;
    resultsDiv.appendChild(technicalDiv);
  }

  displayProbabilityTable(probabilities) {
    const tbody = document.getElementById("probabilityTable");
    tbody.innerHTML = "";

    // Sort by probability
    const sortedProbs = Object.entries(probabilities).sort(
      ([, a], [, b]) => b - a
    );

    sortedProbs.forEach(([genre, prob]) => {
      const row = document.createElement("tr");
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
      const response = await fetch("/api/genre_classification/info");
      const info = await response.json();

      // Update model status indicators
      const models = ["rf", "svm", "nn", "lstm", "cnn"];
      const modelNames = [
        "random_forest",
        "svm",
        "neural_network",
        "lstm",
        "cnn",
      ];

      models.forEach((model, index) => {
        const statusElement = document.getElementById(model + "ModelStatus");
        if (statusElement) {
          const isLoaded =
            info.models_loaded && info.models_loaded[modelNames[index]];
          statusElement.className =
            "status-indicator " +
            (isLoaded ? "status-active" : "status-inactive");
        }
      });
    } catch (error) {
      console.warn("Could not load model info:", error);
    }
  }

  async classifyWithMethod(method) {
    if (!this.currentFile) {
      this.showError("Vui lòng upload file audio trước");
      return;
    }

    try {
      let statusMessage =
        method === "ml"
          ? "Đang phân tích với AI/ML hệ thống..."
          : "Đang phân tích với thư viện tốt nhất...";

      this.showProcessingStatus(statusMessage);

      const endpoint = "/api/genre_classification/classify_best";

      const body =
        method === "ml"
          ? JSON.stringify({ option: "option2" }) // AI/ML Hệ Thống = Custom ML
          : JSON.stringify({ option: "option1" }); // Thư Viện Tốt Nhất = Advanced Librosa

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
      });

      const result = await response.json();

      if (result.success) {
        this.hideProcessingStatus();
        this.displayGenreResults(result);

        const methodName =
          method === "ml" ? "AI/ML Hệ Thống" : "Thư Viện Tốt Nhất";
        this.showSuccess(
          `Kết quả: ${result.predicted_genre} (${(
            result.confidence * 100
          ).toFixed(1)}%) - ${methodName}`
        );
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.hideProcessingStatus();
      this.showError("Phân tích thất bại: " + error.message);
    }
  }

  displayBestClassificationResults(result) {
    // Update basic results
    document.getElementById("predictedGenre").textContent =
      result.predicted_genre;
    document.getElementById("genreConfidence").textContent = (
      result.confidence * 100
    ).toFixed(1);
    document.getElementById("methodUsed").textContent =
      result.method || "Advanced Classification";

    const confidenceBar = document.getElementById("confidenceBar");
    confidenceBar.style.width = result.confidence * 100 + "%";

    document.getElementById("genreResults").style.display = "block";

    // Show comparison results if both methods were tested
    if (result.comparison) {
      this.displayComparisonResults(result.comparison);
    }

    // Show detailed analysis if available
    if (result.additional_info) {
      if (result.additional_info.ensemble_probabilities) {
        this.displayProbabilityTable(
          result.additional_info.ensemble_probabilities
        );
        document.getElementById("detailedAnalysis").style.display = "block";
      }
    }
  }

  displayComparisonResults(comparison) {
    // Create comparison display if it doesn't exist
    let comparisonDiv = document.getElementById("comparisonResults");
    if (!comparisonDiv) {
      comparisonDiv = document.createElement("div");
      comparisonDiv.id = "comparisonResults";
      comparisonDiv.className = "mt-3";
      document.getElementById("genreResults").appendChild(comparisonDiv);
    }

    comparisonDiv.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h6 class="mb-0"><i class="fas fa-chart-bar"></i> Method Comparison</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="comparison-item ${
                              comparison.best_method === "Musicnn"
                                ? "winner"
                                : ""
                            }">
                                <strong>🤖 Musicnn Deep Learning</strong><br>
                                <span class="text-info">${
                                  comparison.option1_result.predicted_genre
                                }</span>
                                <span class="confidence-badge">${(
                                  comparison.option1_result.confidence * 100
                                ).toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="comparison-item ${
                              comparison.best_method === "Custom ML"
                                ? "winner"
                                : ""
                            }">
                                <strong>🎯 Custom ML</strong><br>
                                <span class="text-success">${
                                  comparison.option2_result.predicted_genre
                                }</span>
                                <span class="confidence-badge">${(
                                  comparison.option2_result.confidence * 100
                                ).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                    <div class="text-center mt-2">
                        <span class="badge bg-primary">Winner: ${
                          comparison.best_method
                        }</span>
                    </div>
                </div>
            </div>
        `;
    comparisonDiv.style.display = "block";
  }

  // Real-time Processing Module
  setupRealtimeProcessing() {
    console.log("🔧 Setting up Real-time Processing Module");

    // Device management - add null checks
    const refreshDevicesBtn = document.getElementById("refreshDevices");
    if (refreshDevicesBtn) {
      refreshDevicesBtn.addEventListener("click", () => {
        this.loadAudioDevices();
      });
    }

    const testLatencyBtn = document.getElementById("testLatency");
    if (testLatencyBtn) {
      testLatencyBtn.addEventListener("click", () => {
        this.testLatency();
      });
    }

    // Real-time control
    document.getElementById("startRealtime").addEventListener("click", () => {
      console.log("🔥 Start Realtime button clicked!");
      this.startRealtimeProcessing();
    });

    document.getElementById("stopRealtime").addEventListener("click", () => {
      console.log("⏹️ Stop Realtime button clicked!");
      this.stopRealtimeProcessing();
    });

    // Recording controls trong tab Real-time
    document.getElementById("startRecording").addEventListener("click", () => {
      this.startRealtimeRecording();
    });

    document.getElementById("stopRecording").addEventListener("click", () => {
      this.stopRealtimeRecording();
    });

    // Test microphone button
    document.getElementById("testMicrophone").addEventListener("click", () => {
      this.toggleMicrophoneTest();
    });

    // Real-time Audio Processing Controls
    this.setupRealtimeControls();

    // Initialize audio visualizer
    this.initAudioVisualizer();

    console.log("✅ Real-time Processing Module setup complete");
  }

  setupRealtimeControls() {
    console.log("🎛️ Setting up real-time audio processing controls");

    // EQ Controls
    const eqCheckboxes = ["lowCut", "highCut"];
    eqCheckboxes.forEach((id) => {
      const element = document.getElementById(id);
      if (element) {
        element.addEventListener("change", () => {
          console.log(`📊 EQ Control ${id} changed:`, element.checked);
          this.updateRealtimeEffects();
        });
      }
    });

    // Noise Reduction Controls
    const noiseCheckbox = document.getElementById("denoise");
    if (noiseCheckbox) {
      noiseCheckbox.addEventListener("change", () => {
        console.log("🔇 Denoise changed:", noiseCheckbox.checked);
        this.updateRealtimeEffects();
      });
    }

    // Filter Type Controls (if exists)
    const filterControls = ["iir", "fir"];
    filterControls.forEach((id) => {
      const element = document.getElementById(id);
      if (element) {
        element.addEventListener("change", () => {
          console.log(`🎚️ Filter ${id} changed:`, element.checked);
          this.updateRealtimeEffects();
        });
      }
    });
  }

  async updateRealtimeEffects() {
    if (!this.isRealtimeActive) return;

    console.log("🔄 Updating real-time effects...");

    // Get current state of all controls
    const effects = {
      lowCut: document.getElementById("lowCut")?.checked || false,
      highCut: document.getElementById("highCut")?.checked || false,
      denoise: document.getElementById("denoise")?.checked || false,
      iir: document.getElementById("iir")?.checked || false,
      fir: document.getElementById("fir")?.checked || false,
    };

    try {
      const response = await fetch("/api/realtime/update_effects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(effects),
      });

      if (response.ok) {
        console.log("✅ Real-time effects updated", effects);
      } else {
        console.error("❌ Failed to update effects:", response.status);
      }
    } catch (error) {
      console.error("❌ Error updating effects:", error);
    }
  }

  async loadAudioDevices() {
    try {
      const response = await fetch("/api/audio_devices");
      const devices = await response.json();

      // Populate input devices
      const inputSelect = document.getElementById("inputDevice");
      inputSelect.innerHTML = '<option value="">Default Input Device</option>';
      devices.input?.forEach((device) => {
        const option = document.createElement("option");
        option.value = device.index;
        option.textContent = `${device.name} (${device.channels} ch)`;
        inputSelect.appendChild(option);
      });

      // Populate output devices
      const outputSelect = document.getElementById("outputDevice");
      outputSelect.innerHTML =
        '<option value="">Default Output Device</option>';
      devices.output?.forEach((device) => {
        const option = document.createElement("option");
        option.value = device.index;
        option.textContent = `${device.name} (${device.channels} ch)`;
        outputSelect.appendChild(option);
      });
    } catch (error) {
      this.showError("Failed to load audio devices: " + error.message);
    }
  }

  async startRealtimeProcessing() {
    console.log("🎯 startRealtimeProcessing() called");

    const equalizerParams = this.getEqualizerGains();
    const noiseMethodEl = document.getElementById("noiseMethod");
    const reductionLevelEl = document.getElementById("reductionLevel");

    const noiseMethod = noiseMethodEl ? noiseMethodEl.value : "spectral";
    const noiseLevel = reductionLevelEl
      ? parseFloat(reductionLevelEl.value)
      : 0.7;

    const enabledModules = {
      equalizer: document.getElementById("enableEqualizer")?.checked || true,
      noise_reduction:
        document.getElementById("enableNoiseReduction")?.checked || true,
      genre_classification:
        document.getElementById("enableGenreClassification")?.checked || true,
    };

    console.log("📊 Realtime params:", {
      equalizerParams,
      noiseMethod,
      noiseLevel,
      enabledModules,
    });

    try {
      console.log("📡 Sending request to /api/realtime/start");

      const response = await fetch("/api/realtime/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          equalizer_params: equalizerParams,
          noise_method: noiseMethod,
          noise_reduction_level: noiseLevel,
          enabled_modules: enabledModules,
        }),
      });

      console.log("📨 Response received:", response.status);
      const result = await response.json();
      console.log("📋 Response data:", result);

      if (result.success) {
        this.isRealtimeActive = true;
        document.getElementById("startRealtime").disabled = true;
        document.getElementById("stopRealtime").disabled = false;
        document.getElementById("realtimeStats").style.display = "block";

        this.showSuccess("Real-time processing started!");
        this.startStatsUpdater();

        // Start microphone capture for visualization
        console.log(
          "🎤 Starting microphone capture for real-time visualization"
        );
        await this.initMicrophoneCapture();
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      console.error("❌ Realtime start error:", error);
      this.showError("Failed to start real-time processing: " + error.message);
    }
  }

  async stopRealtimeProcessing() {
    try {
      const response = await fetch("/api/realtime/stop", { method: "POST" });
      const result = await response.json();

      if (result.success) {
        this.isRealtimeActive = false;
        document.getElementById("startRealtime").disabled = false;
        document.getElementById("stopRealtime").disabled = true;

        this.showSuccess("Real-time processing stopped");
        this.stopStatsUpdater();
      }
    } catch (error) {
      this.showError("Failed to stop real-time processing: " + error.message);
    }
  }

  startStatsUpdater() {
    if (this.statsInterval) clearInterval(this.statsInterval);

    this.statsInterval = setInterval(async () => {
      if (!this.isRealtimeActive) return;

      try {
        const response = await fetch("/api/realtime/stats");
        const stats = await response.json();

        document.getElementById("avgLatency").textContent =
          stats.avg_latency_ms?.toFixed(1) + " ms" || "0 ms";
        document.getElementById("chunksProcessed").textContent =
          stats.chunks_processed || "0";
        document.getElementById("processingErrors").textContent =
          stats.processing_errors || "0";
      } catch (error) {
        console.warn("Failed to update stats:", error);
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
    // Initialize both input and output canvas
    const inputCanvas = document.getElementById("inputVisualizer");
    const outputCanvas = document.getElementById("outputVisualizer");
    const inputCtx = inputCanvas ? inputCanvas.getContext("2d") : null;
    const outputCtx = outputCanvas ? outputCanvas.getContext("2d") : null;

    // Store canvas references
    this.inputCanvas = inputCanvas;
    this.outputCanvas = outputCanvas;
    this.inputCtx = inputCtx;
    this.outputCtx = outputCtx;

    // Initialize audio context and analysis
    this.audioContext = null;
    this.analyser = null;
    this.microphone = null;
    this.isRecordingActive = false;

    // Enhanced audio analysis buffers
    this.frequencyData = new Uint8Array(2048);
    this.timeData = new Uint8Array(2048);
    this.outputFrequencyData = new Uint8Array(2048);
    this.signalLevel = 0;
    this.outputSignalLevel = 0;

    // Set canvas size to fill container for both canvas
    const resizeCanvas = () => {
      if (inputCanvas) {
        const container = inputCanvas.parentElement;
        inputCanvas.width = container.clientWidth;
        inputCanvas.height = container.clientHeight;
        console.log(
          `Input Canvas resized: ${inputCanvas.width}x${inputCanvas.height}`
        );
      }
      if (outputCanvas) {
        const container = outputCanvas.parentElement;
        outputCanvas.width = container.clientWidth;
        outputCanvas.height = container.clientHeight;
        console.log(
          `Output Canvas resized: ${outputCanvas.width}x${outputCanvas.height}`
        );
      }
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Drawing function for dual waveform visualization
    const drawWaveform = () => {
      this.drawInputWaveform();
      this.drawOutputWaveform();
      this.drawSignalBars();
      requestAnimationFrame(drawWaveform);
    };

    drawWaveform();
  }

  drawInputWaveform() {
    if (!this.inputCtx || !this.inputCanvas) return;

    const ctx = this.inputCtx;
    const canvas = this.inputCanvas;

    // Clear canvas with input gradient background (green theme)
    const bgGradient = ctx.createLinearGradient(
      0,
      0,
      canvas.width,
      canvas.height
    );
    bgGradient.addColorStop(0, "#1a2e1a");
    bgGradient.addColorStop(0.5, "#162e21");
    bgGradient.addColorStop(1, "#0f4620");
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (this.isRecordingActive && this.analyser) {
      // Get frequency data for input waveform
      this.analyser.getByteFrequencyData(this.frequencyData);

      // Calculate overall volume level
      let total = 0;
      for (let i = 0; i < this.frequencyData.length; i++) {
        total += this.frequencyData[i];
      }
      const avgVolume = total / this.frequencyData.length;

      // Draw input amplitude bars
      const centerY = canvas.height / 2;
      const barCount = 60;
      const barWidth = canvas.width / barCount;
      const maxBarHeight = canvas.height * 0.4;

      // Input bars (green theme)
      const barGradient = ctx.createLinearGradient(
        0,
        centerY - maxBarHeight,
        0,
        centerY + maxBarHeight
      );
      barGradient.addColorStop(0, "#00ff88");
      barGradient.addColorStop(0.5, "#88ff00");
      barGradient.addColorStop(1, "#00ff88");

      ctx.fillStyle = barGradient;
      ctx.shadowColor = "#00ff88";
      ctx.shadowBlur = 3;

      for (let i = 0; i < barCount; i++) {
        const sampleIndex = Math.floor(
          (i / barCount) * this.frequencyData.length
        );
        const amplitude = this.frequencyData[sampleIndex] || 0;
        const barHeight = (amplitude / 255) * maxBarHeight;
        const naturalVariation = (Math.random() - 0.5) * 0.1 * barHeight;
        const finalHeight = Math.max(2, barHeight + naturalVariation);
        const x = i * barWidth;

        ctx.fillRect(x, centerY - finalHeight / 2, barWidth - 1, finalHeight);
      }
      ctx.shadowBlur = 0;

      // Draw center line
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(canvas.width, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Input label
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px Arial";
      ctx.fillText("INPUT", 10, 15);
    } else {
      // Static display for input
      const centerY = canvas.height / 2;
      const barCount = 60;
      const barWidth = canvas.width / barCount;

      ctx.fillStyle = "rgba(100, 150, 200, 0.2)";
      for (let i = 0; i < barCount; i++) {
        const x = i * barWidth;
        const variation = Math.random() * 2 - 1;
        const barHeight = 3 + Math.abs(variation);
        ctx.fillRect(x, centerY - barHeight / 2, barWidth - 1, barHeight);
      }
    }
  }

  drawOutputWaveform() {
    if (!this.outputCtx || !this.outputCanvas) return;

    const ctx = this.outputCtx;
    const canvas = this.outputCanvas;

    // Clear canvas with output gradient background (orange/red theme)
    const bgGradient = ctx.createLinearGradient(
      0,
      0,
      canvas.width,
      canvas.height
    );
    bgGradient.addColorStop(0, "#2e1a1a");
    bgGradient.addColorStop(0.5, "#2e1621");
    bgGradient.addColorStop(1, "#460f20");
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (this.isRecordingActive && this.analyser) {
      // For output, apply some processing simulation
      // (In real implementation, this would come from processed audio)

      const centerY = canvas.height / 2;
      const barCount = 60;
      const barWidth = canvas.width / barCount;
      const maxBarHeight = canvas.height * 0.4;

      // Output bars (orange/red theme)
      const barGradient = ctx.createLinearGradient(
        0,
        centerY - maxBarHeight,
        0,
        centerY + maxBarHeight
      );
      barGradient.addColorStop(0, "#ff8800");
      barGradient.addColorStop(0.5, "#ffaa00");
      barGradient.addColorStop(1, "#ff8800");

      ctx.fillStyle = barGradient;
      ctx.shadowColor = "#ff8800";
      ctx.shadowBlur = 3;

      // Simulate processed output (modified version of input)
      for (let i = 0; i < barCount; i++) {
        const sampleIndex = Math.floor(
          (i / barCount) * this.frequencyData.length
        );
        let amplitude = this.frequencyData[sampleIndex] || 0;

        // Apply simulated processing effects
        const selectedDSP =
          document.querySelector('input[name="dspAlgorithm"]:checked')?.value ||
          "bypass";

        switch (selectedDSP) {
          case "fir":
            amplitude *= 0.8; // Slight attenuation
            break;
          case "iir":
            amplitude *= 1.2; // Slight boost
            break;
          case "fft":
            amplitude *= 0.8 + Math.sin(i * 0.1) * 0.3; // Frequency-dependent
            break;
          default: // bypass
            break;
        }

        const barHeight = (amplitude / 255) * maxBarHeight;
        const naturalVariation = (Math.random() - 0.5) * 0.1 * barHeight;
        const finalHeight = Math.max(2, barHeight + naturalVariation);
        const x = i * barWidth;

        ctx.fillRect(x, centerY - finalHeight / 2, barWidth - 1, finalHeight);
      }
      ctx.shadowBlur = 0;

      // Draw center line
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(canvas.width, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Output label
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px Arial";
      ctx.fillText("OUTPUT", 10, 15);
    } else {
      // Static display for output
      const centerY = canvas.height / 2;
      const barCount = 60;
      const barWidth = canvas.width / barCount;

      ctx.fillStyle = "rgba(200, 150, 100, 0.2)";
      for (let i = 0; i < barCount; i++) {
        const x = i * barWidth;
        const variation = Math.random() * 2 - 1;
        const barHeight = 3 + Math.abs(variation);
        ctx.fillRect(x, centerY - barHeight / 2, barWidth - 1, barHeight);
      }
    }
  }

  // Analysis Module
  setupAnalysis() {
    document.getElementById("runAnalysis").addEventListener("click", () => {
      this.runAnalysis();
    });

    document.getElementById("exportResults").addEventListener("click", () => {
      this.exportResults();
    });
  }

  async runAnalysis() {
    if (!this.currentFile) {
      this.showError("Please upload an audio file first");
      return;
    }

    try {
      this.showProcessingStatus("Running comprehensive audio analysis...");

      // Get selected analysis options
      const options = {
        waveform: document.getElementById("analyzeWaveform").checked,
        spectrogram: document.getElementById("analyzeSpectrogram").checked,
        frequency: document.getElementById("analyzeFrequency").checked,
        mfcc: document.getElementById("analyzeMFCC").checked,
        chroma: document.getElementById("analyzeChroma").checked,
        tempo: document.getElementById("analyzeTempo").checked,
      };

      const response = await fetch("/api/analysis/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ options }),
      });

      const result = await response.json();

      if (response.ok && result.success) {
        // Display analysis results
        this.displayAnalysisResults(result.results);
        this.showSuccess("Audio analysis completed successfully!");
      } else {
        this.showError(result.error || "Analysis failed");
      }
    } catch (error) {
      console.error("Analysis error:", error);
      this.showError("Analysis failed: " + error.message);
    } finally {
      this.hideProcessingStatus();
    }
  }

  displayAnalysisResults(results) {
    // Show results container
    document.getElementById("analysisResults").style.display = "block";

    // Display file information
    this.displayFileInfo(results.file_info);

    // Display analysis plots
    this.displayAnalysisPlots(results.analysis_plots);

    // Display insights
    this.displayInsights(results.insights);

    // Display features summary
    this.displayFeaturesSummary(results.features);

    // Show processing summary
    document.getElementById("processingSummary").style.display = "block";
  }

  displayFileInfo(fileInfo) {
    const infoHtml = `
            <div class="alert alert-info">
                <h6><i class="fas fa-info-circle"></i> File Information</h6>
                <div class="row">
                    <div class="col-md-3">
                        <strong>Duration:</strong> ${(
                          fileInfo.duration || 0
                        ).toFixed(2)}s
                    </div>
                    <div class="col-md-3">
                        <strong>Sample Rate:</strong> ${
                          fileInfo.sample_rate || "N/A"
                        } Hz
                    </div>
                    <div class="col-md-3">
                        <strong>Samples:</strong> ${(
                          fileInfo.total_samples || 0
                        ).toLocaleString()}
                    </div>
                    <div class="col-md-3">
                        <strong>Channels:</strong> ${fileInfo.channels || 1}
                    </div>
                </div>
            </div>
        `;

    // Insert before analysis results
    const resultsDiv = document.getElementById("analysisResults");
    resultsDiv.insertAdjacentHTML("afterbegin", infoHtml);
  }

  displayAnalysisPlots(plots) {
    // Create plot containers
    const plotsHtml = `
            <div class="row mt-4">
                ${
                  plots.waveform
                    ? `
                <div class="col-md-6 mb-3">
                    <div class="analysis-plot">
                        <h6>🎵 Waveform Analysis</h6>
                        <img src="data:image/png;base64,${plots.waveform}" class="img-fluid" alt="Waveform Analysis">
                    </div>
                </div>`
                    : ""
                }
                
                ${
                  plots.spectrogram
                    ? `
                <div class="col-md-6 mb-3">
                    <div class="analysis-plot">
                        <h6>🎼 Spectrogram Analysis</h6>
                        <img src="data:image/png;base64,${plots.spectrogram}" class="img-fluid" alt="Spectrogram Analysis">
                    </div>
                </div>`
                    : ""
                }
                
                ${
                  plots.frequency
                    ? `
                <div class="col-md-6 mb-3">
                    <div class="analysis-plot">
                        <h6>📊 Frequency Analysis</h6>
                        <img src="data:image/png;base64,${plots.frequency}" class="img-fluid" alt="Frequency Analysis">
                    </div>
                </div>`
                    : ""
                }
                
                ${
                  plots.mfcc
                    ? `
                <div class="col-md-6 mb-3">
                    <div class="analysis-plot">
                        <h6>🎼 MFCC Features</h6>
                        <img src="data:image/png;base64,${plots.mfcc}" class="img-fluid" alt="MFCC Features">
                    </div>
                </div>`
                    : ""
                }
                
                ${
                  plots.chroma
                    ? `
                <div class="col-md-6 mb-3">
                    <div class="analysis-plot">
                        <h6>🎹 Chroma Features</h6>
                        <img src="data:image/png;base64,${plots.chroma}" class="img-fluid" alt="Chroma Features">
                    </div>
                </div>`
                    : ""
                }
                
                ${
                  plots.tempo
                    ? `
                <div class="col-md-6 mb-3">
                    <div class="analysis-plot">
                        <h6>🥁 Tempo & Beat Analysis</h6>
                        <img src="data:image/png;base64,${plots.tempo}" class="img-fluid" alt="Tempo Analysis">
                    </div>
                </div>`
                    : ""
                }
            </div>
        `;

    document
      .getElementById("analysisResults")
      .insertAdjacentHTML("beforeend", plotsHtml);
  }

  displayInsights(insights) {
    if (!insights || insights.length === 0) return;

    const insightsHtml = `
            <div class="alert alert-success mt-4">
                <h6><i class="fas fa-lightbulb"></i> Analysis Insights</h6>
                <ul class="mb-0">
                    ${insights.map((insight) => `<li>${insight}</li>`).join("")}
                </ul>
            </div>
        `;

    document
      .getElementById("analysisResults")
      .insertAdjacentHTML("beforeend", insightsHtml);
  }

  displayFeaturesSummary(features) {
    let summaryItems = [];

    if (features.tempo) {
      summaryItems.push(
        `<strong>Tempo:</strong> ${features.tempo.tempo_bpm.toFixed(1)} BPM`
      );
    }

    if (features.frequency_stats) {
      summaryItems.push(
        `<strong>Spectral Centroid:</strong> ${features.frequency_stats.spectral_centroid_mean.toFixed(
          0
        )} Hz`
      );
    }

    if (features.chroma) {
      summaryItems.push(
        `<strong>Key:</strong> ${features.chroma.dominant_note}`
      );
    }

    if (features.amplitude_stats) {
      summaryItems.push(
        `<strong>Dynamic Range:</strong> ${features.amplitude_stats.dynamic_range.toFixed(
          2
        )}`
      );
    }

    if (summaryItems.length > 0) {
      const summaryHtml = `
                <div class="alert alert-warning mt-4">
                    <h6><i class="fas fa-chart-bar"></i> Features Summary</h6>
                    <div class="row">
                        ${summaryItems
                          .map((item) => `<div class="col-md-3">${item}</div>`)
                          .join("")}
                    </div>
                </div>
            `;

      document
        .getElementById("analysisResults")
        .insertAdjacentHTML("beforeend", summaryHtml);
    }
  }

  // Socket Events
  setupSocketEvents() {
    this.socket.on("connect", () => {
      console.log("✓ Connected to server");
    });

    this.socket.on("realtime_audio", (data) => {
      // Update audio visualizer
      if (this.visualizerData) {
        // Simulate audio data visualization
        this.visualizerData = this.visualizerData.map(
          () => Math.random() * 100
        );
      }
    });

    this.socket.on("realtime_genre", (data) => {
      // Update current genre display
      document.getElementById("currentGenre").textContent = data.genre || "-";
    });

    this.socket.on("disconnect", () => {
      console.log("⚠️ Disconnected from server");
    });
  }

  // Utility Methods
  showProcessingStatus(message) {
    const statusDiv = document.getElementById("processingStatus");
    statusDiv.querySelector("span").textContent = message;
    statusDiv.style.display = "block";
    this.isProcessing = true;
  }

  hideProcessingStatus() {
    document.getElementById("processingStatus").style.display = "none";
    this.isProcessing = false;
  }

  showSuccess(message) {
    this.showAlert(message, "success");
  }

  showError(message) {
    this.showAlert(message, "danger");
  }

  showWarning(message) {
    this.showAlert(message, "warning");
  }

  showInfo(message) {
    this.showAlert(message, "info");
  }

  showAlert(message, type = "info") {
    // Remove existing alerts
    document
      .querySelectorAll(".alert-custom")
      .forEach((alert) => alert.remove());

    const alertDiv = document.createElement("div");
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
    this.showSuccess("Latency test feature coming soon!");
  }

  compareNoiseReduction() {
    this.showSuccess("Noise comparison feature coming soon!");
  }

  // Real-time Recording Functions
  async startRealtimeRecording() {
    try {
      // Đảm bảo real-time processing đang chạy (required for actual recording)
      if (!this.isRealtimeActive) {
        this.showError("Please start Real-time Processing first!");
        return;
      }

      // Stop microphone test if it's running
      const testBtn = document.getElementById("testMicrophone");
      if (testBtn.textContent.includes("Stop Test")) {
        await this.toggleMicrophoneTest();
      }

      const duration = document.getElementById("recordDuration").value;
      const filename = `realtime_record_${Date.now()}.wav`;

      this.showInfo("🎙️ Starting real-time recording...");

      // Initialize Web Audio API for live visualization FIRST
      await this.initMicrophoneCapture();

      // Set recording active for visualization
      this.isRecordingActive = true;

      const response = await fetch("/api/realtime/start_recording", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: filename,
          duration: duration ? parseFloat(duration) : null,
        }),
      });

      const result = await response.json();

      if (result.success) {
        this.showSuccess(`✓ Recording started: ${filename}`);

        // Update UI
        document.getElementById("startRecording").disabled = true;
        document.getElementById("stopRecording").disabled = false;
        document.getElementById("testMicrophone").disabled = true;

        // Auto-stop after duration if specified
        if (duration) {
          setTimeout(() => {
            this.stopRealtimeRecording();
          }, parseFloat(duration) * 1000);
        }
      } else {
        throw new Error(result.error || "Recording failed");
      }
    } catch (error) {
      this.showError("Recording failed: " + error.message);
      this.stopMicrophoneCapture();
    }
  }

  async initMicrophoneCapture() {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false, // Turn off to get raw audio
          noiseSuppression: false, // Turn off to get full dynamic range
          autoGainControl: false, // Turn off auto gain
          sampleRate: 44100,
        },
      });

      // Create audio context
      this.audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      this.analyser = this.audioContext.createAnalyser();

      // Configure analyser for better sensitivity
      this.analyser.fftSize = 4096; // Higher resolution
      this.analyser.smoothingTimeConstant = 0.3; // Less smoothing for more responsive
      this.analyser.minDecibels = -90; // Lower threshold
      this.analyser.maxDecibels = -10; // Higher threshold

      // Connect microphone to analyser
      this.microphone = this.audioContext.createMediaStreamSource(stream);
      this.microphone.connect(this.analyser);

      // Store stream reference for cleanup
      this.microphoneStream = stream;

      // Initialize frequency data arrays
      this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
      this.timeData = new Uint8Array(this.analyser.fftSize);

      console.log("✓ Microphone capture initialized with high sensitivity");
    } catch (error) {
      console.error("Microphone access failed:", error);
      throw new Error("Could not access microphone: " + error.message);
    }
  }

  stopMicrophoneCapture() {
    try {
      // Stop microphone stream
      if (this.microphoneStream) {
        this.microphoneStream.getTracks().forEach((track) => track.stop());
        this.microphoneStream = null;
      }

      // Disconnect audio nodes
      if (this.microphone) {
        this.microphone.disconnect();
        this.microphone = null;
      }

      // Close audio context
      if (this.audioContext && this.audioContext.state !== "closed") {
        this.audioContext.close();
        this.audioContext = null;
      }

      this.analyser = null;
      this.isRecordingActive = false;

      console.log("✓ Microphone capture stopped");
    } catch (error) {
      console.error("Error stopping microphone:", error);
    }
  }

  async toggleMicrophoneTest() {
    const testBtn = document.getElementById("testMicrophone");

    if (this.isRecordingActive) {
      // Stop microphone test
      this.stopMicrophoneCapture();
      testBtn.innerHTML = '<i class="fas fa-microphone"></i> Test Mic';
      testBtn.classList.remove("btn-success");
      testBtn.classList.add("btn-info-custom");
      this.showInfo("Microphone test stopped");
    } else {
      // Start microphone test
      try {
        await this.initMicrophoneCapture();
        this.isRecordingActive = true; // Enable visualization
        testBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Test';
        testBtn.classList.remove("btn-info-custom");
        testBtn.classList.add("btn-success");
        this.showSuccess(
          "✓ Microphone test started - You should see live waveform!"
        );

        // Auto-stop after 30 seconds
        setTimeout(() => {
          if (
            this.isRecordingActive &&
            testBtn.textContent.includes("Stop Test")
          ) {
            this.toggleMicrophoneTest();
          }
        }, 30000);
      } catch (error) {
        this.showError("Microphone test failed: " + error.message);
      }
    }
  }

  async stopRealtimeRecording() {
    try {
      this.showInfo("⏹️ Stopping recording...");

      // Stop microphone capture first
      this.stopMicrophoneCapture();

      const response = await fetch("/api/realtime/stop_recording", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const result = await response.json();

      if (result.success) {
        this.showSuccess(`✓ Recording saved: ${result.filename}`);
        const duration =
          result.duration != null ? Number(result.duration).toFixed(2) : "N/A";
        this.showSuccess(`📊 Duration: ${duration}s`);

        // Update UI
        document.getElementById("startRecording").disabled = false;
        document.getElementById("stopRecording").disabled = true;
        document.getElementById("testMicrophone").disabled = false;

        // Hiển thị thông tin file đã ghi
        this.displayRecordingResult(result);
      } else {
        throw new Error(result.error || "Stop recording failed");
      }
    } catch (error) {
      this.showError("Stop recording failed: " + error.message);

      // Reset UI on error
      document.getElementById("startRecording").disabled = false;
      document.getElementById("stopRecording").disabled = true;
      document.getElementById("testMicrophone").disabled = false;
      this.stopMicrophoneCapture();
    }
  }

  displayRecordingResult(result) {
    // Tạo thông báo thành công với link download
    const alertDiv = document.createElement("div");
    alertDiv.className = "alert alert-success alert-dismissible fade show mt-3";

    const duration =
      result.duration != null ? Number(result.duration).toFixed(2) : "N/A";

    alertDiv.innerHTML = `
      <h6><i class="fas fa-check-circle"></i> Recording Complete!</h6>
      <p class="mb-2">
        <strong>File:</strong> ${result.filename}<br>
        <strong>Duration:</strong> ${duration} seconds<br>
        <strong>Size:</strong> ${result.file_size || "N/A"}
      </p>
      <div class="d-flex gap-2">
        <a href="/api/audio/download/${
          result.filename
        }" class="btn btn-sm btn-primary">
          <i class="fas fa-download"></i> Download
        </a>
        <button class="btn btn-sm btn-info" onclick="app.loadRecordedFileToUpload('${
          result.filename
        }')">
          <i class="fas fa-upload"></i> Load to Upload Tab
        </button>
      </div>
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Thêm vào tab Real-time
    const realtimeTab = document.getElementById("realtime");
    const moduleCard = realtimeTab.querySelector(".module-card");
    moduleCard.appendChild(alertDiv);

    // Auto-remove after 10 seconds
    setTimeout(() => {
      if (alertDiv.parentNode) {
        alertDiv.remove();
      }
    }, 10000);
  }

  loadRecordedFileToUpload(filename) {
    // Switch to upload tab and simulate file loading
    const uploadTab = document.querySelector('[data-bs-target="#upload"]');
    uploadTab.click();

    // Simulate file info
    this.currentFile = { name: filename };
    document.getElementById("fileName").textContent = filename;
    document.getElementById("fileInfo").style.display = "block";

    this.showSuccess(`✓ File ${filename} loaded to Upload tab!`);
  }

  async exportResults() {
    try {
      this.showProcessingStatus("Exporting analysis results...");

      const response = await fetch("/api/analysis/export", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const result = await response.json();

      if (response.ok && result.success) {
        const plotCount = result.plot_files?.length || 0;
        this.showSuccess(
          `Analysis exported successfully! Generated ${plotCount} plots and 1 report file.`
        );
      } else {
        this.showError(result.error || "Export failed");
      }
    } catch (error) {
      console.error("Export error:", error);
      this.showError("Export failed: " + error.message);
    } finally {
      this.hideProcessingStatus();
    }
  }

  drawSignalBars() {
    // Draw signal level bars in the stats section when real-time is active
    if (!this.isRealtimeActive || !this.isRecordingActive) return;

    const inputBar = document.getElementById("inputSignalBar");
    const outputBar = document.getElementById("outputSignalBar");

    if (!inputBar || !outputBar) return;

    // Calculate signal levels from frequency data
    if (this.frequencyData) {
      let inputTotal = 0;
      let outputTotal = 0;

      for (let i = 0; i < this.frequencyData.length; i++) {
        inputTotal += this.frequencyData[i];
        // Simulate output processing
        const processedValue =
          this.frequencyData[i] * this.getProcessingMultiplier();
        outputTotal += processedValue;
      }

      this.signalLevel = inputTotal / this.frequencyData.length / 255;
      this.outputSignalLevel = outputTotal / this.frequencyData.length / 255;

      // Update signal bars
      this.updateSignalBar(inputBar, this.signalLevel, "#00ff88");
      this.updateSignalBar(outputBar, this.outputSignalLevel, "#ff8800");
    }
  }

  updateSignalBar(bar, level, color) {
    const percentage = Math.min(100, Math.max(0, level * 100));
    bar.style.width = percentage + "%";
    bar.style.backgroundColor = color;
    bar.style.boxShadow = `0 0 10px ${color}`;

    // Add clipping indicator
    if (percentage > 85) {
      bar.style.backgroundColor = "#ff4444";
      bar.style.boxShadow = "0 0 15px #ff4444";
    }
  }

  getProcessingMultiplier() {
    // Get current DSP algorithm and apply corresponding multiplier
    const selectedDSP =
      document.querySelector('input[name="dspAlgorithm"]:checked')?.value ||
      "bypass";

    switch (selectedDSP) {
      case "fir":
        return 0.8; // Slight attenuation
      case "iir":
        return 1.2; // Slight boost
      case "fft":
        return 0.9; // Frequency processing
      default:
        return 1.0; // bypass
    }
  }

  async generateEqualizerVisualizations(gains) {
    try {
      console.log("🎨 Generating enhanced 2D equalizer visualizations...");
      console.log("📊 Current gains:", gains);

      const response = await fetch("/api/equalizer/visualize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gains: gains,
          plot_options: {
            include_spectrogram: true // Request spectrogram if needed
          }
        }),
      });

      console.log("🌐 API Response status:", response.status);
      const result = await response.json();
      console.log("📋 API Response data:", result);

      if (result.success && result.plot_paths) {
        console.log("🎯 Plot paths received:", result.plot_paths);
        
        // Display waveform comparison plot
        if (result.plot_paths.waveform_comparison) {
          const waveformImg = document.getElementById("eqWaveformPlot");
          const imagePath = "/" + result.plot_paths.waveform_comparison;
          console.log("📈 Loading waveform plot:", imagePath);
          waveformImg.src = imagePath;
          waveformImg.style.display = "block";
          waveformImg.onload = () => console.log("✅ Waveform plot loaded successfully");
          waveformImg.onerror = () => console.error("❌ Failed to load waveform plot:", imagePath);
        }

        // Display overlay comparison plot
        if (result.plot_paths.overlay_comparison) {
          const overlayImg = document.getElementById("eqOverlayPlot");
          const imagePath = "/" + result.plot_paths.overlay_comparison;
          console.log("📊 Loading overlay plot:", imagePath);
          overlayImg.src = imagePath;
          overlayImg.style.display = "block";
          overlayImg.onload = () => console.log("✅ Overlay plot loaded successfully");
          overlayImg.onerror = () => console.error("❌ Failed to load overlay plot:", imagePath);
        }

        // Display frequency response plot
        if (result.plot_paths.frequency_response) {
          const freqImg = document.getElementById("eqFreqResponsePlot");
          const imagePath = "/" + result.plot_paths.frequency_response;
          console.log("📡 Loading frequency response plot:", imagePath);
          freqImg.src = imagePath;
          freqImg.style.display = "block";
          freqImg.onload = () => console.log("✅ Frequency response plot loaded successfully");
          freqImg.onerror = () => console.error("❌ Failed to load frequency response plot:", imagePath);
        }

        // Display spectrogram if available
        if (result.plot_paths.spectrogram_comparison) {
          const spectroImg = document.getElementById("eqSpectrogramPlot");
          const imagePath = "/" + result.plot_paths.spectrogram_comparison;
          console.log("🌈 Loading spectrogram plot:", imagePath);
          spectroImg.src = imagePath;
          spectroImg.style.display = "block";
          spectroImg.onload = () => console.log("✅ Spectrogram plot loaded successfully");
          spectroImg.onerror = () => console.error("❌ Failed to load spectrogram plot:", imagePath);
        }

        console.log("✅ Enhanced 2D visualizations loaded successfully");
      } else {
        console.warn("⚠️ No visualization data received:", result);
        this.showError("Failed to generate enhanced visualizations: " + (result.error || 'Unknown error'));
      }
    } catch (error) {
      console.error("❌ Error generating visualizations:", error);
      this.showError("Visualization generation failed: " + error.message);
    }
  }
}

// Initialize the application
const app = new AdvancedAudioApp();
