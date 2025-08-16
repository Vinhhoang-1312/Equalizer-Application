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
        if (target === "#real-time") {
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
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
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

    // Preset loading
    document.getElementById("loadPreset").addEventListener("click", () => {
      this.loadEqualizerPreset();
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

  async processEqualizer() {
    if (!this.currentFile) {
      this.showError("Please upload an audio file first");
      return;
    }

    const gains = this.getEqualizerGains();
    const method = document.getElementById("eqMethod").value;
    const preset = document.getElementById("eqPreset").value;

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
        this.showSuccess(
          `Equalizer applied! RMS change: ${result.rms_change_db.toFixed(2)} dB`
        );
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
      sub_bass: parseFloat(document.getElementById("subBass").value),
      bass: parseFloat(document.getElementById("bass").value),
      low_mid: parseFloat(document.getElementById("lowMid").value),
      mid: parseFloat(document.getElementById("mid").value),
      high_mid: parseFloat(document.getElementById("highMid").value),
      presence: parseFloat(document.getElementById("presence").value),
      brilliance: parseFloat(document.getElementById("brilliance").value),
      air: parseFloat(document.getElementById("air").value),
      ultra_high: parseFloat(document.getElementById("ultraHigh").value),
      extreme: parseFloat(document.getElementById("extreme").value),
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

    document.getElementById("eqPreset").value = "";
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

    // Hiển thị 2 file audio để người dùng có thể nghe so sánh
    this.displayAudioComparisonPlayer(audioFiles);

    // Hiển thị metrics so sánh
    this.displayComparisonMetrics(comparisonAnalysis);

    // Hiển thị giải thích kỹ thuật chi tiết
    this.displayTechnicalExplanation(comparisonAnalysis.technical_explanation);

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

    const original = analysis.original_metrics;
    const processed = analysis.processed_metrics;
    const comparison = analysis.comparison_metrics;

    container.innerHTML = `
      <div class="row">
        <div class="col-md-4">
          <div class="card border-danger">
            <div class="card-header bg-danger text-white">
              <h6 class="mb-0">📊 Audio Gốc</h6>
            </div>
            <div class="card-body">
              <div class="metric-item">
                <strong>SNR:</strong> ${original.snr_estimate.toFixed(1)} dB
              </div>
              <div class="metric-item">
                <strong>RMS Level:</strong> ${original.rms_level.toFixed(4)}
              </div>
              <div class="metric-item">
                <strong>Dynamic Range:</strong> ${original.dynamic_range.toFixed(
                  1
                )} dB
              </div>
              <div class="metric-item">
                <strong>Noise Floor:</strong> ${original.noise_floor.toFixed(4)}
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
                <strong>SNR:</strong> ${processed.snr_estimate.toFixed(1)} dB
              </div>
              <div class="metric-item">
                <strong>RMS Level:</strong> ${processed.rms_level.toFixed(4)}
              </div>
              <div class="metric-item">
                <strong>Dynamic Range:</strong> ${processed.dynamic_range.toFixed(
                  1
                )} dB
              </div>
              <div class="metric-item">
                <strong>Noise Floor:</strong> ${processed.noise_floor.toFixed(
                  4
                )}
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
                comparison.snr_improvement_db > 0
                  ? "text-success"
                  : "text-warning"
              }">
                <strong>SNR Cải Thiện:</strong> ${
                  comparison.snr_improvement_db > 0 ? "+" : ""
                }${comparison.snr_improvement_db.toFixed(1)} dB
              </div>
              <div class="metric-item ${
                comparison.rms_reduction_percent > 0
                  ? "text-success"
                  : "text-warning"
              }">
                <strong>RMS Giảm:</strong> ${comparison.rms_reduction_percent.toFixed(
                  1
                )}%
              </div>
              <div class="metric-item">
                <strong>Noise Floor Giảm:</strong> ${comparison.noise_floor_reduction.toFixed(
                  4
                )}
              </div>
              <div class="metric-item">
                <strong>Dynamic Range:</strong> ${
                  comparison.dynamic_range_change > 0 ? "+" : ""
                }${comparison.dynamic_range_change.toFixed(1)} dB
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

    container.innerHTML = `
      <div class="card">
        <div class="card-header">
          <h6 class="mb-0">📈 Biểu Đồ Phân Tích Chi Tiết</h6>
        </div>
        <div class="card-body">
          <img src="/static/results/${chartPath}?${new Date().getTime()}" 
               class="img-fluid w-100" 
               alt="Noise Reduction Detailed Analysis"
               style="max-height: 800px; object-fit: contain;">
          <div class="mt-3">
            <small class="text-muted">
              <strong>Giải thích biểu đồ:</strong><br>
              • <strong>Waveform:</strong> So sánh dạng sóng âm thanh trước và sau xử lý<br>
              • <strong>Spectrogram:</strong> Phân tích tần số theo thời gian (màu càng sáng = cường độ càng cao)<br>
              • <strong>Metrics:</strong> So sánh các chỉ số kỹ thuật<br>
              • <strong>Processing Details:</strong> Thông tin chi tiết về quá trình xử lý và sample được lấy từ đâu
            </small>
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
    const analysis = result.original_analysis;

    // Populate simplified noise results
    document.getElementById("snrEstimate").textContent =
      analysis.snr_estimate?.toFixed(1) || "N/A";
    document.getElementById("dynamicRange").textContent =
      analysis.dynamic_range?.toFixed(1) || "N/A";

    // Show SNR improvement (key metric for đề bài)
    const snrImprovementElement = document.getElementById("snrImprovement");
    if (snrImprovementElement && result.snr_improvement) {
      snrImprovementElement.textContent =
        "+" + result.snr_improvement.toFixed(1);
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
        this.displayNoiseResults(result);

        const methodName =
          method === "ml" ? "AI/ML Hệ Thống" : "Thư Viện Tốt Nhất";
        this.showSuccess(
          `Giảm nhiễu thành công! SNR cải thiện: ${result.snr_improvement.toFixed(
            2
          )} dB - ${methodName}`
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
    // Device management
    document.getElementById("refreshDevices").addEventListener("click", () => {
      this.loadAudioDevices();
    });

    document.getElementById("testLatency").addEventListener("click", () => {
      this.testLatency();
    });

    // Real-time control
    document.getElementById("startRealtime").addEventListener("click", () => {
      this.startRealtimeProcessing();
    });

    document.getElementById("stopRealtime").addEventListener("click", () => {
      this.stopRealtimeProcessing();
    });

    // Recording controls trong tab Real-time
    document.getElementById("startRecording").addEventListener("click", () => {
      this.startRealtimeRecording();
    });

    document.getElementById("stopRecording").addEventListener("click", () => {
      this.stopRealtimeRecording();
    });

    // Initialize audio visualizer
    this.initAudioVisualizer();
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
    const equalizerParams = this.getEqualizerGains();
    const noiseMethod = document.getElementById("noiseMethod").value;
    const noiseLevel = parseFloat(
      document.getElementById("reductionLevel").value
    );
    const enabledModules = {
      equalizer: document.getElementById("enableEqualizer").checked,
      noise_reduction: document.getElementById("enableNoiseReduction").checked,
      genre_classification: document.getElementById("enableGenreClassification")
        .checked,
    };

    try {
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

      const result = await response.json();

      if (result.success) {
        this.isRealtimeActive = true;
        document.getElementById("startRealtime").disabled = true;
        document.getElementById("stopRealtime").disabled = false;
        document.getElementById("realtimeStats").style.display = "block";

        this.showSuccess("Real-time processing started!");
        this.startStatsUpdater();
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
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
    const canvas = document.getElementById("audioVisualizer");
    const ctx = canvas.getContext("2d");

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Simple audio visualization
    this.visualizerData = new Array(128).fill(0);

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const barWidth = canvas.width / this.visualizerData.length;
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, "rgb(102, 126, 234)");
      gradient.addColorStop(1, "rgb(118, 75, 162)");

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
      // Đảm bảo real-time processing đang chạy
      if (!this.isRealtimeActive) {
        this.showError("Please start Real-time Processing first!");
        return;
      }

      const duration = document.getElementById("recordDuration").value;
      const filename = `realtime_record_${Date.now()}.wav`;

      this.showInfo("🎙️ Starting real-time recording...");

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
    }
  }

  async stopRealtimeRecording() {
    try {
      this.showInfo("⏹️ Stopping recording...");

      const response = await fetch("/api/realtime/stop_recording", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const result = await response.json();

      if (result.success) {
        this.showSuccess(`✓ Recording saved: ${result.filename}`);
        this.showSuccess(`📊 Duration: ${result.duration.toFixed(2)}s`);

        // Update UI
        document.getElementById("startRecording").disabled = false;
        document.getElementById("stopRecording").disabled = true;

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
    }
  }

  displayRecordingResult(result) {
    // Tạo thông báo thành công với link download
    const alertDiv = document.createElement("div");
    alertDiv.className = "alert alert-success alert-dismissible fade show mt-3";
    alertDiv.innerHTML = `
      <h6><i class="fas fa-check-circle"></i> Recording Complete!</h6>
      <p class="mb-2">
        <strong>File:</strong> ${result.filename}<br>
        <strong>Duration:</strong> ${result.duration.toFixed(2)} seconds<br>
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
        this.showSuccess(
          `Analysis exported successfully! Generated ${result.plot_files.length} plots and 1 report file.`
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
}

// Initialize the application
const app = new AdvancedAudioApp();
