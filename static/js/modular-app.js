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
    this.setupRealtimeProcessing();
    this.setupAnalysis();
    this.setupSocketEvents();

    // Load initial data
    this.loadAudioDevices();
    this.loadEqualizerPresets();
    this.loadModelInfo();

    console.log("✓ Advanced Audio Processing App initialized");
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
      this.showProcessingStatus("Reducing noise...");

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
        this.displayNoiseResults(result);
        this.showSuccess(
          `Noise reduced! SNR improvement: ${result.snr_improvement.toFixed(
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

    // Show audio features analysis
    if (result.audio_features) {
      this.displayAudioFeatures(result.audio_features);
    }

    // Show classification reasoning
    if (result.classification_reasoning) {
      this.displayClassificationReasoning(result.classification_reasoning);
    }

    document.getElementById("genreResults").style.display = "block";

    // NO complex probability tables - just simple result as per đề bài requirements
  }

  displayAudioFeatures(features) {
    const featuresHtml = `
            <div class="alert alert-info mt-3">
                <h6><i class="fas fa-chart-line"></i> Phân Tích Đặc Trưng Âm Thanh</h6>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>🎵 Spectral Centroid:</strong> ${features.spectral_centroid.toFixed(
                          0
                        )} Hz</p>
                        <p><strong>📊 Spectral Rolloff:</strong> ${features.spectral_rolloff.toFixed(
                          0
                        )} Hz</p>
                        <p><strong>🥁 Tempo:</strong> ${features.tempo.toFixed(
                          0
                        )} BPM</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>⚡ Zero Crossing Rate:</strong> ${features.zero_crossing_rate.toFixed(
                          3
                        )}</p>
                        <p><strong>🎼 Harmonic Ratio:</strong> ${features.harmonic_ratio.toFixed(
                          2
                        )}</p>
                    </div>
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

    // Recording control
    document.getElementById("startRecording").addEventListener("click", () => {
      this.startRecording();
    });

    document.getElementById("stopRecording").addEventListener("click", () => {
      this.stopRecording();
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

  startRecording() {
    this.showSuccess("Recording feature coming soon!");
  }

  stopRecording() {
    this.showSuccess("Stop recording feature coming soon!");
  }

  compareNoiseReduction() {
    this.showSuccess("Noise comparison feature coming soon!");
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
