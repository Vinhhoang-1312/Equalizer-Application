/**
 * Real-time Processing Fixes and Enhancements
 * This file contains fixes for the signal bars and enhanced controls
 */

// Add enhanced EQ controls event listeners
document.addEventListener("DOMContentLoaded", function () {
  // EQ Band Controls
  const eqControls = ["eqBass", "eqMid", "eqTreble"];
  eqControls.forEach((id) => {
    const slider = document.getElementById(id);
    const valueSpan = document.getElementById(id + "Value");

    if (slider && valueSpan) {
      slider.addEventListener("input", function () {
        valueSpan.textContent = this.value;
        updateRealtimeEQ();
      });
    }
  });

  // Master Gain Control
  const masterGain = document.getElementById("masterGain");
  const masterGainValue = document.getElementById("masterGainValue");

  if (masterGain && masterGainValue) {
    masterGain.addEventListener("input", function () {
      masterGainValue.textContent = this.value;
      updateRealtimeEQ();
    });
  }

  // EQ Preset Selection
  const eqPreset = document.getElementById("eqPreset");
  if (eqPreset) {
    eqPreset.addEventListener("change", function () {
      applyEQPreset(this.value);
    });
  }

  // DSP Algorithm Selection
  const dspAlgorithms = document.querySelectorAll('input[name="dspAlgorithm"]');
  dspAlgorithms.forEach((radio) => {
    radio.addEventListener("change", function () {
      if (this.checked) {
        updateDSPAlgorithm(this.value);
      }
    });
  });
});

// Enhanced Signal Bars Update Function
function updateSignalBars(inputLevel, outputLevel) {
  const inputBar = document.getElementById("inputSignalBar");
  const outputBar = document.getElementById("outputSignalBar");

  if (inputBar) {
    const inputPercentage = Math.min(100, Math.max(0, inputLevel * 100));
    inputBar.style.width = inputPercentage + "%";
    inputBar.style.backgroundColor =
      inputPercentage > 85 ? "#ff4444" : "#00ff88";
    inputBar.style.boxShadow = `0 0 10px ${
      inputPercentage > 85 ? "#ff4444" : "#00ff88"
    }`;
  }

  if (outputBar) {
    const outputPercentage = Math.min(100, Math.max(0, outputLevel * 100));
    outputBar.style.width = outputPercentage + "%";
    outputBar.style.backgroundColor =
      outputPercentage > 85 ? "#ff4444" : "#ff8800";
    outputBar.style.boxShadow = `0 0 10px ${
      outputPercentage > 85 ? "#ff4444" : "#ff8800"
    }`;
  }
}

// Apply EQ Presets
function applyEQPreset(preset) {
  const presets = {
    flat: { bass: 0, mid: 0, treble: 0 },
    vocal: { bass: -2, mid: 4, treble: 2 },
    bass: { bass: 6, mid: 0, treble: -2 },
    treble: { bass: -2, mid: 0, treble: 6 },
    rock: { bass: 4, mid: 2, treble: 4 },
    classical: { bass: 0, mid: 2, treble: 3 },
  };

  const preset_values = presets[preset] || presets.flat;

  // Update sliders
  document.getElementById("eqBass").value = preset_values.bass;
  document.getElementById("eqMid").value = preset_values.mid;
  document.getElementById("eqTreble").value = preset_values.treble;

  // Update value displays
  document.getElementById("eqBassValue").textContent = preset_values.bass;
  document.getElementById("eqMidValue").textContent = preset_values.mid;
  document.getElementById("eqTrebleValue").textContent = preset_values.treble;

  // Apply to real-time processing
  updateRealtimeEQ();
}

// Update Real-time EQ
async function updateRealtimeEQ() {
  const eqParams = {
    bass: parseFloat(document.getElementById("eqBass")?.value || 0),
    mid: parseFloat(document.getElementById("eqMid")?.value || 0),
    treble: parseFloat(document.getElementById("eqTreble")?.value || 0),
    master_gain: parseFloat(document.getElementById("masterGain")?.value || 0),
    low_cut: document.getElementById("lowCut")?.checked || false,
    high_cut: document.getElementById("highCut")?.checked || false,
    denoise: document.getElementById("denoise")?.checked || false,
  };

  try {
    const response = await fetch("/api/realtime/update_eq", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(eqParams),
    });

    if (response.ok) {
      console.log("✓ EQ parameters updated:", eqParams);
    }
  } catch (error) {
    console.warn("Failed to update EQ:", error);
  }
}

// Update DSP Algorithm
async function updateDSPAlgorithm(algorithm) {
  try {
    const response = await fetch("/api/realtime/update_dsp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithm: algorithm }),
    });

    if (response.ok) {
      console.log("✓ DSP algorithm updated:", algorithm);
    }
  } catch (error) {
    console.warn("Failed to update DSP algorithm:", error);
  }
}

// Enhanced visualization function for signal bars
function startSignalBarsVisualization() {
  setInterval(() => {
    // Get signal levels from the main app if available
    if (window.app && window.app.signalLevel !== undefined) {
      const inputLevel = window.app.signalLevel || 0;
      const outputLevel = window.app.outputSignalLevel || inputLevel;
      updateSignalBars(inputLevel, outputLevel);
    }
  }, 100); // Update every 100ms for smooth animation
}

// Start visualization when page loads
document.addEventListener("DOMContentLoaded", function () {
  setTimeout(startSignalBarsVisualization, 1000);
});
