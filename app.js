const fileInput = document.getElementById("image-file");
const urlInput = document.getElementById("image-url");
const preview = document.getElementById("preview");
const previewEmpty = document.getElementById("preview-empty");
const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");
const statusEl = document.getElementById("status");
const resultPanel = document.getElementById("result-panel");
const noFacePanel = document.getElementById("noface-panel");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const barEl = document.getElementById("bar");
const reasonsEl = document.getElementById("reasons");

const allowedExtensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"];
const maxFileSizeBytes = 100 * 1024 * 1024;
const aiThreshold = 50;
const apiBaseUrl = window.localStorage.getItem("apiBaseUrl") || "";

function setStatus(text, type) {
  statusEl.textContent = text;
  statusEl.className = `status status-${type}`;
}

function clearResult() {
  resultPanel.hidden = true;
  noFacePanel.hidden = true;
  reasonsEl.innerHTML = "";
  barEl.style.width = "0%";
}

function showPreview(src) {
  preview.src = src;
  preview.hidden = false;
  previewEmpty.hidden = true;
}

function clearPreview() {
  preview.src = "";
  preview.hidden = true;
  previewEmpty.hidden = false;
}

function extensionFromName(name) {
  const parts = name.toLowerCase().split(".");
  return parts.length > 1 ? parts.pop() : "";
}

function isAllowedFormat(nameOrUrl) {
  const ext = extensionFromName(nameOrUrl);
  return allowedExtensions.includes(ext);
}

function mockAnalyze(inputRef) {
  const ref = inputRef.toLowerCase();

  if (ref.includes("noface") || ref.includes("landscape") || ref.includes("cat")) {
    return {
      faceCount: 0,
    };
  }

  if (ref.includes("group") || ref.includes("crowd") || ref.includes("twofaces") || ref.includes("multiface")) {
    return {
      faceCount: 2,
    };
  }

  if (ref.includes("ai") || ref.includes("generated") || ref.includes("midjourney")) {
    return {
      faceCount: 1,
      aiConfidence: 88,
      reasons: [
        "Skin texture appears unusually uniform in several regions",
        "Fine background details show synthetic blending artifacts",
        "Specular highlights around the eyes appear physically inconsistent",
      ],
    };
  }

  return {
    faceCount: 1,
    aiConfidence: 18,
    reasons: [
      "Skin pores and micro-variations look naturally distributed",
      "Lighting and shadow transitions align with realistic face geometry",
      "No strong synthetic boundary artifacts are visible",
    ],
  };
}

async function analyzeWithApi({ file, url }) {
  if (!apiBaseUrl) {
    return null;
  }

  const endpoint = file ? `${apiBaseUrl}/api/predict/file` : `${apiBaseUrl}/api/predict/url`;
  let response;

  if (file) {
    const formData = new FormData();
    formData.append("image", file);
    response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });
  } else {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ imageUrl: url }),
    });
  }

  let data = {};
  try {
    data = await response.json();
  } catch {
    data = {};
  }

  if (!response.ok) {
    const message = data.message || "Backend request failed.";
    throw new Error(message);
  }

  return data;
}

fileInput.addEventListener("change", () => {
  clearResult();
  const file = fileInput.files?.[0];
  if (!file) {
    clearPreview();
    setStatus("Waiting for input...", "idle");
    return;
  }
  if (!isAllowedFormat(file.name)) {
    clearPreview();
    setStatus("Unsupported file format. Please use JPG, JPEG, PNG, WEBP, BMP, or TIFF.", "error");
    fileInput.value = "";
    return;
  }
  if (file.size > maxFileSizeBytes) {
    clearPreview();
    setStatus("File is too large. Maximum upload size is 100MB.", "error");
    fileInput.value = "";
    return;
  }
  showPreview(URL.createObjectURL(file));
  setStatus("File selected. Ready to analyze.", "idle");
});

urlInput.addEventListener("input", () => {
  clearResult();
  const url = urlInput.value.trim();
  if (!url) {
    clearPreview();
    setStatus("Waiting for input...", "idle");
    return;
  }

  try {
    const parsed = new URL(url);
    if (!isAllowedFormat(parsed.pathname)) {
      setStatus("Unsupported URL format. Use an image link ending in JPG, JPEG, PNG, WEBP, BMP, or TIFF.", "error");
      clearPreview();
      return;
    }
    showPreview(url);
    setStatus("Image URL set. Ready to analyze.", "idle");
  } catch {
    clearPreview();
    setStatus("Please enter a valid image URL.", "error");
  }
});

analyzeBtn.addEventListener("click", async () => {
  clearResult();
  const file = fileInput.files?.[0];
  const url = urlInput.value.trim();

  if (!file && !url) {
    setStatus("Please upload an image or provide an image URL.", "error");
    return;
  }

  let inputRef = "";
  if (file) {
    if (file.size > maxFileSizeBytes) {
      setStatus("File is too large. Maximum upload size is 100MB.", "error");
      return;
    }
    inputRef = file.name;
  } else {
    try {
      const parsed = new URL(url);
      inputRef = parsed.pathname;
    } catch {
      setStatus("Please enter a valid image URL.", "error");
      return;
    }
  }

  setStatus(apiBaseUrl ? "Analyzing image with backend..." : "Analyzing image (mock)...", "working");
  let apiResult = null;
  try {
    apiResult = await analyzeWithApi({ file, url });
  } catch (error) {
    setStatus(error.message || "Request failed.", "error");
    return;
  }

  let label;
  let confidence;
  let reasons;

  if (apiResult) {
    if (!apiResult.ok && apiResult.hasFace === false) {
      noFacePanel.hidden = false;
      setStatus(apiResult.message || "No face detected.", "error");
      return;
    }
    if (!apiResult.ok) {
      setStatus(apiResult.message || "Analysis failed.", "error");
      return;
    }

    label = apiResult.label;
    confidence = apiResult.confidence;
    reasons = apiResult.reasons || [];
  } else {
    await new Promise((resolve) => setTimeout(resolve, 600));
    const result = mockAnalyze(inputRef);

    if (result.faceCount === 0) {
      noFacePanel.hidden = false;
      setStatus("Analysis complete: no face detected.", "error");
      return;
    }
    if (result.faceCount > 1) {
      setStatus("Please upload an image with exactly one human face.", "error");
      return;
    }

    const isAiGenerated = result.aiConfidence >= aiThreshold;
    label = isAiGenerated ? "AI-generated" : "Real";
    confidence = isAiGenerated ? result.aiConfidence : 100 - result.aiConfidence;
    reasons = result.reasons;
  }

  resultPanel.hidden = false;
  labelEl.textContent = label;
  confidenceEl.textContent = `${confidence}%`;
  barEl.style.width = `${confidence}%`;

  if (confidence >= 75) {
    barEl.style.background = "var(--ok)";
  } else if (confidence >= 50) {
    barEl.style.background = "var(--warn)";
  } else {
    barEl.style.background = "var(--bad)";
  }

  reasons.forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    reasonsEl.appendChild(li);
  });

  setStatus("Analysis complete.", "done");
});

resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  urlInput.value = "";
  clearPreview();
  clearResult();
  setStatus("Waiting for input...", "idle");
});
