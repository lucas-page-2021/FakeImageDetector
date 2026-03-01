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
const visualPanelEl = document.getElementById("visual-panel");
const visualNoteEl = document.getElementById("visual-note");
const visualOverlayEl = document.getElementById("visual-overlay");
const visualMarkersEl = document.getElementById("visual-markers");
const visualEvidenceEl = document.getElementById("visual-evidence");
const verFrontendEl = document.getElementById("ver-frontend");
const verRenderEl = document.getElementById("ver-render");
const verHfEl = document.getElementById("ver-hf");
const verNoteEl = document.getElementById("ver-note");

const allowedExtensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"];
const maxFileSizeBytes = 20 * 1024 * 1024;
const aiThreshold = 50;
const configuredApiBaseUrl = window.APP_CONFIG?.apiBaseUrl?.trim() || "";
const configuredFrontendVersion = window.APP_CONFIG?.frontendVersion?.trim() || "unknown";
const localApiBaseUrl = window.localStorage.getItem("apiBaseUrl") || "";
const apiBaseUrl = configuredApiBaseUrl || localApiBaseUrl;

function setStatus(text, type) {
  statusEl.textContent = text;
  statusEl.className = `status status-${type}`;
}

function clearResult() {
  resultPanel.hidden = true;
  noFacePanel.hidden = true;
  reasonsEl.innerHTML = "";
  barEl.style.width = "0%";
  clearVisualReasoning();
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

async function loadVersions() {
  if (verFrontendEl) {
    verFrontendEl.textContent = configuredFrontendVersion;
  }

  if (!apiBaseUrl) {
    if (verRenderEl) verRenderEl.textContent = "not configured";
    if (verHfEl) verHfEl.textContent = "not configured";
    if (verNoteEl) verNoteEl.textContent = "Set APP_CONFIG.apiBaseUrl to load Render/HF runtime versions.";
    return;
  }

  try {
    const response = await fetch(`${apiBaseUrl}/api/health`);
    const health = await response.json();
    if (!response.ok || !health?.ok) {
      throw new Error("Health endpoint unavailable.");
    }

    if (verRenderEl) verRenderEl.textContent = String(health.apiVersion || "unknown");
    if (verHfEl) verHfEl.textContent = String(health?.transfer?.version || "unknown");

    const transferState = health?.transfer?.ok ? "connected" : "unavailable";
    const transferUrl = health?.transfer?.serviceUrl || "not configured";
    if (verNoteEl) {
      verNoteEl.textContent = `Render↔HF status: ${transferState}. Transfer URL: ${transferUrl}`;
    }
  } catch {
    if (verRenderEl) verRenderEl.textContent = "unreachable";
    if (verHfEl) verHfEl.textContent = "unreachable";
    if (verNoteEl) verNoteEl.textContent = "Failed to load versions from Render API health endpoint.";
  }
}

function clearVisualReasoning() {
  if (visualPanelEl) visualPanelEl.hidden = true;
  if (visualOverlayEl) visualOverlayEl.removeAttribute("src");
  if (visualMarkersEl) visualMarkersEl.innerHTML = "";
  if (visualEvidenceEl) visualEvidenceEl.innerHTML = "";
  if (visualNoteEl) visualNoteEl.textContent = "Model focus overlay and top evidence points.";
}

function toSentenceCase(text) {
  if (!text) return "";
  const trimmed = String(text).trim();
  if (!trimmed) return "";
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
}

function humanizePointType(type) {
  const normalized = String(type || "attention").trim().toLowerCase();
  switch (normalized) {
    case "attention":
      return "Model focus";
    case "texture":
      return "Texture cue";
    case "boundary":
      return "Edge cue";
    case "lighting":
      return "Lighting cue";
    case "tone":
      return "Tone cue";
    case "geometry":
      return "Shape cue";
    case "eyes":
      return "Eye detail cue";
    default:
      return `${toSentenceCase(normalized)} cue`;
  }
}

function humanizeEvidenceLabel(point) {
  const rawLabel = String(point?.label || "").trim();
  const rawRegion = String(point?.region || "").trim();

  if (!rawLabel || /^high[-\s]influence image region\.?$/i.test(rawLabel) || /^evidence point$/i.test(rawLabel)) {
    if (rawRegion) {
      return `Model focused strongly on the ${rawRegion}.`;
    }
    return "Model focused strongly on this area.";
  }

  const cleaned = rawLabel.replace(/\s*\(region:\s*([^)]+)\)\.?$/i, "").trim();
  if (cleaned) {
    return `${toSentenceCase(cleaned).replace(/\.$/, "")}.`;
  }
  if (rawRegion) {
    return `Model focused strongly on the ${rawRegion}.`;
  }
  return "Model focused strongly on this area.";
}

function humanizeReason(reason) {
  const raw = String(reason || "").trim();
  if (!raw) return "";

  if (/^evidence was mixed/i.test(raw)) {
    return "The evidence is mixed, but the strongest image cues still support this result.";
  }

  return raw
    .replace(/^Counter-signal noted:\s*/i, "A competing signal was detected: ")
    .replace(/\bregion:\s*/gi, "area: ")
    .replace(/\s+/g, " ")
    .trim();
}

function uniqueHumanReasons(reasons) {
  const out = [];
  const seen = new Set();
  const input = Array.isArray(reasons) ? reasons : [];
  input.forEach((reason) => {
    const human = humanizeReason(reason);
    if (!human) return;
    const key = human.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push(human);
  });
  return out;
}

function setActiveVisualPoint(pointId) {
  if (!visualMarkersEl || !visualEvidenceEl) {
    return;
  }
  const markers = visualMarkersEl.querySelectorAll(".visual-marker");
  const items = visualEvidenceEl.querySelectorAll(".visual-evidence-item");
  markers.forEach((node) => {
    node.classList.toggle("active", node.dataset.pointId === String(pointId));
  });
  items.forEach((node) => {
    node.classList.toggle("active", node.dataset.pointId === String(pointId));
  });
}

function renderVisualReasoning(visualReasoning, visualReasoningVersion) {
  clearVisualReasoning();
  if (!visualPanelEl) {
    return;
  }
  if (!visualReasoning || typeof visualReasoning !== "object") {
    return;
  }

  const overlaySrc = typeof visualReasoning.overlayImageBase64 === "string" ? visualReasoning.overlayImageBase64 : "";
  const points = Array.isArray(visualReasoning.evidencePoints) ? visualReasoning.evidencePoints : [];
  const width = Number(visualReasoning.imageSize?.width || 0);
  const height = Number(visualReasoning.imageSize?.height || 0);

  if (!overlaySrc || !points.length || !width || !height) {
    return;
  }

  visualPanelEl.hidden = false;
  visualOverlayEl.src = overlaySrc;
  if (visualNoteEl) {
    const version = visualReasoningVersion || visualReasoning.visualVersion || "unknown";
    visualNoteEl.textContent = `Heatmap overlay from ${version}. Higher intensity indicates stronger model influence.`;
  }

  points.forEach((point) => {
    const pointId = String(point.id || "");
    const readableLabel = humanizeEvidenceLabel(point);
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "visual-marker";
    marker.textContent = String(point.id || "?");
    marker.style.left = `${(Number(point.x) / width) * 100}%`;
    marker.style.top = `${(Number(point.y) / height) * 100}%`;
    marker.dataset.pointId = pointId;
    marker.style.pointerEvents = "auto";
    marker.title = readableLabel;
    marker.addEventListener("click", () => {
      setActiveVisualPoint(point.id || "");
    });
    visualMarkersEl.appendChild(marker);

    const li = document.createElement("li");
    const item = document.createElement("div");
    item.className = "visual-evidence-item";
    item.dataset.pointId = pointId;
    const label = document.createElement("div");
    label.textContent = readableLabel;
    const meta = document.createElement("div");
    meta.className = "visual-strength";
    const pointType = humanizePointType(point.type || "attention");
    const strength = Number(point.strength || 0);
    meta.textContent = `${pointType} | Influence: ${Math.round(strength * 100)}%`;
    item.appendChild(label);
    item.appendChild(meta);
    item.addEventListener("click", () => {
      setActiveVisualPoint(point.id || "");
    });
    li.appendChild(item);
    visualEvidenceEl.appendChild(li);
  });

  const firstId = points[0]?.id;
  if (firstId !== undefined) {
    setActiveVisualPoint(firstId);
  }
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
    setStatus("File is too large. Maximum upload size is 20MB.", "error");
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
      setStatus("File is too large. Maximum upload size is 20MB.", "error");
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
  let visualReasoning = null;
  let visualReasoningVersion = null;

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
    visualReasoning = apiResult.visualReasoning || null;
    visualReasoningVersion = apiResult.visualReasoningVersion || null;
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
    visualReasoning = null;
    visualReasoningVersion = null;
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

  const readableReasons = uniqueHumanReasons(reasons);
  readableReasons.forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    reasonsEl.appendChild(li);
  });
  renderVisualReasoning(visualReasoning, visualReasoningVersion);

  setStatus("Analysis complete.", "done");
});

resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  urlInput.value = "";
  clearPreview();
  clearResult();
  setStatus("Waiting for input...", "idle");
});

loadVersions();
