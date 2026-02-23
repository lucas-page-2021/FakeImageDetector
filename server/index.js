import express from "express";
import multer from "multer";
import net from "node:net";

const app = express();
const port = process.env.PORT || 8787;

const allowedExtensions = new Set(["jpg", "jpeg", "png", "webp", "bmp", "tiff"]);
const allowedMimeTypes = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"]);
const maxFileSizeBytes = 100 * 1024 * 1024;
const aiThreshold = 50;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: maxFileSizeBytes,
  },
});

app.use(express.json({ limit: "1mb" }));
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }
  return next();
});

function extensionFromName(name) {
  const parts = String(name).toLowerCase().split(".");
  return parts.length > 1 ? parts.pop() : "";
}

function isAllowedExtension(nameOrUrlPath) {
  return allowedExtensions.has(extensionFromName(nameOrUrlPath));
}

function getMockInference(inputRef) {
  const ref = String(inputRef).toLowerCase();

  if (ref.includes("noface") || ref.includes("landscape") || ref.includes("cat")) {
    return { faceCount: 0 };
  }

  if (ref.includes("group") || ref.includes("crowd") || ref.includes("twofaces") || ref.includes("multiface")) {
    return { faceCount: 2 };
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

function toResponsePayload(mockResult) {
  if (mockResult.faceCount === 0) {
    return {
      ok: false,
      hasFace: false,
      message: "No human face detected. Please reupload an image with a clear human face.",
    };
  }

  if (mockResult.faceCount > 1) {
    return {
      ok: false,
      hasFace: true,
      message: "Please upload an image with exactly one human face.",
    };
  }

  const isAiGenerated = mockResult.aiConfidence >= aiThreshold;
  const label = isAiGenerated ? "AI-generated" : "Real";
  const confidence = isAiGenerated ? mockResult.aiConfidence : 100 - mockResult.aiConfidence;

  return {
    ok: true,
    hasFace: true,
    label,
    confidence,
    reasons: mockResult.reasons,
  };
}

function isPrivateOrLocalIp(ip) {
  if (net.isIP(ip) === 4) {
    if (ip === "127.0.0.1") return true;
    if (ip.startsWith("10.")) return true;
    if (ip.startsWith("192.168.")) return true;
    if (/^172\.(1[6-9]|2\d|3[0-1])\./.test(ip)) return true;
    if (ip.startsWith("169.254.")) return true;
    return false;
  }

  if (net.isIP(ip) === 6) {
    const normalized = ip.toLowerCase();
    if (normalized === "::1") return true;
    if (normalized.startsWith("fc") || normalized.startsWith("fd")) return true;
    if (normalized.startsWith("fe80")) return true;
    return false;
  }

  return false;
}

function validateUrlSafety(imageUrl) {
  let parsed;
  try {
    parsed = new URL(imageUrl);
  } catch {
    return { valid: false, message: "Invalid URL format." };
  }

  if (!["http:", "https:"].includes(parsed.protocol)) {
    return { valid: false, message: "Only HTTP/HTTPS image URLs are allowed." };
  }

  if (!isAllowedExtension(parsed.pathname)) {
    return {
      valid: false,
      message: "Unsupported URL format. Use an image link ending in JPG, JPEG, PNG, WEBP, BMP, or TIFF.",
    };
  }

  const hostname = parsed.hostname.toLowerCase();
  if (hostname === "localhost" || hostname.endsWith(".local")) {
    return { valid: false, message: "Localhost/local network URLs are not allowed." };
  }

  if (isPrivateOrLocalIp(hostname)) {
    return { valid: false, message: "Private or loopback IP URLs are not allowed." };
  }

  return { valid: true, parsed };
}

async function fetchAndValidateRemoteImage(imageUrl) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);

  try {
    const response = await fetch(imageUrl, {
      method: "GET",
      redirect: "follow",
      signal: controller.signal,
    });

    if (!response.ok) {
      return { valid: false, message: `Failed to fetch image URL (HTTP ${response.status}).` };
    }

    const contentType = (response.headers.get("content-type") || "").split(";")[0].trim().toLowerCase();
    if (!allowedMimeTypes.has(contentType)) {
      return {
        valid: false,
        message: "URL does not point to a supported image type (JPG, JPEG, PNG, WEBP, BMP, TIFF).",
      };
    }

    const contentLengthHeader = response.headers.get("content-length");
    if (contentLengthHeader && Number(contentLengthHeader) > maxFileSizeBytes) {
      return { valid: false, message: "Remote image is too large. Maximum size is 100MB." };
    }

    const imageBuffer = Buffer.from(await response.arrayBuffer());
    if (imageBuffer.length > maxFileSizeBytes) {
      return { valid: false, message: "Remote image is too large. Maximum size is 100MB." };
    }

    return { valid: true, bytes: imageBuffer.length };
  } catch {
    return { valid: false, message: "Failed to fetch image URL safely." };
  } finally {
    clearTimeout(timeout);
  }
}

app.get("/api/health", (_, res) => {
  res.json({ ok: true, message: "API is running" });
});

app.post("/api/predict/file", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ ok: false, message: "No image file provided in field 'image'." });
  }

  if (!isAllowedExtension(req.file.originalname)) {
    return res.status(400).json({
      ok: false,
      message: "Unsupported file format. Please use JPG, JPEG, PNG, WEBP, BMP, or TIFF.",
    });
  }

  const mimeType = (req.file.mimetype || "").toLowerCase();
  if (!allowedMimeTypes.has(mimeType)) {
    return res.status(400).json({
      ok: false,
      message: "Unsupported MIME type. Please upload JPG, JPEG, PNG, WEBP, BMP, or TIFF.",
    });
  }

  const payload = toResponsePayload(getMockInference(req.file.originalname));
  return res.json(payload);
});

app.post("/api/predict/url", async (req, res) => {
  const imageUrl = req.body?.imageUrl;
  if (!imageUrl || typeof imageUrl !== "string") {
    return res.status(400).json({ ok: false, message: "imageUrl is required." });
  }

  const safety = validateUrlSafety(imageUrl);
  if (!safety.valid) {
    return res.status(400).json({ ok: false, message: safety.message });
  }

  const remoteValidation = await fetchAndValidateRemoteImage(imageUrl);
  if (!remoteValidation.valid) {
    return res.status(400).json({ ok: false, message: remoteValidation.message });
  }

  const payload = toResponsePayload(getMockInference(safety.parsed.pathname));
  return res.json(payload);
});

app.use((err, _, res, __) => {
  if (err?.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({ ok: false, message: "File is too large. Maximum upload size is 100MB." });
  }

  return res.status(500).json({ ok: false, message: "Unexpected server error." });
});

app.listen(port, () => {
  console.log(`API server listening on http://localhost:${port}`);
});
