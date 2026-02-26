import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const modelPath = path.resolve(process.cwd(), "server/model/classifier.json");

let cachedModel = null;

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function safeStd(v) {
  return v > 1e-8 ? v : 1;
}

function sobelFeatures(gray, width, height) {
  let sum = 0;
  let sqSum = 0;
  let count = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const idx = y * width + x;
      const tl = gray[idx - width - 1];
      const tc = gray[idx - width];
      const tr = gray[idx - width + 1];
      const ml = gray[idx - 1];
      const mr = gray[idx + 1];
      const bl = gray[idx + width - 1];
      const bc = gray[idx + width];
      const br = gray[idx + width + 1];

      const gx = -tl - 2 * ml - bl + tr + 2 * mr + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const mag = Math.sqrt(gx * gx + gy * gy);

      sum += mag;
      sqSum += mag * mag;
      count += 1;
    }
  }

  const mean = count ? sum / count : 0;
  const variance = count ? sqSum / count - mean * mean : 0;
  return [mean, Math.sqrt(Math.max(0, variance))];
}

export async function extractFeaturesFromBuffer(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .resize(128, 128, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixelCount = info.width * info.height;
  const gray = new Float32Array(pixelCount);

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let rSq = 0;
  let gSq = 0;
  let bSq = 0;
  let graySum = 0;
  let graySq = 0;

  for (let i = 0, p = 0; i < data.length; i += 3, p += 1) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;

    rSum += r;
    gSum += g;
    bSum += b;
    rSq += r * r;
    gSq += g * g;
    bSq += b * b;

    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[p] = y;
    graySum += y;
    graySq += y * y;
  }

  const rMean = rSum / pixelCount;
  const gMean = gSum / pixelCount;
  const bMean = bSum / pixelCount;
  const rStd = Math.sqrt(Math.max(0, rSq / pixelCount - rMean * rMean));
  const gStd = Math.sqrt(Math.max(0, gSq / pixelCount - gMean * gMean));
  const bStd = Math.sqrt(Math.max(0, bSq / pixelCount - bMean * bMean));
  const grayMean = graySum / pixelCount;
  const grayStd = Math.sqrt(Math.max(0, graySq / pixelCount - grayMean * grayMean));
  const [edgeMean, edgeStd] = sobelFeatures(gray, info.width, info.height);

  return [rMean, gMean, bMean, rStd, gStd, bStd, grayMean, grayStd, edgeMean, edgeStd];
}

export async function loadClassifierModel() {
  if (cachedModel) {
    return cachedModel;
  }

  try {
    const raw = await fs.readFile(modelPath, "utf8");
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.weights) || !Array.isArray(parsed.featureMean) || !Array.isArray(parsed.featureStd)) {
      throw new Error("Invalid model file format");
    }
    cachedModel = parsed;
    return cachedModel;
  } catch {
    return null;
  }
}

export async function classifyImageBuffer(imageBuffer, inputRef = "") {
  const model = await loadClassifierModel();
  if (!model) {
    return null;
  }

  const features = await extractFeaturesFromBuffer(imageBuffer);
  const normalized = features.map((value, idx) => (value - model.featureMean[idx]) / safeStd(model.featureStd[idx]));

  let logit = model.bias;
  for (let i = 0; i < normalized.length; i += 1) {
    logit += normalized[i] * model.weights[i];
  }

  const pReal = sigmoid(logit);
  const pFake = 1 - pReal;
  const aiConfidence = Math.max(0, Math.min(100, Math.round(pFake * 100)));

  const reasons =
    aiConfidence >= 50
      ? [
          "Model detected synthetic-like texture and edge distribution patterns.",
          "Color and contrast statistics diverge from typical real-face training samples.",
          "Prediction is based on learned visual feature patterns from the archive dataset.",
        ]
      : [
          "Model detected natural-like texture and edge distribution patterns.",
          "Color and contrast statistics align with typical real-face training samples.",
          "Prediction is based on learned visual feature patterns from the archive dataset.",
        ];

  return {
    aiConfidence,
    reasons,
    source: "archive-trained-logistic",
    inputRef,
  };
}
