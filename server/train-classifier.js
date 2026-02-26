import fs from "node:fs/promises";
import path from "node:path";
import { extractFeaturesFromBuffer } from "./classifier.js";

const archiveDir = path.resolve(process.cwd(), "archive");
const datasetRoot = path.join(archiveDir, "rvf10k");
const trainCsvPath = path.join(archiveDir, "train.csv");
const validCsvPath = path.join(archiveDir, "valid.csv");
const outputPath = path.resolve(process.cwd(), "server/model/classifier.json");

const maxTrainPerClass = Number(process.env.MAX_TRAIN_PER_CLASS || 1200);
const maxValidPerClass = Number(process.env.MAX_VALID_PER_CLASS || 300);
const epochs = Number(process.env.EPOCHS || 160);
const learningRate = Number(process.env.LR || 0.08);

function parseCsvRows(csvText) {
  const lines = csvText.trim().split("\n");
  const rows = [];

  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;

    const cols = line.split(",");
    const labelStr = cols[4];
    const relPath = cols[5];
    if (!labelStr || !relPath) continue;

    rows.push({
      label: labelStr === "real" ? 1 : 0,
      relPath,
    });
  }

  return rows;
}

function limitPerClass(rows, limit) {
  const real = [];
  const fake = [];

  for (const row of rows) {
    if (row.label === 1 && real.length < limit) real.push(row);
    if (row.label === 0 && fake.length < limit) fake.push(row);
    if (real.length >= limit && fake.length >= limit) break;
  }

  return [...real, ...fake];
}

async function loadSplit(csvPath, limit) {
  const csv = await fs.readFile(csvPath, "utf8");
  const parsed = parseCsvRows(csv);
  return limitPerClass(parsed, limit);
}

async function buildFeatureMatrix(rows) {
  const features = [];
  const labels = [];

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const imagePath = path.join(datasetRoot, row.relPath);

    try {
      const buffer = await fs.readFile(imagePath);
      const feature = await extractFeaturesFromBuffer(buffer);
      features.push(feature);
      labels.push(row.label);
    } catch {
      // Skip unreadable files.
    }

    if ((i + 1) % 200 === 0) {
      console.log(`Processed ${i + 1}/${rows.length}`);
    }
  }

  return { features, labels };
}

function computeFeatureStats(features) {
  const dims = features[0].length;
  const mean = Array(dims).fill(0);
  const std = Array(dims).fill(0);

  for (const row of features) {
    for (let i = 0; i < dims; i += 1) {
      mean[i] += row[i];
    }
  }

  for (let i = 0; i < dims; i += 1) {
    mean[i] /= features.length;
  }

  for (const row of features) {
    for (let i = 0; i < dims; i += 1) {
      const d = row[i] - mean[i];
      std[i] += d * d;
    }
  }

  for (let i = 0; i < dims; i += 1) {
    std[i] = Math.sqrt(std[i] / features.length);
    if (std[i] < 1e-8) std[i] = 1;
  }

  return { mean, std };
}

function normalizeFeatures(features, mean, std) {
  return features.map((row) => row.map((value, idx) => (value - mean[idx]) / std[idx]));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function predictProb(weights, bias, row) {
  let z = bias;
  for (let i = 0; i < row.length; i += 1) {
    z += weights[i] * row[i];
  }
  return sigmoid(z);
}

function accuracy(weights, bias, x, y) {
  let correct = 0;
  for (let i = 0; i < x.length; i += 1) {
    const p = predictProb(weights, bias, x[i]);
    const pred = p >= 0.5 ? 1 : 0;
    if (pred === y[i]) correct += 1;
  }
  return correct / x.length;
}

function trainLogisticRegression(trainX, trainY, validX, validY) {
  const featureCount = trainX[0].length;
  const weights = Array(featureCount).fill(0);
  let bias = 0;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const gradW = Array(featureCount).fill(0);
    let gradB = 0;

    for (let i = 0; i < trainX.length; i += 1) {
      const p = predictProb(weights, bias, trainX[i]);
      const error = p - trainY[i];
      for (let j = 0; j < featureCount; j += 1) {
        gradW[j] += error * trainX[i][j];
      }
      gradB += error;
    }

    const invN = 1 / trainX.length;
    for (let j = 0; j < featureCount; j += 1) {
      weights[j] -= learningRate * gradW[j] * invN;
    }
    bias -= learningRate * gradB * invN;

    if ((epoch + 1) % 20 === 0 || epoch === 0 || epoch === epochs - 1) {
      const trainAcc = accuracy(weights, bias, trainX, trainY);
      const validAcc = accuracy(weights, bias, validX, validY);
      console.log(`Epoch ${epoch + 1}/${epochs} train_acc=${trainAcc.toFixed(4)} valid_acc=${validAcc.toFixed(4)}`);
    }
  }

  return { weights, bias, validAcc: accuracy(weights, bias, validX, validY) };
}

async function main() {
  console.log("Loading CSV splits...");
  const trainRows = await loadSplit(trainCsvPath, maxTrainPerClass);
  const validRows = await loadSplit(validCsvPath, maxValidPerClass);

  console.log(`Train rows selected: ${trainRows.length}`);
  console.log(`Valid rows selected: ${validRows.length}`);

  console.log("Extracting train features...");
  const trainData = await buildFeatureMatrix(trainRows);
  console.log(`Train feature rows extracted: ${trainData.features.length}`);

  console.log("Extracting valid features...");
  const validData = await buildFeatureMatrix(validRows);
  console.log(`Valid feature rows extracted: ${validData.features.length}`);

  if (!trainData.features.length || !validData.features.length) {
    throw new Error("Not enough features extracted for training.");
  }

  const stats = computeFeatureStats(trainData.features);
  const trainX = normalizeFeatures(trainData.features, stats.mean, stats.std);
  const validX = normalizeFeatures(validData.features, stats.mean, stats.std);

  const trained = trainLogisticRegression(trainX, trainData.labels, validX, validData.labels);

  const artifact = {
    version: 1,
    trainedAt: new Date().toISOString(),
    dataset: "archive/rvf10k",
    featureNames: [
      "rMean",
      "gMean",
      "bMean",
      "rStd",
      "gStd",
      "bStd",
      "grayMean",
      "grayStd",
      "edgeMean",
      "edgeStd",
    ],
    featureMean: stats.mean,
    featureStd: stats.std,
    weights: trained.weights,
    bias: trained.bias,
    metrics: {
      validAccuracy: trained.validAcc,
      trainCount: trainX.length,
      validCount: validX.length,
      epochs,
      learningRate,
    },
  };

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(artifact, null, 2));

  console.log(`Model written to ${outputPath}`);
  console.log(`Validation accuracy: ${trained.validAcc.toFixed(4)}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
