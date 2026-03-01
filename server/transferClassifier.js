const enableTransferModel = process.env.ENABLE_TRANSFER_MODEL === "1";
const transferServiceUrl = (process.env.TRANSFER_SERVICE_URL || "").trim();

let checkedAvailability = false;
let transferAvailable = false;

async function checkTransferAvailability() {
  if (checkedAvailability) {
    return transferAvailable;
  }

  checkedAvailability = true;
  if (!enableTransferModel) {
    transferAvailable = false;
    return transferAvailable;
  }
  transferAvailable = Boolean(transferServiceUrl);

  return transferAvailable;
}

export async function classifyWithTransferModel(imageBuffer) {
  const available = await checkTransferAvailability();
  if (!available) {
    return null;
  }

  try {
    const formData = new FormData();
    formData.append("image", new Blob([imageBuffer], { type: "image/jpeg" }), "input.jpg");

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    const response = await fetch(`${transferServiceUrl.replace(/\/$/, "")}/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!response.ok) {
      return null;
    }

    const output = await response.json();
    if (!output.ok || typeof output.aiConfidence !== "number") {
      return null;
    }

    return {
      aiConfidence: output.aiConfidence,
      reasons: Array.isArray(output.reasons) ? output.reasons : [],
      reasoningV2: output.reasoningV2 && typeof output.reasoningV2 === "object" ? output.reasoningV2 : null,
      reasoningVersion: typeof output.reasoningVersion === "string" ? output.reasoningVersion : null,
      source: output.source || "transfer-resnet18",
    };
  } catch {
    return null;
  }
}

export async function getTransferModelStatus() {
  const available = await checkTransferAvailability();
  return {
    enabled: enableTransferModel,
    available,
    transferServiceUrl,
  };
}
