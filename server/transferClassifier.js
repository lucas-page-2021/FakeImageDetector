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
      visualReasoning: output.visualReasoning && typeof output.visualReasoning === "object" ? output.visualReasoning : null,
      visualReasoningVersion: typeof output.visualReasoningVersion === "string" ? output.visualReasoningVersion : null,
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

export async function getTransferServiceHealth() {
  const available = await checkTransferAvailability();
  if (!available) {
    return {
      ok: false,
      enabled: enableTransferModel,
      available,
      transferServiceUrl,
      message: "Transfer service is disabled or not configured.",
    };
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);
    const response = await fetch(`${transferServiceUrl.replace(/\/$/, "")}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      return {
        ok: false,
        enabled: enableTransferModel,
        available,
        transferServiceUrl,
        message: `Transfer health check failed (HTTP ${response.status}).`,
      };
    }

    let payload = {};
    try {
      payload = await response.json();
    } catch {
      payload = {};
    }

    return {
      ok: true,
      enabled: enableTransferModel,
      available,
      transferServiceUrl,
      health: payload,
    };
  } catch (error) {
    return {
      ok: false,
      enabled: enableTransferModel,
      available,
      transferServiceUrl,
      message: error?.name === "AbortError" ? "Transfer health check timed out." : "Transfer health check failed.",
    };
  }
}
