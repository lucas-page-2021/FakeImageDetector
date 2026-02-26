import os
import io
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from PIL import Image
from torch import nn
from torchvision import models, transforms

app = Flask(__name__)

CHECKPOINT_PATH = Path(os.getenv("TRANSFER_CHECKPOINT_PATH", "ml/artifacts/best_resnet18.pt"))

_model = None
_device = torch.device("cpu")


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, 2)
        self.model = base

    def forward(self, x):
        return self.model(x)


def load_model():
    global _model
    if _model is not None:
        return _model

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=_device)
    model = ClassifierHead().to(_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _model = model
    return _model


def reasons_for_confidence(ai_conf: int):
    if ai_conf >= 70:
        return [
            "Transfer model found strong synthetic-generation cues.",
            "Global structure and texture patterns are atypical for camera-captured faces.",
            "Confidence is high under the ResNet-based classifier.",
        ]
    if ai_conf >= 50:
        return [
            "Transfer model found moderate synthetic-like cues.",
            "Some regions look less consistent with natural camera capture.",
            "Prediction is near the decision boundary; verify with additional images.",
        ]
    if ai_conf <= 30:
        return [
            "Transfer model found strong natural-photo cues.",
            "Face structure and texture align with camera-captured patterns.",
            "Confidence is high under the ResNet-based classifier.",
        ]
    return [
        "Transfer model found moderate natural-photo cues.",
        "Most regions align with camera-captured face characteristics.",
        "Prediction is near the decision boundary; verify with additional images.",
    ]


def infer_image(image_bytes: bytes):
    model = load_model()

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = tfm(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    p_fake = float(probs[0])
    ai_confidence = int(round(max(0.0, min(1.0, p_fake)) * 100))

    return {
        "ok": True,
        "aiConfidence": ai_confidence,
        "reasons": reasons_for_confidence(ai_confidence),
        "source": "transfer-resnet18-service",
    }


@app.get("/health")
def health():
    status = {"ok": True, "checkpoint": str(CHECKPOINT_PATH), "checkpoint_exists": CHECKPOINT_PATH.exists()}
    return jsonify(status)


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "Missing image file field 'image'."}), 400

    file = request.files["image"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"ok": False, "error": "Empty image payload."}), 400

    try:
        result = infer_image(image_bytes)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "9000"))
    app.run(host="0.0.0.0", port=port)
