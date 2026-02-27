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
    raw_state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(raw_state, dict):
      raise ValueError("Unsupported checkpoint format: expected state dict.")

    # Support both key styles:
    # - resnet keys: conv1.weight, layer1.0...
    # - wrapped keys: model.conv1.weight, model.layer1.0...
    if any(key.startswith("model.") for key in raw_state.keys()):
      state_dict = {
          (key[6:] if key.startswith("model.") else key): value
          for key, value in raw_state.items()
      }
    else:
      state_dict = raw_state

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model = model.to(_device)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    return _model


def reasons_for_confidence(ai_conf: int):
    if ai_conf >= 70:
        return [
            "Facial texture looks overly smooth or repetitive in multiple regions.",
            "Edges around facial features and hairline show subtle blending artifacts.",
            "Lighting and skin detail consistency looks less typical of a camera photo.",
        ]
    if ai_conf >= 50:
        return [
            "Some facial areas show mild texture and detail inconsistencies.",
            "Transitions around eyes, nose, or mouth look slightly less natural.",
            "Lighting and shading cues are mixed, so this is a borderline case.",
        ]
    if ai_conf <= 30:
        return [
            "Skin texture variation and fine detail look natural across the face.",
            "Feature boundaries (eyes, nose, mouth, hairline) look clean and coherent.",
            "Lighting and shadow transitions are consistent with a real photograph.",
        ]
    return [
        "Most facial regions look natural, with only minor irregularities.",
        "Fine details are generally consistent but not strongly decisive.",
        "Visual cues are mixed, so confidence is moderate for this image.",
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
        probs = torch.softmax(logits, dim=1)[0]

    p_fake = float(probs[0].item())
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
