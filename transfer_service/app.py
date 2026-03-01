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
_RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
SERVICE_VERSION = os.getenv("TRANSFER_SERVICE_VERSION", "v2-image-cues")


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, 2)
        self.model = base

    def forward(self, x):
        return self.model(x)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _region_map(gray: torch.Tensor):
    h, w = gray.shape
    y0 = int(h * 0.12)
    y1 = int(h * 0.38)
    y2 = int(h * 0.66)
    x0 = int(w * 0.28)
    x1 = int(w * 0.72)
    return {
        "forehead/upper-face": gray[y0:y1, x0:x1],
        "eyes/nose area": gray[y1:y2, x0:x1],
        "mouth/chin area": gray[y2:int(h * 0.9), x0:x1],
    }


def _texture_map(gray: torch.Tensor) -> torch.Tensor:
    pad = torch.nn.functional.pad(gray.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    unfolded = torch.nn.functional.unfold(pad, kernel_size=(3, 3))
    neighborhoods = unfolded.transpose(1, 2).reshape(gray.numel(), 9)
    local_std = neighborhoods.std(dim=1, unbiased=False)
    return local_std.reshape(gray.shape)


def _compute_image_cues(image: Image.Image):
    proc = image.resize((256, 256), _RESAMPLE_BICUBIC)
    rgb = transforms.ToTensor()(proc)
    gray = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).contiguous()

    grad_x = torch.abs(gray[:, 1:] - gray[:, :-1]).mean().item()
    grad_y = torch.abs(gray[1:, :] - gray[:-1, :]).mean().item()
    grad_mean = (grad_x + grad_y) * 0.5
    texture_std = float(gray.std(unbiased=False).item())

    h, w = gray.shape
    half = w // 2
    left = gray[:, :half]
    right = torch.flip(gray[:, w - half :], dims=[1])
    symmetry_error = float(torch.mean(torch.abs(left - right)).item())

    center = gray[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    top = gray[: int(h * 0.2), :]
    bottom = gray[int(h * 0.8) :, :]
    left_band = gray[:, : int(w * 0.2)]
    right_band = gray[:, int(w * 0.8) :]
    border = torch.cat(
        [
            top.flatten(),
            bottom.flatten(),
            left_band.flatten(),
            right_band.flatten(),
        ]
    )
    center_border_delta = float(torch.abs(center.mean() - border.mean()).item())

    highlight_ratio = float((gray > 0.95).float().mean().item())
    shadow_ratio = float((gray < 0.05).float().mean().item())
    clipped_tone_ratio = highlight_ratio + shadow_ratio

    texture_local = _texture_map(gray)
    regions = _region_map(gray)
    region_texture = {}
    for name, patch in regions.items():
        if patch.numel() == 0:
            region_texture[name] = 0.0
        else:
            region_texture[name] = float(patch.std(unbiased=False).item())
    smooth_region = min(region_texture, key=region_texture.get)
    detail_region = max(region_texture, key=region_texture.get)

    cues = {
        "smoothness": {
            "strength": _clamp01((0.16 - texture_std) / 0.10),
            "region": smooth_region,
            "detail": "Fine skin/fabric micro-texture appears unusually smooth.",
        },
        "lighting_inconsistency": {
            "strength": _clamp01((center_border_delta - 0.07) / 0.16),
            "region": "face center vs surrounding regions",
            "detail": "Brightness transitions are less consistent across face and background.",
        },
        "symmetry_inconsistency": {
            "strength": _clamp01((symmetry_error - 0.055) / 0.08),
            "region": "left/right facial halves",
            "detail": "Left-right structure consistency is weaker than expected for a natural photo.",
        },
        "tone_clipping": {
            "strength": _clamp01((clipped_tone_ratio - 0.015) / 0.08),
            "region": "highlight and shadow zones",
            "detail": "Extreme highlights/shadows occur more often than expected.",
        },
        "natural_detail": {
            "strength": _clamp01((texture_std - 0.065) / 0.10),
            "region": detail_region,
            "detail": "Micro-detail variation looks naturally distributed.",
        },
        "natural_edges": {
            "strength": _clamp01((grad_mean - 0.045) / 0.08),
            "region": "feature boundaries",
            "detail": "Edge transitions look coherent and camera-like.",
        },
        "lighting_coherence": {
            "strength": _clamp01((0.11 - center_border_delta) / 0.10),
            "region": "face and nearby background",
            "detail": "Lighting balance looks physically coherent.",
        },
        "tone_balance": {
            "strength": _clamp01((0.05 - clipped_tone_ratio) / 0.04),
            "region": "overall tonal range",
            "detail": "Highlight/shadow levels are within a natural photographic range.",
        },
    }

    return cues


def _evidence_item(cue_type: str, cue: dict):
    return {
        "type": cue_type,
        "strength": round(float(cue["strength"]), 3),
        "region": cue["region"],
        "detail": cue["detail"],
    }


def _reasoning_from_cues(ai_confidence: int, cues: dict):
    is_ai = ai_confidence >= 50

    ai_support = [
        _evidence_item("texture", cues["smoothness"]),
        _evidence_item("lighting", cues["lighting_inconsistency"]),
        _evidence_item("geometry", cues["symmetry_inconsistency"]),
        _evidence_item("tone", cues["tone_clipping"]),
    ]
    real_support = [
        _evidence_item("texture", cues["natural_detail"]),
        _evidence_item("edges", cues["natural_edges"]),
        _evidence_item("lighting", cues["lighting_coherence"]),
        _evidence_item("tone", cues["tone_balance"]),
    ]

    supporting = ai_support if is_ai else real_support
    opposing = real_support if is_ai else ai_support

    top_evidence = [e for e in sorted(supporting, key=lambda x: x["strength"], reverse=True) if e["strength"] >= 0.12][:3]
    counter_evidence = [e for e in sorted(opposing, key=lambda x: x["strength"], reverse=True) if e["strength"] >= 0.12][:2]

    if not top_evidence:
        fallback_detail = (
            "The classifier signal leans toward synthetic image patterns."
            if is_ai
            else "The classifier signal leans toward natural camera-image patterns."
        )
        top_evidence = [
            {
                "type": "classifier-signal",
                "strength": 0.12,
                "region": "whole image",
                "detail": fallback_detail,
            }
        ]

    reasons = []
    seen = set()

    for item in top_evidence:
        text = f"{item['detail']} (region: {item['region']})."
        if text not in seen:
            reasons.append(text)
            seen.add(text)

    for item in counter_evidence:
        if len(reasons) >= 3:
            break
        text = (
            f"Counter-signal noted: {item['detail']} "
            f"(region: {item['region']}), but overall evidence remains {'AI-leaning' if is_ai else 'real-leaning'}."
        )
        if text not in seen:
            reasons.append(text)
            seen.add(text)

    fallback_texts = [
        "The decision is based on aggregated image-cue measurements rather than a single visual cue.",
        "Evidence strength is moderate in this image, so interpretation should focus on the strongest regions listed above.",
        "This explanation is cue-based and may be less reliable on heavily compressed, cropped, or stylized inputs.",
    ]
    for text in fallback_texts:
        if len(reasons) >= 3:
            break
        if text not in seen:
            reasons.append(text)
            seen.add(text)

    explainability_confidence = int(
        round(
            100
            * _clamp01(
                sum(item["strength"] for item in top_evidence[:3]) / max(1, len(top_evidence[:3]))
            )
        )
    )

    return {
        "ai_confidence": ai_confidence,
        "top_evidence": top_evidence,
        "counter_evidence": counter_evidence,
        "explainability_confidence": explainability_confidence,
    }, reasons


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
    cues = _compute_image_cues(image)
    x = tfm(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    p_fake = float(probs[0].item())
    ai_confidence = int(round(max(0.0, min(1.0, p_fake)) * 100))
    reasoning_v2, reasons = _reasoning_from_cues(ai_confidence, cues)

    return {
        "ok": True,
        "aiConfidence": ai_confidence,
        "reasons": reasons,
        "reasoningV2": reasoning_v2,
        "reasoningVersion": "v2-image-cues",
        "source": "transfer-resnet18-service",
    }


@app.get("/health")
def health():
    status = {
        "ok": True,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "reasoningVersion": "v2-image-cues",
        "version": SERVICE_VERSION,
    }
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
