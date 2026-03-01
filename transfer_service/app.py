import os
import io
import base64
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms

app = Flask(__name__)

CHECKPOINT_PATH = Path(os.getenv("TRANSFER_CHECKPOINT_PATH", "ml/artifacts/best_resnet18.pt"))

_model = None
_device = torch.device("cpu")
_RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
SERVICE_VERSION = os.getenv("TRANSFER_SERVICE_VERSION", "v2-image-cues")
ENABLE_VISUAL_REASONING = os.getenv("ENABLE_VISUAL_REASONING", "1") == "1"
VISUAL_REASONING_VERSION = os.getenv("VISUAL_REASONING_VERSION", "v1-gradcam")
VISUAL_MAX_SIDE = int(os.getenv("VISUAL_MAX_SIDE", "640"))
VISUAL_MAX_POINTS = int(os.getenv("VISUAL_MAX_POINTS", "3"))
VISUAL_MIN_POINT_STRENGTH = float(os.getenv("VISUAL_MIN_POINT_STRENGTH", "0.20"))
VISUAL_MIN_EVIDENCE_STRENGTH = float(os.getenv("VISUAL_MIN_EVIDENCE_STRENGTH", "0.16"))
VISUAL_MIN_EXPLAINABILITY_CONF = int(os.getenv("VISUAL_MIN_EXPLAINABILITY_CONF", "35"))
VISUAL_SUPPRESS_LOW_CONF = os.getenv("VISUAL_SUPPRESS_LOW_CONF", "1") == "1"


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
        "This decision uses multiple image cues together, not a single pixel or region.",
        "Some cues are moderate-strength, so prioritize the strongest regions listed above.",
        "Explanations may be less reliable on heavily compressed, cropped, or highly stylized images.",
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


def _human_visual_label(item: dict, target_idx: int):
    """Build a plain-language label for a visual evidence point."""
    target_is_ai = target_idx == 0
    if item:
        detail = str(item.get("detail", "")).strip()
        region = str(item.get("region", "")).strip()
        cue_type = str(item.get("type", "focus")).strip().lower()

        if detail:
            clean = detail.rstrip(".")
            if region:
                return f"{clean} (area: {region})."
            return f"{clean}."

        if cue_type == "texture":
            return "Texture patterns in this area strongly influenced the decision."
        if cue_type in ("boundary", "edges"):
            return "Edge transitions in this area strongly influenced the decision."
        if cue_type == "lighting":
            return "Lighting behavior in this area strongly influenced the decision."
        if cue_type == "geometry":
            return "Facial structure in this area strongly influenced the decision."
        if cue_type == "tone":
            return "Highlight and shadow balance in this area strongly influenced the decision."

    if target_is_ai:
        return "This highlighted area contributed strongly to an AI-generated decision."
    return "This highlighted area contributed strongly to a real-image decision."


def _to_data_url_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _resize_for_visual(image: Image.Image):
    w, h = image.size
    longest = max(w, h)
    if longest <= VISUAL_MAX_SIDE:
        return image.copy()
    scale = VISUAL_MAX_SIDE / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), _RESAMPLE_BICUBIC)


def _extract_peak_points(
    cam_map: torch.Tensor,
    max_points: int = 3,
    min_distance: int = 28,
    min_strength: float = 0.15,
):
    if cam_map.numel() == 0:
        return []

    work = cam_map.clone()
    h, w = work.shape
    points = []

    while len(points) < max_points:
        flat_idx = int(torch.argmax(work).item())
        value = float(work.view(-1)[flat_idx].item())
        if value < float(min_strength):
            break

        y = flat_idx // w
        x = flat_idx % w
        points.append({"x": int(x), "y": int(y), "strength": round(value, 3)})

        y0 = max(0, y - min_distance)
        y1 = min(h, y + min_distance + 1)
        x0 = max(0, x - min_distance)
        x1 = min(w, x + min_distance + 1)
        work[y0:y1, x0:x1] = 0.0

    return points


def _compute_gradcam(model: nn.Module, x: torch.Tensor, target_idx: int):
    activations = {}
    gradients = {}

    def _forward_hook(_, __, output):
        activations["value"] = output

    def _backward_hook(_, grad_input, grad_output):
        _ = grad_input
        gradients["value"] = grad_output[0]

    handle_fwd = model.layer4.register_forward_hook(_forward_hook)
    handle_bwd = model.layer4.register_full_backward_hook(_backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        score = logits[:, target_idx].sum()
        score.backward()

        act = activations.get("value")
        grad = gradients.get("value")
        if act is None or grad is None:
            return None

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam_min = float(cam.min().item())
        cam_max = float(cam.max().item())
        if cam_max - cam_min < 1e-8:
            return None
        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.detach()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def _empty_visual_reasoning(
    target_idx: int,
    image_size: dict,
    explainability_confidence: int,
    warning: str,
):
    return {
        "visualVersion": VISUAL_REASONING_VERSION,
        "targetClass": "ai" if target_idx == 0 else "real",
        "imageSize": image_size,
        "evidencePoints": [],
        "overlayImageBase64": "",
        "heatmapImageBase64": "",
        "explainabilityConfidence": int(explainability_confidence),
        "quality": "low",
        "available": False,
        "warnings": [warning] if warning else [],
    }


def _build_visual_reasoning(image: Image.Image, cam_224: torch.Tensor, target_idx: int, reasoning_v2: dict):
    display_img = _resize_for_visual(image)
    disp_w, disp_h = display_img.size
    image_size = {"width": disp_w, "height": disp_h}
    explainability_confidence = int(reasoning_v2.get("explainability_confidence", 0))

    if cam_224 is None:
        return _empty_visual_reasoning(
            target_idx,
            image_size,
            explainability_confidence,
            "Visual explanation is unavailable for this image.",
        )

    cam_disp = F.interpolate(
        cam_224.unsqueeze(0).unsqueeze(0),
        size=(disp_h, disp_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0)

    img_t = transforms.ToTensor()(display_img)
    heat_r = cam_disp
    heat_g = (1.0 - torch.abs(cam_disp - 0.5) * 2.0).clamp(0.0, 1.0)
    heat_b = (1.0 - cam_disp).clamp(0.0, 1.0)
    heat_rgb = torch.stack([heat_r, heat_g, heat_b], dim=0)
    overlay_t = (0.62 * img_t + 0.38 * heat_rgb).clamp(0.0, 1.0)

    overlay_img = transforms.ToPILImage()(overlay_t)
    heatmap_img = transforms.ToPILImage()(heat_rgb)

    peaks = _extract_peak_points(
        cam_disp,
        max_points=max(1, VISUAL_MAX_POINTS),
        min_strength=VISUAL_MIN_POINT_STRENGTH,
    )
    if not peaks:
        return _empty_visual_reasoning(
            target_idx,
            image_size,
            explainability_confidence,
            "Visual explanation was suppressed because model-focus signal is weak.",
        )

    top_evidence_all = reasoning_v2.get("top_evidence") or []
    top_evidence = [item for item in top_evidence_all if float(item.get("strength", 0.0)) >= VISUAL_MIN_EVIDENCE_STRENGTH]
    if not top_evidence:
        top_evidence = top_evidence_all

    if VISUAL_SUPPRESS_LOW_CONF and explainability_confidence < VISUAL_MIN_EXPLAINABILITY_CONF:
        return _empty_visual_reasoning(
            target_idx,
            image_size,
            explainability_confidence,
            "Visual explanation was suppressed because explainability confidence is low.",
        )

    evidence_points = []
    for idx, point in enumerate(peaks):
        item = top_evidence[idx] if idx < len(top_evidence) else None
        point_type = item.get("type", "focus") if item else "focus"
        point_region = item.get("region", "highlighted area") if item else "highlighted area"
        point_detail = item.get("detail", "") if item else ""
        evidence_points.append(
            {
                "id": idx + 1,
                "x": point["x"],
                "y": point["y"],
                "strength": point["strength"],
                "type": point_type,
                "region": point_region,
                "detail": point_detail,
                "label": _human_visual_label(item, target_idx),
            }
        )

    avg_peak_strength = sum(float(p.get("strength", 0.0)) for p in evidence_points) / max(1, len(evidence_points))
    quality_score = int(round(100.0 * _clamp01(0.6 * avg_peak_strength + 0.4 * (explainability_confidence / 100.0))))
    if quality_score >= 65:
        quality = "high"
    elif quality_score >= 40:
        quality = "medium"
    else:
        quality = "low"

    return {
        "visualVersion": VISUAL_REASONING_VERSION,
        "targetClass": "ai" if target_idx == 0 else "real",
        "imageSize": image_size,
        "evidencePoints": evidence_points,
        "overlayImageBase64": _to_data_url_png(overlay_img),
        "heatmapImageBase64": _to_data_url_png(heatmap_img),
        "explainabilityConfidence": int(explainability_confidence),
        "quality": quality,
        "available": True,
        "warnings": [],
    }


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
    target_idx = 0 if ai_confidence >= 50 else 1

    visual_reasoning = None
    if ENABLE_VISUAL_REASONING:
        cam_224 = _compute_gradcam(model, x, target_idx)
        visual_reasoning = _build_visual_reasoning(image, cam_224, target_idx, reasoning_v2)
        if (
            visual_reasoning
            and not visual_reasoning.get("available", True)
            and visual_reasoning.get("warnings")
        ):
            warning_text = str(visual_reasoning["warnings"][0]).strip()
            if warning_text and warning_text not in reasons:
                reasons.append(warning_text)

    return {
        "ok": True,
        "aiConfidence": ai_confidence,
        "reasons": reasons,
        "reasoningV2": reasoning_v2,
        "reasoningVersion": "v2-image-cues",
        "visualReasoning": visual_reasoning,
        "visualReasoningVersion": VISUAL_REASONING_VERSION,
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
        "visualReasoningEnabled": ENABLE_VISUAL_REASONING,
        "visualReasoningVersion": VISUAL_REASONING_VERSION,
        "visualGuardrails": {
            "minPointStrength": VISUAL_MIN_POINT_STRENGTH,
            "minEvidenceStrength": VISUAL_MIN_EVIDENCE_STRENGTH,
            "minExplainabilityConfidence": VISUAL_MIN_EXPLAINABILITY_CONF,
            "suppressLowConfidence": VISUAL_SUPPRESS_LOW_CONF,
        },
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
