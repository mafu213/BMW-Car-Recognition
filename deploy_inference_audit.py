import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


PROJECT_DIR = Path(__file__).resolve().parent
RUN_DIR = PROJECT_DIR / "runs_deploy_audit"
RUN_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_IDS = ["28", "29", "32", "37"]
IDX_TO_CLASS = {str(i): cid for i, cid in enumerate(CLASS_IDS)}


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return (exp / exp.sum()).astype(np.float64)


def fmt_probs(probs):
    return json.dumps(
        {IDX_TO_CLASS[str(i)]: round(float(probs[i]), 6) for i in range(len(CLASS_IDS))},
        ensure_ascii=False,
    )


def top1_from_probs(probs):
    index = int(np.argmax(probs))
    return IDX_TO_CLASS[str(index)]


def load_efficientnet(checkpoint_path):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CLASS_IDS))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state, dict) and any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def eval_tensor(image):
    transform = transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 1.14), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
        ]
    )
    return transform(image).unsqueeze(0).numpy().astype(np.float32)


def web_stretch_tensor(image):
    # Mirrors the old browser code: drawImage(source, 0, 0, 224, 224).
    resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    chw = np.transpose(arr, (2, 0, 1))
    return chw[np.newaxis, ...].astype(np.float32)


def web_cover_crop_tensor(image):
    # Mirrors the fixed browser cover-center-crop preprocessing.
    src_w, src_h = image.size
    scale = max(IMG_SIZE / src_w, IMG_SIZE / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    left = max(0, (new_w - IMG_SIZE) // 2)
    top = max(0, (new_h - IMG_SIZE) // 2)
    crop = resized.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = np.asarray(crop).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    chw = np.transpose(arr, (2, 0, 1))
    return chw[np.newaxis, ...].astype(np.float32)


def collect_samples(per_class=5):
    rows = []
    for split_name in ["train", "val", "test"]:
        csv_path = PROJECT_DIR / "runs_accuracy_boost" / f"{split_name}_split.csv"
        if not csv_path.exists():
            continue
        by_class = {cid: [] for cid in CLASS_IDS}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = str(row["class_name"])
                if cid in by_class and len(by_class[cid]) < per_class:
                    by_class[cid].append(row["path"])
        for cid in CLASS_IDS:
            for path in by_class[cid]:
                rows.append({"split": split_name, "true_label": cid, "image_path": path})
    return rows


def run_audit():
    checkpoint_path = PROJECT_DIR / "runs_accuracy_boost" / "efficientnet_b0" / "best_model.pth"
    onnx_path = PROJECT_DIR / "mobile_web" / "model" / "bmw_model.onnx"
    model = load_efficientnet(checkpoint_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rows = []
    summary = {
        "samples": 0,
        "pytorch_onnx_top1_match": 0,
        "old_web_stretch_matches_onnx_eval": 0,
        "fixed_web_crop_matches_onnx_eval": 0,
        "old_web_stretch_top1_counts": {},
        "fixed_web_crop_top1_counts": {},
        "onnx_eval_top1_counts": {},
        "max_logits_error": 0.0,
    }

    with torch.no_grad():
        for sample in collect_samples():
            image = Image.open(sample["image_path"]).convert("RGB")
            eval_input = eval_tensor(image)
            stretch_input = web_stretch_tensor(image)
            crop_input = web_cover_crop_tensor(image)

            torch_logits = model(torch.from_numpy(eval_input)).numpy()[0]
            onnx_logits = session.run([output_name], {input_name: eval_input})[0][0]
            old_web_logits = session.run([output_name], {input_name: stretch_input})[0][0]
            fixed_web_logits = session.run([output_name], {input_name: crop_input})[0][0]

            torch_probs = softmax(torch_logits)
            onnx_probs = softmax(onnx_logits)
            old_web_probs = softmax(old_web_logits)
            fixed_web_probs = softmax(fixed_web_logits)

            torch_top1 = top1_from_probs(torch_probs)
            onnx_top1 = top1_from_probs(onnx_probs)
            old_web_top1 = top1_from_probs(old_web_probs)
            fixed_web_top1 = top1_from_probs(fixed_web_probs)
            max_error = float(np.max(np.abs(torch_logits - onnx_logits)))

            summary["samples"] += 1
            summary["pytorch_onnx_top1_match"] += int(torch_top1 == onnx_top1)
            summary["old_web_stretch_matches_onnx_eval"] += int(old_web_top1 == onnx_top1)
            summary["fixed_web_crop_matches_onnx_eval"] += int(fixed_web_top1 == onnx_top1)
            summary["max_logits_error"] = max(summary["max_logits_error"], max_error)
            summary["onnx_eval_top1_counts"][onnx_top1] = summary["onnx_eval_top1_counts"].get(onnx_top1, 0) + 1
            summary["old_web_stretch_top1_counts"][old_web_top1] = summary["old_web_stretch_top1_counts"].get(old_web_top1, 0) + 1
            summary["fixed_web_crop_top1_counts"][fixed_web_top1] = summary["fixed_web_crop_top1_counts"].get(fixed_web_top1, 0) + 1

            rows.append(
                {
                    **sample,
                    "pytorch_top1": torch_top1,
                    "pytorch_probs": fmt_probs(torch_probs),
                    "onnx_top1": onnx_top1,
                    "onnx_probs": fmt_probs(onnx_probs),
                    "top1_consistent": torch_top1 == onnx_top1,
                    "max_logits_error": max_error,
                    "old_web_stretch_top1": old_web_top1,
                    "old_web_stretch_probs": fmt_probs(old_web_probs),
                    "fixed_web_crop_top1": fixed_web_top1,
                    "fixed_web_crop_probs": fmt_probs(fixed_web_probs),
                }
            )

    csv_path = RUN_DIR / "pytorch_vs_onnx.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    web_rows = [
        {
            "image_path": row["image_path"],
            "true_label": row["true_label"],
            "expected_web_top1": row["fixed_web_crop_top1"],
            "expected_web_probs": row["fixed_web_crop_probs"],
            "onnx_eval_top1": row["onnx_top1"],
            "onnx_eval_probs": row["onnx_probs"],
        }
        for row in rows
        if row["split"] == "test"
    ]
    web_csv_path = RUN_DIR / "web_check_expected.csv"
    with web_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(web_rows[0].keys()))
        writer.writeheader()
        writer.writerows(web_rows)

    summary["pytorch_onnx_top1_match_rate"] = summary["pytorch_onnx_top1_match"] / max(1, summary["samples"])
    summary["old_web_stretch_match_rate"] = summary["old_web_stretch_matches_onnx_eval"] / max(1, summary["samples"])
    summary["fixed_web_crop_match_rate"] = summary["fixed_web_crop_matches_onnx_eval"] / max(1, summary["samples"])
    summary["csv_path"] = str(csv_path)
    summary["web_check_expected_csv_path"] = str(web_csv_path)
    (RUN_DIR / "pytorch_vs_onnx_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_audit()
