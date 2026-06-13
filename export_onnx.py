import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from train import build_dataset_splits, build_model, get_eval_transform


def load_checkpoint(model_path, class_to_idx_path):
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("当前导出脚本需要 train.py 保存的 checkpoint 字典格式。")

    class_to_idx = json.loads(Path(class_to_idx_path).read_text(encoding="utf-8"))
    idx_to_class = {str(idx): class_name for class_name, idx in class_to_idx.items()}
    model_name = checkpoint.get("model_name", "mobilenet_v3_small")
    img_size = int(checkpoint.get("img_size", 224))

    model = build_model(len(class_to_idx), model_name=model_name, use_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, model_name, img_size, class_to_idx, idx_to_class, checkpoint


def export_model(model, output_path, img_size, opset):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def image_to_numpy(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transform(img_size)
    tensor = transform(image).unsqueeze(0)
    return tensor.numpy().astype(np.float32)


def collect_test_images(data_dir, seed, count):
    _, _, _, _, splits, _ = build_dataset_splits(data_dir, seed)
    samples = splits.get("test") or splits.get("val") or []
    if not samples:
        raise RuntimeError("没有找到可用于 ONNX 验证的 test/val 图片。")
    rng = random.Random(seed)
    selected = list(samples)
    rng.shuffle(selected)
    return selected[: min(count, len(selected))]


def check_onnx_consistency(model, onnx_path, img_size, data_dir, seed, count):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_samples = collect_test_images(data_dir, seed, count)
    max_abs_error = 0.0
    checked = []
    mismatches = []

    with torch.no_grad():
        for image_path, label_idx, class_name in test_samples:
            input_np = image_to_numpy(image_path, img_size)
            torch_logits = model(torch.from_numpy(input_np)).numpy()
            onnx_logits = session.run(["logits"], {"input": input_np})[0]

            error = float(np.max(np.abs(torch_logits - onnx_logits)))
            max_abs_error = max(max_abs_error, error)
            torch_top1 = int(np.argmax(torch_logits, axis=1)[0])
            onnx_top1 = int(np.argmax(onnx_logits, axis=1)[0])
            row = {
                "image_path": str(image_path),
                "true_class": class_name,
                "torch_top1": torch_top1,
                "onnx_top1": onnx_top1,
                "max_abs_error": error,
            }
            checked.append(row)
            if torch_top1 != onnx_top1:
                mismatches.append(row)

    return {
        "checked_images": len(checked),
        "top1_consistent": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "max_abs_error": max_abs_error,
        "mismatches": mismatches,
        "samples": checked,
    }


def main(args):
    project_dir = Path(__file__).resolve().parent
    model_path = Path(args.model_path)
    class_to_idx_path = Path(args.class_to_idx)
    if not model_path.is_absolute():
        model_path = project_dir / model_path
    if not class_to_idx_path.is_absolute():
        class_to_idx_path = project_dir / class_to_idx_path

    mobile_model_dir = project_dir / "mobile_web" / "model"
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model, model_name, img_size, class_to_idx, idx_to_class, checkpoint = load_checkpoint(model_path, class_to_idx_path)
    onnx_path = mobile_model_dir / "bmw_model.onnx"

    export_error = None
    for opset in [args.opset, 13, 12]:
        try:
            print(f"[ONNX] 导出 {model_name}，opset={opset} -> {onnx_path}")
            export_model(model, onnx_path, img_size, opset)
            export_error = None
            break
        except Exception as exc:
            export_error = exc
            print(f"[ONNX] opset={opset} 导出失败：{exc}")
    if export_error is not None:
        raise RuntimeError(
            "MobileNetV3-Small 导出失败。请先训练或提供浏览器兼容性更好的 ResNet18/EfficientNet-B0 checkpoint。"
        ) from export_error

    (mobile_model_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (mobile_model_dir / "idx_to_class.json").write_text(
        json.dumps(idx_to_class, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    check = check_onnx_consistency(model, onnx_path, img_size, args.data_dir, args.seed, args.check_count)
    check.update({
        "model_name": model_name,
        "onnx_path": str(onnx_path),
        "input_name": "input",
        "input_shape": [1, 3, img_size, img_size],
        "input_dtype": "float32",
        "output_name": "logits",
        "output_shape": [1, len(class_to_idx)],
        "best_val_acc": checkpoint.get("best_val_acc"),
    })
    check_path = results_dir / "onnx_check.json"
    check_path.write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== ONNX 导出完成 ==========")
    print(f"ONNX 模型：{onnx_path}")
    print(f"类别映射：{mobile_model_dir / 'class_to_idx.json'}")
    print(f"一致性检查：{check_path}")
    print(f"检查图片数：{check['checked_images']}")
    print(f"Top-1 是否全部一致：{check['top1_consistent']}")
    print(f"最大数值误差：{check['max_abs_error']:.8f}")
    if not check["top1_consistent"]:
        raise RuntimeError("PyTorch 与 ONNX 存在 top1 不一致，请检查导出模型。")


def parse_args():
    parser = argparse.ArgumentParser(description="导出 BMW PyTorch 模型为浏览器可用的 ONNX 模型")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--class_to_idx", type=str, default="checkpoints/class_to_idx.json")
    parser.add_argument("--data_dir", type=str, default=r"D:\BMW")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--check_count", type=int, default=10)
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
