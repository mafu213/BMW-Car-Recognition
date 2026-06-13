import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

from train import build_model, get_eval_transform


def load_artifacts(model_path):
    project_dir = Path(__file__).resolve().parent
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = project_dir / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    checkpoint_dir = project_dir / "checkpoints"
    class_path = checkpoint_dir / "class_to_idx.json"
    if not class_path.exists():
        raise FileNotFoundError(f"类别映射文件不存在：{class_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_name = checkpoint.get("model_name", "mobilenet_v3_small")
        img_size = int(checkpoint.get("img_size", 224))
    else:
        state_dict = checkpoint
        model_name = "mobilenet_v3_small"
        img_size = 224

    class_to_idx = json.loads(class_path.read_text(encoding="utf-8"))
    idx_to_class = {int(idx): class_name for class_name, idx in class_to_idx.items()}

    desc_path = checkpoint_dir / "class_descriptions.json"
    descriptions = {}
    if desc_path.exists():
        descriptions = json.loads(desc_path.read_text(encoding="utf-8"))

    model = build_model(len(class_to_idx), model_name=model_name, use_pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device, idx_to_class, descriptions, img_size, model_name


def predict_image(image_path, model, device, idx_to_class, descriptions, img_size):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transform(img_size)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    top_k = min(4, len(idx_to_class))
    values, indices = torch.topk(probs, k=top_k)
    top_results = []
    for value, index in zip(values.tolist(), indices.tolist()):
        class_name = idx_to_class[int(index)]
        top_results.append({
            "class_name": class_name,
            "description": descriptions.get(class_name, ""),
            "probability": float(value),
        })
    return top_results


def main(args):
    model, device, idx_to_class, descriptions, img_size, model_name = load_artifacts(args.model_path)
    top_results = predict_image(args.image_path, model, device, idx_to_class, descriptions, img_size)
    best = top_results[0]
    display_name = best["class_name"]
    if best.get("description"):
        display_name = f"{display_name} - {best['description']}"

    print("========== 单张图片预测 ==========")
    print(f"模型：{model_name}")
    print(f"设备：{device}")
    print(f"图片：{args.image_path}")
    print(f"预测类别：{display_name}")
    print(f"置信度：{best['probability'] * 100:.2f}%")
    print("Top-4 概率：")
    for item in top_results:
        name = item["class_name"]
        if item.get("description"):
            name = f"{name} - {item['description']}"
        print(f"  {name}: {item['probability'] * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="BMW 单张车型图片预测")
    parser.add_argument("--image_path", type=str, required=True, help="待预测图片路径")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="模型权重路径")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main(parse_args())
