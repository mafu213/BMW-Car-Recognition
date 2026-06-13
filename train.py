import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# 固定随机种子，保证数据划分和训练过程尽量可复现。
DEFAULT_SEED = 2026
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageListDataset(Dataset):
    """从扫描得到的图片路径列表读取图像，避免强制改动原始数据目录。"""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_idx, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_project_dirs(project_dir):
    for name in ["checkpoints", "results", "report_materials", "templates", "static"]:
        (project_dir / name).mkdir(parents=True, exist_ok=True)


def list_image_files(directory):
    files = []
    if not directory.exists():
        return files
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def collect_class_samples(directory):
    """读取一个目录下的类别子文件夹，返回 {class_name: [image_path, ...]}。"""
    per_class = {}
    if not directory.exists():
        return per_class
    for child in sorted([p for p in directory.iterdir() if p.is_dir()], key=lambda p: p.name):
        # train/val/test 不是类别；外层套了一层真实数据集时，也不要把外层文件夹误判成类别。
        if child.name.lower() in SPLIT_NAMES:
            continue
        if any((child / split).is_dir() for split in SPLIT_NAMES):
            continue
        images = list_image_files(child)
        if images:
            per_class[child.name] = images
    return per_class


def looks_like_dataset_root(path):
    if not path.exists() or not path.is_dir():
        return False
    has_split = any(collect_class_samples(path / split) for split in SPLIT_NAMES)
    has_direct_classes = bool(collect_class_samples(path))
    return has_split or has_direct_classes


def resolve_dataset_root(data_dir):
    """兼容 D:\\BMW 和 D:\\BMW\\BMW 这类外层套一层的情况。"""
    raw_root = Path(data_dir).expanduser()
    if not raw_root.exists():
        raise FileNotFoundError(f"数据集路径不存在：{raw_root}")
    if looks_like_dataset_root(raw_root):
        return raw_root.resolve()

    candidates = []
    for child in sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if looks_like_dataset_root(child):
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0].resolve()
    if candidates:
        candidate_text = "\n".join(str(p) for p in candidates)
        raise RuntimeError(f"发现多个可能的数据集根目录，请明确指定其中一个：\n{candidate_text}")
    raise RuntimeError(
        "数据结构不符合预期。请使用 D:\\BMW\\类别名\\图片 或 "
        "D:\\BMW\\train\\类别名\\图片 / val / test 结构。"
    )


def parse_class_descriptions(dataset_root):
    """读取 readme.txt 中的可读车型名称，类别名仍以文件夹名为准。"""
    descriptions = {}
    for candidate in [dataset_root / "readme.txt", dataset_root.parent / "readme.txt"]:
        if not candidate.exists():
            continue
        try:
            for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    descriptions[parts[0]] = parts[1]
        except OSError:
            pass
    return descriptions


def split_two_way(per_class, train_ratio, seed):
    rng = random.Random(seed)
    train_part = {}
    val_part = {}
    for class_name, images in per_class.items():
        images = list(images)
        rng.shuffle(images)
        if len(images) <= 1:
            train_count = len(images)
        else:
            train_count = int(round(len(images) * train_ratio))
            train_count = max(1, min(train_count, len(images) - 1))
        train_part[class_name] = sorted(images[:train_count], key=lambda p: str(p).lower())
        val_part[class_name] = sorted(images[train_count:], key=lambda p: str(p).lower())
    return train_part, val_part


def split_three_way(per_class, seed):
    rng = random.Random(seed)
    train_part = {}
    val_part = {}
    test_part = {}
    for class_name, images in per_class.items():
        images = list(images)
        rng.shuffle(images)
        n = len(images)
        if n >= 3:
            train_count = max(1, int(n * 0.7))
            val_count = max(1, int(n * 0.2))
            if train_count + val_count >= n:
                train_count = max(1, n - 2)
                val_count = 1
        elif n == 2:
            train_count = 1
            val_count = 1
        else:
            train_count = n
            val_count = 0
        train_part[class_name] = sorted(images[:train_count], key=lambda p: str(p).lower())
        val_part[class_name] = sorted(images[train_count:train_count + val_count], key=lambda p: str(p).lower())
        test_part[class_name] = sorted(images[train_count + val_count:], key=lambda p: str(p).lower())
    return train_part, val_part, test_part


def merge_per_class(*parts):
    merged = defaultdict(list)
    for part in parts:
        for class_name, images in part.items():
            merged[class_name].extend(images)
    return {k: sorted(v, key=lambda p: str(p).lower()) for k, v in merged.items()}


def flatten_samples(per_class, class_to_idx):
    samples = []
    for class_name in sorted(per_class):
        for image_path in per_class[class_name]:
            samples.append((str(image_path), class_to_idx[class_name], class_name))
    return samples


def count_per_class(per_class):
    return {class_name: len(images) for class_name, images in sorted(per_class.items())}


def build_dataset_splits(data_dir, seed=DEFAULT_SEED):
    dataset_root = resolve_dataset_root(data_dir)
    split_dirs = {split: dataset_root / split for split in SPLIT_NAMES}
    found_splits = {
        split: collect_class_samples(split_dir)
        for split, split_dir in split_dirs.items()
        if split_dir.exists()
    }
    found_splits = {k: v for k, v in found_splits.items() if v}

    if found_splits:
        structure_type = "predefined_splits"
        if "train" not in found_splits:
            combined = merge_per_class(*found_splits.values())
            train_part, val_part, test_part = split_three_way(combined, seed)
            found_splits = {"train": train_part, "val": val_part, "test": test_part}
            structure_type = "split_from_existing_non_train_dirs"
        elif "val" not in found_splits and "test" in found_splits:
            train_part, val_part = split_two_way(found_splits["train"], 0.8, seed)
            found_splits["train"] = train_part
            found_splits["val"] = val_part
            structure_type = "train_test_with_val_split_from_train"
        elif "val" not in found_splits or "test" not in found_splits:
            combined = merge_per_class(*found_splits.values())
            train_part, val_part, test_part = split_three_way(combined, seed)
            found_splits = {"train": train_part, "val": val_part, "test": test_part}
            structure_type = "auto_split_7_2_1_from_partial_splits"
    else:
        direct = collect_class_samples(dataset_root)
        if not direct:
            raise RuntimeError("没有在数据集中找到可识别的图像文件。")
        train_part, val_part, test_part = split_three_way(direct, seed)
        found_splits = {"train": train_part, "val": val_part, "test": test_part}
        structure_type = "auto_split_7_2_1_from_class_folders"

    classes = sorted(set().union(*[set(part.keys()) for part in found_splits.values()]))
    if len(classes) != 4:
        print(f"[警告] 当前识别到 {len(classes)} 个类别：{classes}。项目要求为 BMW 四类，请确认数据集。")
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    splits = {split: flatten_samples(found_splits.get(split, {}), class_to_idx) for split in SPLIT_NAMES}
    descriptions = parse_class_descriptions(dataset_root)
    stats = make_dataset_stats(dataset_root, structure_type, found_splits, classes, descriptions, seed)
    return dataset_root, classes, class_to_idx, descriptions, splits, stats


def make_dataset_stats(dataset_root, structure_type, per_split, classes, descriptions, seed):
    split_counts = {}
    total_by_class = {class_name: 0 for class_name in classes}
    for split in SPLIT_NAMES:
        counts = count_per_class(per_split.get(split, {}))
        split_counts[split] = {class_name: counts.get(class_name, 0) for class_name in classes}
        for class_name, count in split_counts[split].items():
            total_by_class[class_name] += count

    total_images = int(sum(total_by_class.values()))
    nonzero_counts = [count for count in total_by_class.values() if count > 0]
    max_count = max(nonzero_counts) if nonzero_counts else 0
    min_count = min(nonzero_counts) if nonzero_counts else 0
    ratio = round(max_count / min_count, 4) if min_count else None
    majority = max(total_by_class, key=total_by_class.get) if total_by_class else None
    minority = min(total_by_class, key=total_by_class.get) if total_by_class else None

    return {
        "dataset_root": str(dataset_root),
        "structure_type": structure_type,
        "seed": seed,
        "classes": classes,
        "class_descriptions": descriptions,
        "total_images": total_images,
        "class_counts": total_by_class,
        "split_counts": split_counts,
        "split_totals": {split: int(sum(split_counts[split].values())) for split in SPLIT_NAMES},
        "imbalance": {
            "max_count": int(max_count),
            "min_count": int(min_count),
            "max_min_ratio": ratio,
            "majority_class": majority,
            "minority_class": minority,
            "is_imbalanced": bool(ratio and ratio >= 1.5),
        },
    }


def save_dataset_stats(stats, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "dataset_stats.json"
    csv_path = results_dir / "dataset_stats.csv"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_name", "description", "count"])
        descriptions = stats.get("class_descriptions", {})
        for split in SPLIT_NAMES:
            for class_name in stats["classes"]:
                writer.writerow([
                    split,
                    class_name,
                    descriptions.get(class_name, ""),
                    stats["split_counts"][split].get(class_name, 0),
                ])
        writer.writerow(["total", "ALL", "", stats["total_images"]])
    return json_path, csv_path


def get_train_transform(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
        transforms.RandomRotation(degrees=12),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(img_size):
    resize_size = int(img_size * 1.14)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_model(num_classes, model_name="mobilenet_v3_small", use_pretrained=False):
    """按名称构建模型。推理阶段 use_pretrained=False，避免再次下载权重。"""
    if model_name == "mobilenet_v3_small":
        weights = None
        if use_pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        try:
            model = models.mobilenet_v3_small(weights=weights)
        except TypeError:
            model = models.mobilenet_v3_small(pretrained=bool(use_pretrained))
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = None
        if use_pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        try:
            model = models.efficientnet_b0(weights=weights)
        except TypeError:
            model = models.efficientnet_b0(pretrained=bool(use_pretrained))
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet18":
        try:
            model = models.resnet18(weights=None)
        except TypeError:
            model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"不支持的模型名称：{model_name}")


def create_training_model(num_classes):
    """优先使用 ImageNet 预训练 MobileNetV3-Small，失败时自动降级。"""
    try:
        print("[模型] 尝试加载 ImageNet 预训练 MobileNetV3-Small ...")
        model = build_model(num_classes, "mobilenet_v3_small", use_pretrained=True)
        return model, "mobilenet_v3_small", True
    except Exception as exc:
        print(f"[模型] 预训练 MobileNetV3-Small 加载失败：{exc}")
        print("[模型] 自动降级为 ResNet18 随机初始化，继续训练。")
        model = build_model(num_classes, "resnet18", use_pretrained=False)
        return model, "resnet18", False


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).sum().item(), targets.size(0)


def run_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        correct, count = accuracy_from_logits(outputs.detach(), labels)
        total_loss += loss.item() * count
        total_correct += correct
        total_samples += count
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        correct, count = accuracy_from_logits(outputs, labels)
        total_loss += loss.item() * count
        total_correct += correct
        total_samples += count
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def save_training_curves(history, results_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, [v * 100 for v in history["train_acc"]], marker="o", label="Train Accuracy")
    plt.plot(epochs, [v * 100 for v in history["val_acc"]], marker="s", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    acc_path = results_dir / "accuracy_curve.png"
    plt.savefig(acc_path)
    plt.close()

    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    loss_path = results_dir / "loss_curve.png"
    plt.savefig(loss_path)
    plt.close()
    return acc_path, loss_path


def print_dataset_summary(stats):
    print("\n========== 数据集统计 ==========")
    print(f"数据根目录：{stats['dataset_root']}")
    print(f"结构识别：{stats['structure_type']}")
    print(f"类别：{', '.join(stats['classes'])}")
    for class_name, count in stats["class_counts"].items():
        description = stats.get("class_descriptions", {}).get(class_name, "")
        suffix = f" - {description}" if description else ""
        print(f"  {class_name}{suffix}: {count} 张")
    print(f"总图像数量：{stats['total_images']} 张")
    for split, total in stats["split_totals"].items():
        print(f"{split}: {total} 张 -> {stats['split_counts'][split]}")
    imbalance = stats["imbalance"]
    print(
        "类别不均衡："
        f"最大/最小={imbalance['max_min_ratio']}，"
        f"多数类={imbalance['majority_class']}，少数类={imbalance['minority_class']}，"
        f"是否明显不均衡={imbalance['is_imbalanced']}"
    )
    print("================================\n")


def train(args):
    project_dir = Path(__file__).resolve().parent
    checkpoints_dir = project_dir / "checkpoints"
    results_dir = project_dir / "results"
    ensure_project_dirs(project_dir)
    set_seed(args.seed)

    dataset_root, classes, class_to_idx, descriptions, splits, stats = build_dataset_splits(args.data_dir, args.seed)
    save_dataset_stats(stats, results_dir)
    print_dataset_summary(stats)

    if not splits["train"]:
        raise RuntimeError("训练集为空，无法训练。")
    if not splits["val"]:
        raise RuntimeError("验证集为空，无法保存最佳模型。")

    (checkpoints_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (checkpoints_dir / "class_descriptions.json").write_text(
        json.dumps(descriptions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_dataset = ImageListDataset(splits["train"], transform=get_train_transform(args.img_size))
    val_dataset = ImageListDataset(splits["val"], transform=get_eval_transform(args.img_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用：{device}")
    if device.type == "cuda":
        print(f"[设备] GPU：{torch.cuda.get_device_name(0)}")

    model, model_name, used_pretrained = create_training_model(len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = 0
    best_path = checkpoints_dir / "best_model.pth"
    start_time = time.time()

    print("\n========== 开始训练 ==========")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "num_classes": len(classes),
                    "class_to_idx": class_to_idx,
                    "class_descriptions": descriptions,
                    "img_size": args.img_size,
                    "best_val_acc": float(best_val_acc),
                    "best_epoch": int(best_epoch),
                    "used_pretrained": bool(used_pretrained),
                    "train_args": vars(args),
                },
                best_path,
            )

        mark = " *保存最佳*" if improved else ""
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc * 100:.2f}% val_acc={val_acc * 100:.2f}%{mark}"
        )

    elapsed = time.time() - start_time
    print(f"========== 训练结束，用时 {elapsed / 60:.2f} 分钟 ==========")
    print(f"最佳验证准确率：{best_val_acc * 100:.2f}% (epoch {best_epoch})")
    print(f"最佳模型已保存：{best_path}")

    (results_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    acc_path, loss_path = save_training_curves(history, results_dir)
    print(f"训练准确率曲线：{acc_path}")
    print(f"损失曲线：{loss_path}")

    if not args.skip_eval:
        print("\n========== 自动运行 evaluate.py ==========")
        eval_cmd = [
            sys.executable,
            str(project_dir / "evaluate.py"),
            "--data_dir",
            str(args.data_dir),
            "--model_path",
            str(best_path),
            "--img_size",
            str(args.img_size),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ]
        subprocess.run(eval_cmd, cwd=str(project_dir), check=True)

    return best_val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="BMW 四类车型识别训练脚本")
    parser.add_argument("--data_dir", type=str, default=r"D:\BMW", help="BMW 数据集根目录")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW 初始学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW 权重衰减")
    parser.add_argument("--num_workers", type=int, default=0, help="Windows 下默认 0 更稳定")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="固定随机种子")
    parser.add_argument("--skip_eval", action="store_true", help="训练后不自动运行评估")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    train(parse_args())
