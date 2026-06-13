import argparse
import json
import math
import os
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from train import (
    ImageListDataset,
    build_dataset_splits,
    build_model,
    ensure_project_dirs,
    get_eval_transform,
    save_dataset_stats,
)


def load_class_mapping(checkpoint, checkpoint_dir):
    class_to_idx_path = checkpoint_dir / "class_to_idx.json"
    if class_to_idx_path.exists():
        class_to_idx = json.loads(class_to_idx_path.read_text(encoding="utf-8"))
    else:
        class_to_idx = checkpoint.get("class_to_idx")
    if not class_to_idx:
        raise RuntimeError("缺少 checkpoints/class_to_idx.json，无法恢复类别映射。")
    idx_to_class = {int(idx): class_name for class_name, idx in class_to_idx.items()}

    desc_path = checkpoint_dir / "class_descriptions.json"
    if desc_path.exists():
        descriptions = json.loads(desc_path.read_text(encoding="utf-8"))
    else:
        descriptions = checkpoint.get("class_descriptions", {})
    return class_to_idx, idx_to_class, descriptions


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []
    all_probs = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.extend(probs.tolist())
        all_pred.extend(preds.tolist())
        all_true.extend(labels.numpy().tolist())
    return np.array(all_true), np.array(all_pred), np.array(all_probs)


def plot_confusion_matrix(cm, labels, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=170)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Class",
        xlabel="Predicted Class",
        title="BMW Classification Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            pct = (cm[i, j] / row_sum * 100) if row_sum else 0
            text = f"{cm[i, j]}\n{pct:.1f}%"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "#1f2937",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_structure(output_path, model_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = [
        ("Input BMW Image\n224 x 224 x 3", "#DDEBFF"),
        ("Data Augmentation\nCrop / Flip / Color / Rotation", "#E7F7ED"),
        (f"CNN Backbone\n{model_name}", "#FFF3D6"),
        ("Global Average Pooling", "#FCE7F3"),
        ("Fully Connected Layer\n4 Classes", "#EDE9FE"),
        ("Softmax\nBMW Class Probability", "#DCFCE7"),
    ]

    fig, ax = plt.subplots(figsize=(13, 3.8), dpi=180)
    ax.set_axis_off()
    box_w = 1.82
    box_h = 0.86
    gap = 0.28
    x0 = 0.25
    y0 = 0.62

    for idx, (label, color) in enumerate(steps):
        x = x0 + idx * (box_w + gap)
        patch = FancyBboxPatch(
            (x, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.03,rounding_size=0.045",
            linewidth=1.4,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + box_w / 2, y0 + box_h / 2, label, ha="center", va="center", fontsize=10.5, color="#111827")
        if idx < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + box_w + 0.03, y0 + box_h / 2),
                (x + box_w + gap - 0.05, y0 + box_h / 2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#475569",
            )
            ax.add_patch(arrow)

    ax.text(
        x0,
        1.78,
        "BMW Four-Class Recognition Network Workflow",
        fontsize=15,
        weight="bold",
        color="#0f172a",
    )
    ax.set_xlim(0, x0 + len(steps) * (box_w + gap) - gap + 0.25)
    ax.set_ylim(0.2, 2.05)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_sample_predictions(samples, true_labels, pred_labels, probs, idx_to_class, descriptions, img_size, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not samples:
        return
    rng = np.random.default_rng(2026)
    count = min(12, len(samples))
    chosen = rng.choice(len(samples), size=count, replace=False)
    rows = math.ceil(count / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3.1 * rows), dpi=160)
    axes = np.array(axes).reshape(-1)

    for ax_idx, sample_idx in enumerate(chosen):
        ax = axes[ax_idx]
        image_path, _, _ = samples[sample_idx]
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((img_size, img_size))
        ax.imshow(image)
        pred_idx = int(pred_labels[sample_idx])
        true_idx = int(true_labels[sample_idx])
        pred_class = idx_to_class[pred_idx]
        true_class = idx_to_class[true_idx]
        confidence = float(probs[sample_idx][pred_idx])
        title_color = "#15803d" if pred_idx == true_idx else "#b91c1c"
        pred_text = descriptions.get(pred_class, pred_class)
        true_text = descriptions.get(true_class, true_class)
        title = f"T: {true_class} | P: {pred_class}\nConf: {confidence * 100:.1f}%"
        if len(pred_text) <= 22 and len(true_text) <= 22:
            title = f"T: {true_text}\nP: {pred_text} ({confidence * 100:.1f}%)"
        ax.set_title(title, fontsize=8.5, color=title_color)
        ax.axis("off")

    for ax in axes[count:]:
        ax.axis("off")
    fig.suptitle("Sample BMW Predictions", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_report_materials(project_dir, stats, metrics, checkpoint):
    report_dir = project_dir / "report_materials"
    report_dir.mkdir(parents=True, exist_ok=True)
    classes = stats["classes"]
    descriptions = stats.get("class_descriptions", {})
    model_name = checkpoint.get("model_name", "unknown")
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    train_args = checkpoint.get("train_args", {})
    class_lines = "\n".join(
        f"- {cls}: {descriptions.get(cls, '以文件夹名作为类别')}" for cls in classes
    )
    per_class_lines = "\n".join(
        f"- {cls}: {acc * 100:.2f}%" for cls, acc in metrics.get("per_class_accuracy", {}).items()
    )

    (report_dir / "algorithm_description.md").write_text(
        f"""# 算法原理说明

## 任务背景
本项目面向 BMW 四类车型图像识别任务，目标是在给定车辆图片后自动判断其所属车型类别。数据集类别以文件夹名为准，本次识别到的类别如下：

{class_lines}

## 车型识别难点
不同 BMW 车型之间存在相似的车身线条、进气格栅、车灯和品牌标识，类间差异相对细微；同时车辆图片可能受到拍摄角度、光照、背景遮挡、车身颜色和分辨率变化影响。传统手工特征难以稳定覆盖这些变化，因此需要使用深度学习自动提取更鲁棒的视觉特征。

## 本项目采用的深度学习方法
项目采用 PyTorch 构建卷积神经网络分类器，输入图像统一缩放到 224×224。训练阶段使用 RandomResizedCrop、RandomHorizontalFlip、ColorJitter、RandomRotation 和 Normalize 进行数据增强，提升模型对角度、颜色和尺度变化的适应能力。分类器使用 CrossEntropyLoss 作为损失函数，使用 AdamW 优化器进行参数更新。

## 为什么使用迁移学习
车型数据集规模较小，从零训练深层网络容易过拟合。迁移学习利用 ImageNet 上学习到的通用边缘、纹理、形状和物体部件特征，只需在 BMW 数据上微调分类层和骨干网络参数，就能在较少样本下获得更稳定的识别效果。本项目优先使用 ImageNet 预训练的 MobileNetV3-Small；如果预训练权重无法下载，则自动降级为可继续训练的 ResNet18。
""",
        encoding="utf-8",
    )

    (report_dir / "network_structure.md").write_text(
        f"""# 网络结构说明

本项目实际保存的最佳模型骨干网络为 `{model_name}`，整体流程可参考 `results/model_structure.png`。

## 输入层
输入为 RGB 车辆图像，统一处理为 224×224×3，并使用 ImageNet 均值和标准差进行标准化。

## 特征提取层
CNN Backbone 负责从图片中提取车身轮廓、车灯、前脸、侧面线条、SUV/轿车比例等层级特征。MobileNetV3-Small 具有轻量化特点，适合课程项目和现场网页演示；在 GPU 或 CPU 上推理都比较稳定。

## 分类层
骨干网络输出的高维特征经过 Global Average Pooling 聚合为空间特征向量，然后输入全连接层。全连接层输出 4 个类别的 logits，对应四类 BMW 车型。

## Softmax 输出
Softmax 将 logits 转换为概率分布，网页端展示最高概率类别和 Top-4 概率条形图。最终类别取概率最大的车型。
""",
        encoding="utf-8",
    )

    (report_dir / "experiment_results.md").write_text(
        f"""# 实验结果分析

## 数据集划分
数据根目录：`{stats['dataset_root']}`

结构识别方式：`{stats['structure_type']}`

- train: {stats['split_totals']['train']} 张
- val: {stats['split_totals']['val']} 张
- test: {stats['split_totals']['test']} 张
- 总计: {stats['total_images']} 张

各类别数量：{json.dumps(stats['class_counts'], ensure_ascii=False)}

## 训练参数
- 输入尺寸：{train_args.get('img_size', 224)}×{train_args.get('img_size', 224)}
- 训练轮数：{train_args.get('epochs', '见训练命令')}
- batch size：{train_args.get('batch_size', '见训练命令')}
- 优化器：AdamW
- 初始学习率：{train_args.get('lr', 1e-4)}
- 损失函数：CrossEntropyLoss
- 最佳验证准确率：{best_val_acc * 100:.2f}%

## 训练曲线分析
训练准确率曲线保存在 `results/accuracy_curve.png`。该曲线用于观察模型在训练集和验证集上的准确率变化。如果训练准确率持续升高而验证准确率停滞或下降，说明可能出现过拟合；如果两条曲线同步上升，说明模型正在有效学习可泛化特征。

## 损失函数曲线分析
损失曲线保存在 `results/loss_curve.png`。训练损失下降表示模型对训练样本拟合增强；验证损失下降说明模型对未见样本的预测更稳定。若验证损失后期上升，可考虑增加数据增强、提前停止或降低学习率。

## 混淆矩阵分析
混淆矩阵保存在 `results/confusion_matrix.png`。对角线数值代表预测正确数量，非对角线代表误分类情况。若某两类之间混淆较多，通常说明它们在车身外观、拍摄角度或局部特征上相似。

## 最终识别率
- 测试集准确率：{metrics.get('test_accuracy', 0) * 100:.2f}%
- Macro Precision：{metrics.get('macro_precision', 0) * 100:.2f}%
- Macro Recall：{metrics.get('macro_recall', 0) * 100:.2f}%
- Macro F1-score：{metrics.get('macro_f1', 0) * 100:.2f}%

## 各类别识别情况
{per_class_lines}
""",
        encoding="utf-8",
    )

    (report_dir / "code_explanation.md").write_text(
        """# 代码说明

## train.py 作用
`train.py` 负责扫描数据集、判断类别、自动划分训练/验证/测试集、保存数据统计、构建模型、执行训练、保存验证集准确率最高的 `checkpoints/best_model.pth`，并在训练结束后自动调用 `evaluate.py` 生成测试结果。

## evaluate.py 作用
`evaluate.py` 加载训练好的模型和类别映射，在测试集上计算 accuracy、precision、recall、F1-score 和各类别识别率，并生成混淆矩阵、分类报告、样例预测图和网络结构示意图。

## predict.py 作用
`predict.py` 用于单张图片命令行预测。输入图片路径和模型路径后，脚本会输出预测类别、置信度以及 Top-4 概率。

## app.py 作用
`app.py` 使用 FastAPI 搭建网页后端，启动后加载 `checkpoints/best_model.pth` 和 `checkpoints/class_to_idx.json`。浏览器上传图片后，后端完成预处理、模型推理和概率计算，并返回 JSON 结果。

## 网页端如何完成上传与识别
网页由 `templates/index.html`、`static/style.css` 和 `static/script.js` 构成。用户在 iPhone Safari 中选择拍照或相册图片，前端显示预览；点击“开始识别”后，JavaScript 使用 `fetch` 将图片提交到 `/predict` 接口，并把预测类别、置信度和 Top-4 概率条形图显示在页面上。
""",
        encoding="utf-8",
    )


def evaluate(args):
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    checkpoints_dir = project_dir / "checkpoints"
    ensure_project_dirs(project_dir)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_dir / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        checkpoint = {}

    class_to_idx, idx_to_class, descriptions = load_class_mapping(checkpoint, checkpoints_dir)
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    model_name = checkpoint.get("model_name", "mobilenet_v3_small")
    img_size = int(checkpoint.get("img_size", args.img_size))

    dataset_root, classes, _, scanned_descriptions, splits, stats = build_dataset_splits(args.data_dir, args.seed)
    if scanned_descriptions and not descriptions:
        descriptions = scanned_descriptions
    save_dataset_stats(stats, results_dir)

    test_samples = splits["test"] if splits["test"] else splits["val"]
    if not test_samples:
        raise RuntimeError("测试集和验证集都为空，无法评估。")

    test_dataset = ImageListDataset(test_samples, transform=get_eval_transform(img_size))
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(len(class_to_idx), model_name=model_name, use_pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    true_labels, pred_labels, probs = collect_predictions(model, test_loader, device)

    label_ids = list(range(len(labels)))
    test_accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=label_ids, average="macro", zero_division=0
    )
    report = classification_report(
        true_labels,
        pred_labels,
        labels=label_ids,
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=label_ids)
    per_class_accuracy = {}
    for i, class_name in enumerate(labels):
        row_sum = cm[i].sum()
        per_class_accuracy[class_name] = float(cm[i, i] / row_sum) if row_sum else 0.0

    metrics = {
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class_accuracy": per_class_accuracy,
        "model_name": model_name,
        "model_path": str(model_path),
        "test_samples": int(len(test_samples)),
    }

    (results_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (results_dir / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_confusion_matrix(cm, labels, results_dir / "confusion_matrix.png")
    plot_sample_predictions(
        test_samples,
        true_labels,
        pred_labels,
        probs,
        idx_to_class,
        descriptions,
        img_size,
        results_dir / "sample_predictions.png",
    )
    plot_model_structure(results_dir / "model_structure.png", model_name)
    write_report_materials(project_dir, stats, metrics, checkpoint)

    print("\n========== 测试集评估 ==========")
    print(report)
    print(f"测试集准确率：{test_accuracy * 100:.2f}%")
    print(f"Macro F1：{f1 * 100:.2f}%")
    print(f"混淆矩阵：{results_dir / 'confusion_matrix.png'}")
    print(f"样例预测：{results_dir / 'sample_predictions.png'}")
    print(f"最终指标：{results_dir / 'final_metrics.json'}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="BMW 四类车型识别评估脚本")
    parser.add_argument("--data_dir", type=str, default=r"D:\BMW", help="BMW 数据集根目录")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="模型权重路径")
    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="Windows 下默认 0 更稳定")
    parser.add_argument("--seed", type=int, default=2026, help="固定随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    evaluate(parse_args())
