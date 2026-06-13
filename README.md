# BMW 四类车型识别网页系统

本项目使用 PyTorch 训练 BMW 四类车型分类模型，并提供可在 iPhone Safari 中现场使用的 FastAPI 网页识别系统。网页支持上传图片、手机拍照、显示预测车型、置信度和 Top-4 概率条形图。

## 本次训练结果

- 模型：MobileNetV3-Small，ImageNet 预训练权重
- 最佳验证准确率：68.75%
- 测试集准确率：69.62%
- Macro Precision：69.54%
- Macro Recall：69.70%
- Macro F1-score：69.18%
- 测试集各类别识别率：`28` 90.00%，`29` 58.54%，`32` 60.00%，`37` 70.27%

## 1. 环境安装

推荐 Python 3.9+。在本机已使用 `D:\Anaconda\envs\DL` 环境完成训练与测试：

```bash
conda activate DL
pip install -r requirements.txt
```

如果是新电脑，可先安装 PyTorch，再安装其余依赖：

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## 2. 数据集路径说明

默认数据集路径：

```text
D:\BMW
```

脚本会自动识别以下结构：

```text
D:\BMW\class1\*.jpg
D:\BMW\train\class1\*.jpg
D:\BMW\val\class1\*.jpg
D:\BMW\test\class1\*.jpg
```

本机数据实际根目录会自动解析为 `D:\BMW\BMW`，类别以文件夹名为准：`28`、`29`、`32`、`37`。若只有总数据集，脚本会按 `train/val/test = 7:2:1` 使用随机种子 `2026` 自动划分；若只有 `train/test`，脚本会从 `train` 中划分验证集。

## 3. 训练命令

```bash
python train.py --data_dir "D:\BMW" --epochs 30 --batch_size 16 --img_size 224
```

训练会自动保存：

- `checkpoints/best_model.pth`
- `checkpoints/class_to_idx.json`
- `results/dataset_stats.json`
- `results/dataset_stats.csv`
- `results/accuracy_curve.png`
- `results/loss_curve.png`

训练结束后会自动运行 `evaluate.py`。

## 4. 测试命令

```bash
python evaluate.py --data_dir "D:\BMW" --model_path checkpoints/best_model.pth
```

测试结果保存到：

- `results/confusion_matrix.png`
- `results/classification_report.txt`
- `results/final_metrics.json`
- `results/sample_predictions.png`
- `results/model_structure.png`

## 5. 单张预测命令

```bash
python predict.py --image_path "某张图片路径" --model_path checkpoints/best_model.pth
```

## 6. 启动网页命令

```bash
python app.py
```

服务监听：

```text
host=0.0.0.0
port=8000
```

电脑浏览器可打开：

```text
http://127.0.0.1:8000
```

## 7. iPhone 现场验收步骤

1. 电脑和 iPhone 连接同一个 Wi-Fi。
2. 电脑进入项目目录并运行：

```bash
python app.py
```

3. 查询电脑局域网 IP。Windows 可执行：

```bash
ipconfig
```

4. 在 iPhone Safari 打开：

```text
http://电脑IP:8000
```

5. 点击“选择图片”，可拍照或从相册选择图片，再点击“开始识别”。

## 8. GitHub 上传步骤

本项目保留 `checkpoints/best_model.pth`，便于现场验收。首次上传可执行：

```bash
git remote add origin <你的仓库地址>
git branch -M main
git push -u origin main
```

如果已经配置 remote，可直接：

```bash
git push
```

## 9. 项目结果展示位置

课程报告可直接引用以下文件：

- 数据统计：`results/dataset_stats.json`、`results/dataset_stats.csv`
- 训练曲线：`results/accuracy_curve.png`、`results/loss_curve.png`
- 混淆矩阵：`results/confusion_matrix.png`
- 分类报告：`results/classification_report.txt`
- 最终指标：`results/final_metrics.json`
- 样例预测：`results/sample_predictions.png`
- 网络结构示意图：`results/model_structure.png`
- 中文报告素材：`report_materials/`

## 10. 常见问题

### 没有 GPU 怎么办

脚本会自动检测 CUDA。如果没有 GPU，会使用 CPU 继续训练和推理，只是训练速度会慢一些。

### iPhone 打不开网页怎么办

确认电脑和 iPhone 在同一个 Wi-Fi；确认电脑防火墙允许 Python 访问网络；确认 Safari 地址使用的是电脑局域网 IP，例如 `http://192.168.1.23:8000`，不要使用 `127.0.0.1`。

### 端口被占用怎么办

默认端口是 `8000`。如果端口被占用，可在 `app.py` 最后一行把 `port=8000` 改为其他端口，例如 `8001`。

### 模型文件缺失怎么办

重新运行训练命令生成模型：

```bash
python train.py --data_dir "D:\BMW" --epochs 30 --batch_size 16 --img_size 224
```

确认 `checkpoints/best_model.pth` 和 `checkpoints/class_to_idx.json` 都存在后再启动网页。
