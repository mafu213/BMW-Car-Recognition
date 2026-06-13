# BMW 四类车型识别网页系统

本项目使用 PyTorch 训练 BMW 四类车型分类模型，并提供适合课程现场验收的 FastAPI 网页识别系统。当前网页支持三种识别方式：

- 摄像头实时识别区：浏览器直接打开摄像头预览并拍照识别。
- 拍照上传备用区：iPhone Safari 调用相机拍照，拍完后上传识别，现场最推荐。
- 本地图片上传区：从相册或文件中选择已有图片识别。

## 纯前端手机版本

新增 `mobile_web/` 目录用于 GitHub Pages 部署。该版本将 `checkpoints/best_model.pth` 导出为 ONNX，并使用 ONNX Runtime Web 在 iPhone 浏览器本地完成推理，不需要电脑运行 `python app.py`。

本地测试：

```bash
cd mobile_web
python -m http.server 8080
```

然后打开：

```text
http://127.0.0.1:8080
```

部署和 iPhone 验收步骤见：

```text
mobile_web/README_mobile.md
```

## 本次训练结果

- 模型：MobileNetV3-Small，ImageNet 预训练权重
- 最佳验证准确率：68.75%
- 测试集准确率：69.62%
- Macro Precision：69.54%
- Macro Recall：69.70%
- Macro F1-score：69.18%
- 测试集各类别识别率：`28` 90.00%，`29` 58.54%，`32` 60.00%，`37` 70.27%

## 支持类别

- `28`: BMW 1 Series Coupe 2012
- `29`: BMW 3 Series Sedan 2012
- `32`: BMW X5 SUV 2007
- `37`: BMW X3 SUV 2012

## 环境安装

推荐 Python 3.9+。在本机已使用 `D:\Anaconda\envs\DL` 环境完成训练、测试和网页验证：

```bash
conda activate DL
pip install -r requirements.txt
```

如果是新电脑，可先安装 PyTorch，再安装其余依赖：

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## 数据集路径说明

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

本机数据实际根目录会自动解析为 `D:\BMW\BMW`，类别以文件夹名为准。

## 训练命令

```bash
python train.py --data_dir "D:\BMW" --epochs 30 --batch_size 16 --img_size 224
```

训练会保存：

- `checkpoints/best_model.pth`
- `checkpoints/class_to_idx.json`
- `results/dataset_stats.json`
- `results/dataset_stats.csv`
- `results/accuracy_curve.png`
- `results/loss_curve.png`

训练结束后会自动运行 `evaluate.py`。

## 测试命令

```bash
python evaluate.py --data_dir "D:\BMW" --model_path checkpoints/best_model.pth
```

测试结果保存到：

- `results/confusion_matrix.png`
- `results/classification_report.txt`
- `results/final_metrics.json`
- `results/sample_predictions.png`
- `results/model_structure.png`

## 单张预测命令

```bash
python predict.py --image_path "某张图片路径" --model_path checkpoints/best_model.pth
```

## 启动网页

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

## iPhone 现场验收推荐步骤

现场最稳方案是使用“拍照上传备用区”，因为它不依赖浏览器实时摄像头权限。

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

5. 使用页面中的“拍照上传备用区”。
6. 点击“拍照上传识别”，调用 iPhone 后置摄像头拍老师给的纸质或屏幕图片。
7. 拍照完成后回到网页，点击“识别照片”。
8. 页面显示预测车型、置信度和 Top-4 概率条形图。

## 实时摄像头说明

页面中的“摄像头实时识别区”使用：

```javascript
navigator.mediaDevices.getUserMedia({
  video: { facingMode: { ideal: "environment" } },
  audio: false
})
```

注意：浏览器通常要求 HTTPS 或 localhost 这类安全环境才能开放实时摄像头权限。iPhone 直接访问 `http://电脑局域网IP:8000` 时，实时摄像头预览可能不可用。

如果实时摄像头不可用，页面会显示：

```text
当前浏览器无法直接打开实时摄像头，请使用下方拍照上传方式。
```

课程验收建议优先使用“拍照上传备用区”。

如果必须使用实时摄像头预览，可选方案：

- 使用 ngrok 或 Cloudflare Tunnel 暴露 HTTPS 地址。
- 或在本地配置 HTTPS 证书后使用 HTTPS 访问。

## 现场测试检查清单

可运行：

```bash
python check_demo.py
```

也可以手动检查：

1. 电脑运行 `python app.py`。
2. 电脑浏览器打开 `http://127.0.0.1:8000`。
3. iPhone 打开 `http://电脑局域网IP:8000`。
4. 测试普通上传图片是否能识别。
5. 测试拍照上传是否能调用 iPhone 后置摄像头。
6. 测试拍照后是否能返回预测类别。
7. 测试 Top-4 概率是否显示。
8. 如果实时摄像头不可用，确认页面有友好提示并可使用拍照上传备用区。

## /predict 返回格式

`/predict` 支持普通文件上传、canvas 生成的 JPEG Blob 和 iPhone 拍照上传文件。推荐字段如下：

```json
{
  "pred_label": "28 - BMW 1 Series Coupe 2012",
  "confidence": 0.95,
  "topk": [
    {"label": "28 - BMW 1 Series Coupe 2012", "prob": 0.95},
    {"label": "29 - BMW 3 Series Sedan 2012", "prob": 0.03},
    {"label": "32 - BMW X5 SUV 2007", "prob": 0.01},
    {"label": "37 - BMW X3 SUV 2012", "prob": 0.01}
  ]
}
```

后端同时保留旧字段 `predicted_class`、`display_name`、`top4`，兼容已有脚本。

## GitHub 上传步骤

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

## 项目结果展示位置

- 数据统计：`results/dataset_stats.json`、`results/dataset_stats.csv`
- 训练曲线：`results/accuracy_curve.png`、`results/loss_curve.png`
- 混淆矩阵：`results/confusion_matrix.png`
- 分类报告：`results/classification_report.txt`
- 最终指标：`results/final_metrics.json`
- 样例预测：`results/sample_predictions.png`
- 网络结构示意图：`results/model_structure.png`
- 中文报告素材：`report_materials/`

## 常见问题

### 没有 GPU 怎么办

脚本会自动检测 CUDA。如果没有 GPU，会使用 CPU 继续训练和推理，只是速度会慢一些。

### iPhone 打不开网页怎么办

确认电脑和 iPhone 在同一个 Wi-Fi；确认电脑防火墙允许 Python 访问网络；确认 Safari 地址使用的是电脑局域网 IP，例如 `http://192.168.1.23:8000`，不要使用 `127.0.0.1`。

### iPhone 打不开实时摄像头怎么办

这是 HTTP 局域网访问的常见限制。现场直接使用“拍照上传备用区”，它通过 `<input type="file" accept="image/*" capture="environment">` 调用 iPhone 后置摄像头，通常比实时摄像头更稳定。

### 端口被占用怎么办

默认端口是 `8000`。如果端口被占用，可在 `app.py` 最后一行把 `port=8000` 改为其他端口，例如 `8001`。

### 模型文件缺失怎么办

重新运行训练命令生成模型：

```bash
python train.py --data_dir "D:\BMW" --epochs 30 --batch_size 16 --img_size 224
```

确认 `checkpoints/best_model.pth` 和 `checkpoints/class_to_idx.json` 都存在后再启动网页。
