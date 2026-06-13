# BMW 手机端纯前端识别版本

本目录用于 GitHub Pages 部署。它不依赖 Python 后端，iPhone 打开 HTTPS 网页后，浏览器会加载 ONNX 模型并使用 ONNX Runtime Web 在本地完成车型识别。

## 文件结构

```text
mobile_web/
├── index.html
├── style.css
├── app.js
├── README_mobile.md
└── model/
    ├── bmw_model.onnx
    ├── class_to_idx.json
    └── idx_to_class.json
```

## 重新导出 ONNX

在项目根目录运行：

```bash
python export_onnx.py --model_path checkpoints/best_model.pth --class_to_idx checkpoints/class_to_idx.json --data_dir "D:\BMW"
```

导出结果：

- `mobile_web/model/bmw_model.onnx`
- `mobile_web/model/class_to_idx.json`
- `mobile_web/model/idx_to_class.json`
- `results/onnx_check.json`

`results/onnx_check.json` 会记录 10 张测试图片的 PyTorch 与 ONNX 一致性检查结果。

## 本地测试

不要直接用 `file://` 打开，因为浏览器通常禁止本地文件加载模型。请在 `mobile_web` 目录运行：

```bash
python -m http.server 8080
```

然后打开：

```text
http://127.0.0.1:8080
```

## GitHub Pages 部署

提交 mobile_web：

```bash
git add mobile_web export_onnx.py results/onnx_check.json
git commit -m "Add mobile browser BMW recognition demo"
git push
```

GitHub 仓库设置：

1. 打开仓库 Settings。
2. 进入 Pages。
3. Source 选择 main branch。
4. 如果 Pages 支持选择 `/mobile_web`，选择该目录。
5. 如果不能直接选择 `mobile_web`，将 `mobile_web` 内容复制到 `docs/`，然后在 Pages 中选择 `/docs`。

复制到 docs 的结构：

```text
docs/
├── index.html
├── style.css
├── app.js
└── model/
```

最终手机访问：

```text
https://你的用户名.github.io/仓库名/
```

## iPhone 现场验收步骤

1. 用 iPhone Safari 打开 GitHub Pages 地址。
2. 等待顶部显示“模型就绪”。
3. 点击“打开摄像头”。
4. 对准老师给的纸质或屏幕车辆图片。
5. 点击“拍照识别”。
6. 查看预测车型、置信度和 Top-4 概率条形图。
7. 如果实时摄像头打不开，使用“拍照上传识别”，它会调用 iPhone 原生相机拍照并在网页内完成识别。

## 注意事项

- 摄像头实时预览要求 HTTPS 或 localhost。GitHub Pages 是 HTTPS，适合 iPhone 摄像头权限。
- 模型约数 MB，首次打开需要等待下载。
- 如果手机推理慢，可压缩 ONNX、改用更小模型、减少输入尺寸，或使用 ONNX Runtime Web 的 WebGL/WebGPU 后端。
