# 代码说明

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
