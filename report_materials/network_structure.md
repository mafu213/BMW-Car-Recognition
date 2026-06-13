# 网络结构说明

本项目实际保存的最佳模型骨干网络为 `mobilenet_v3_small`，整体流程可参考 `results/model_structure.png`。

## 输入层
输入为 RGB 车辆图像，统一处理为 224×224×3，并使用 ImageNet 均值和标准差进行标准化。

## 特征提取层
CNN Backbone 负责从图片中提取车身轮廓、车灯、前脸、侧面线条、SUV/轿车比例等层级特征。MobileNetV3-Small 具有轻量化特点，适合课程项目和现场网页演示；在 GPU 或 CPU 上推理都比较稳定。

## 分类层
骨干网络输出的高维特征经过 Global Average Pooling 聚合为空间特征向量，然后输入全连接层。全连接层输出 4 个类别的 logits，对应四类 BMW 车型。

## Softmax 输出
Softmax 将 logits 转换为概率分布，网页端展示最高概率类别和 Top-4 概率条形图。最终类别取概率最大的车型。
