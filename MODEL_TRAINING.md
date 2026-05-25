# 单视频模型框架说明

当前阶段目标是只针对样例视频 `b7763a0682d156294de373ad97e2c544.mp4` 跑通深度学习模型训练和推理链路。

完整链路：

```text
样例视频
  -> 抽取关键帧
  -> 人工标注 YOLO 标签
  -> 训练目标检测模型
  -> 使用模型推理关键帧
  -> 输出计数结果和标注图
```

## 重要说明

当前路线不再使用旧规则基线生成伪标签。训练数据应来自人工标注。

这意味着：

1. 抽帧脚本只负责准备图片和 YOLO 目录结构。
2. 标签需要人工使用标注工具生成。
3. 模型训练依赖人工标签质量。
4. 单视频训练只能验证工程链路，不能证明泛化能力。

## 1. 安装依赖

推荐使用项目虚拟环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

如果当前系统提示 `ensurepip is not available`，需要先安装 Python venv 支持，例如 Debian/Ubuntu 环境中通常需要：

```bash
sudo apt install python3.12-venv
```

其中 YOLO 训练依赖 `ultralytics`。如果本机没有 GPU，也可以先用 CPU 跑通流程，但训练会慢。

## 2. 抽取关键帧

```bash
python3 scripts/extract_frames_for_labeling.py
```

默认输入：

```text
b7763a0682d156294de373ad97e2c544.mp4
```

默认输出：

```text
datasets/end_cap_single_video/
  data.yaml
  dataset_summary.json
  images/train/
  images/val/
  labels/train/
  labels/val/
```

可调参数示例：

```bash
python3 scripts/extract_frames_for_labeling.py \
  --frame-stride 12 \
  --val-ratio 0.2
```

## 3. 人工标注

使用 LabelImg、CVAT、Roboflow、Label Studio 或其他支持 YOLO 格式的标注工具。

标注类别：

```text
part_end_cap
```

标注完成后，每张图片应有对应的 `.txt` 标签文件：

```text
images/train/frame_00012.jpg
labels/train/frame_00012.txt
```

YOLO 标签格式：

```text
class_id center_x center_y width height
```

坐标为归一化比例，类别 `part_end_cap` 的 `class_id` 为 `0`。

## 4. 训练 YOLO 模型

```bash
python3 scripts/train_single_video_yolo.py --device 0
```

默认使用：

```text
data: datasets/end_cap_single_video/data.yaml
model: yolov8n.pt
epochs: 80
imgsz: 640
batch: 8
```

训练完成后，默认权重路径为：

```text
model_runs/end_cap_single_video/weights/best.pt
```

如果需要指定 CPU：

```bash
python3 scripts/train_single_video_yolo.py --device cpu
```

## 5. 使用训练模型计数

```bash
python3 scripts/count_with_yolo_model.py --device 0
```

默认输入：

```text
video: b7763a0682d156294de373ad97e2c544.mp4
weights: model_runs/end_cap_single_video/weights/best.pt
```

默认输出：

```text
model_count_output/
  count_result.json
  keyframes/
  annotated/
```

输出中的 `estimated_count` 使用强视角融合策略：先统计每个关键帧中的 YOLO 检测数量，再取强视角关键帧的中位数作为估算数量。

## 6. 推荐验证方式

建议按以下顺序验证：

1. 抽取样例视频关键帧。
2. 人工标注端面检测框。
3. 训练 YOLO 模型。
4. 使用训练好的模型对同一个视频计数。
5. 检查每帧检测数量曲线和标注图中的误检、漏检。

## 7. 下一步

当单视频框架跑通后，下一步应进入数据闭环：

1. 增加更多同类视频。
2. 为每条视频记录人工真实数量。
3. 扩充训练集、验证集和测试集。
4. 评估不同视频上的计数误差。
5. 建立人工复核和修正数据回流流程。
