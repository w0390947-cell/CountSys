# 单视频模型框架说明

当前阶段目标是先把模型训练和推理框架跑通，只针对样例视频 `b7763a0682d156294de373ad97e2c544.mp4` 做验证。

这不是为了证明模型已经具备泛化能力，而是为了验证以下链路：

```text
样例视频
  -> 抽帧
  -> 生成 YOLO 数据集
  -> 训练目标检测模型
  -> 使用模型推理关键帧
  -> 输出计数结果和标注图
```

## 重要说明

当前数据集标签由现有规则检测器自动生成，属于伪标签。

这意味着：

1. 模型会学习当前规则检测器的检测习惯。
2. 训练结果适合验证工程流程。
3. 单视频训练结果不能代表真实泛化能力。
4. 后续仍然需要人工标注数据替换伪标签。

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

如果已经处在可写的 Python 环境中，也可以直接安装：

```bash
python3 -m pip install -r requirements.txt
```

其中 YOLO 训练依赖 `ultralytics`。如果本机没有 GPU，也可以先用 CPU 跑通流程，但训练会慢。

## 2. 生成单视频 YOLO 数据集

```bash
python3 scripts/prepare_single_video_yolo_dataset.py
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
  images/
    train/
    val/
  labels/
    train/
    val/
  previews/
```

可调参数示例：

```bash
python3 scripts/prepare_single_video_yolo_dataset.py \
  --frame-stride 12 \
  --val-ratio 0.2 \
  --bbox-scale 2.25
```

## 3. 训练 YOLO 模型

```bash
python3 scripts/train_single_video_yolo.py
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

如果需要指定 GPU：

```bash
python3 scripts/train_single_video_yolo.py --device 0
```

## 4. 使用训练模型计数

```bash
python3 scripts/count_with_yolo_model.py
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

输出中的 `estimated_count` 使用和当前 demo 类似的强视角融合策略：先统计每个关键帧中的 YOLO 检测数量，再取强视角关键帧的中位数作为估算数量。

## 5. 推荐验证方式

建议按以下顺序验证：

1. 先运行现有规则 demo，记录 `estimated_count`。
2. 生成单视频 YOLO 数据集。
3. 检查 `datasets/end_cap_single_video/previews/` 中的伪标签质量。
4. 训练 YOLO 模型。
5. 使用训练好的模型对同一个视频计数。
6. 对比两个结果：
   - 规则检测计数
   - YOLO 模型计数
   - 每帧检测数量曲线
   - 标注图中的误检和漏检

## 6. 下一步

当单视频框架跑通后，下一步应进入人工标注闭环：

1. 抽取关键帧。
2. 人工修正伪标签。
3. 使用修正后的标签重新训练模型。
4. 再增加更多同类视频，测试泛化能力。
