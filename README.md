# Material Counting Prototype

这个项目用于搭建“物资绕拍视频自动计数系统”的深度学习模型路线。

当前阶段先围绕样例视频 `b7763a0682d156294de373ad97e2c544.mp4` 跑通完整工程链路：

```text
抽取关键帧
  -> 人工标注 YOLO 数据
  -> 训练端面检测模型
  -> 使用模型推理计数
  -> Web demo 上传视频并展示结果
```

## 模型路线

本项目不再保留旧规则基线计数流程。计数结果应来自训练好的深度学习检测模型。

当前默认检测类别：

```text
part_end_cap
```

也就是圆柱零件端面。

## 数据准备

从样例视频抽取关键帧，并生成 YOLO 数据集目录骨架：

```bash
python3 scripts/extract_frames_for_labeling.py
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

抽帧后，需要人工标注 `images/train/` 和 `images/val/` 中的图片，并将 YOLO 格式标签保存到对应的 `labels/train/` 和 `labels/val/` 目录。

## 训练模型

```bash
python3 scripts/train_single_video_yolo.py --device 0
```

默认训练输出：

```text
model_runs/end_cap_single_video/weights/best.pt
```

## 模型计数

```bash
python3 scripts/count_with_yolo_model.py --device 0
```

默认输出：

```text
model_count_output/
  count_result.json
  keyframes/
  annotated/
```

## Web Demo

启动本地演示服务：

```bash
python3 server.py
```

然后在浏览器打开：

```text
http://127.0.0.1:8004/
```

页面支持上传同类绕拍视频。后端会调用训练好的 YOLO 模型，返回估算数量、视频信息、每帧检测数量、标注关键帧和结果 JSON。

默认模型权重路径：

```text
model_runs/end_cap_single_video/weights/best.pt
```

如果需要指定其他模型权重，可以设置环境变量：

```bash
COUNTING_MODEL_WEIGHTS=/path/to/best.pt python3 server.py
```

## 详细文档

- 项目目标：`PROJECT_GOAL.md`
- 技术路线：`TECHNICAL_ROADMAP.md`
- 模型训练：`MODEL_TRAINING.md`
- Ubuntu 环境：`UBUNTU_SETUP.md`
