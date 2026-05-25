# Ubuntu 运行与训练环境搭建

本文档说明如何在 Ubuntu 系统上运行本项目，并使用 NVIDIA GeForce RTX 4060 Ti 16GB 显存版本进行单视频 YOLO 模型训练。

当前项目目标是先围绕样例视频 `b7763a0682d156294de373ad97e2c544.mp4` 跑通模型框架：

```text
视频抽帧
  -> 人工标注 YOLO 数据集
  -> 训练端面检测模型
  -> 使用训练模型计数
  -> Web demo 上传与展示结果
```

## 1. 系统前提

推荐环境：

```text
Ubuntu: 22.04 或 24.04
GPU: NVIDIA GeForce RTX 4060 Ti 16GB
Python: 3.10 到 3.12
显卡驱动: 能正常运行 nvidia-smi
```

先确认系统和显卡：

```bash
lsb_release -a
nvidia-smi
```

如果 `nvidia-smi` 能看到 RTX 4060 Ti、显存、驱动版本，说明 NVIDIA 驱动已经可用。

如果 `nvidia-smi` 不存在或无法识别显卡，需要先安装 NVIDIA 驱动。

## 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  git \
  ffmpeg \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1
```

如果你的 Ubuntu 使用 Python 3.12，并且创建虚拟环境时报错 `ensurepip is not available`，可以额外安装：

```bash
sudo apt install -y python3.12-venv
```

如果你的 Ubuntu 使用 Python 3.10，则对应为：

```bash
sudo apt install -y python3.10-venv
```

## 3. 获取项目代码

进入项目目录：

```bash
cd /path/to/Test_12
```

确认项目根目录下能看到这些文件：

```bash
ls
```

应包含：

```text
counting/
server.py
requirements.txt
MODEL_TRAINING.md
PROJECT_GOAL.md
TECHNICAL_ROADMAP.md
b7763a0682d156294de373ad97e2c544.mp4
```

## 4. 创建 Python 虚拟环境

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

确认当前 Python 来自 `.venv`：

```bash
which python
python --version
```

路径应类似：

```text
/path/to/Test_12/.venv/bin/python
```

## 5. 安装 PyTorch GPU 版本

RTX 4060 Ti 可以使用 PyTorch 的 CUDA 版本 wheel。建议先安装 PyTorch，再安装本项目其他依赖，避免 `ultralytics` 自动拉取不合适的 Torch 版本。

优先参考 PyTorch 官网的最新安装选择器：

```text
https://pytorch.org/get-started/locally/
```

选择：

```text
PyTorch Build: Stable
OS: Linux
Package: Pip
Language: Python
Compute Platform: CUDA
```

当前推荐可以优先尝试 CUDA 12.8 wheel：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

如果 CUDA 12.8 wheel 与当前驱动不匹配，可以尝试 CUDA 12.6 wheel：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

一般不需要在系统中额外安装完整 CUDA Toolkit；关键是 NVIDIA 驱动正常，且 PyTorch 安装的是 CUDA wheel。

验证 PyTorch 是否能使用 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

期望输出中包含：

```text
True
NVIDIA GeForce RTX 4060 Ti
```

如果 `torch.cuda.is_available()` 是 `False`，先不要继续训练，优先检查：

1. `nvidia-smi` 是否正常。
2. PyTorch 是否安装了 CUDA wheel，而不是 CPU wheel。
3. 驱动版本是否支持所选 CUDA wheel。

## 6. 安装项目依赖

确认虚拟环境仍处于激活状态：

```bash
source .venv/bin/activate
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

验证关键依赖：

```bash
python -c "import cv2, numpy, sklearn, matplotlib; print('basic deps ok')"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 7. 抽取关键帧并准备 YOLO 数据集

抽取关键帧：

```bash
python scripts/extract_frames_for_labeling.py
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

检查数据集摘要：

```bash
cat datasets/end_cap_single_video/data.yaml
python -c "import json; d=json.load(open('datasets/end_cap_single_video/dataset_summary.json')); print(d['sampled_frame_count'], d['train_frame_count'], d['val_frame_count'])"
```

当前样例视频的默认生成结果通常约为：

```text
sampled_frame_count = 42
train_frame_count = 33
val_frame_count = 9
```

注意：抽帧脚本只创建图片和空标签文件。训练前必须人工标注端面检测框，并保存为 YOLO 格式标签。

## 8. 人工标注

使用 LabelImg、CVAT、Roboflow、Label Studio 或其他支持 YOLO 格式的标注工具。

当前类别：

```text
0: part_end_cap
```

每张图片对应一个标签文件：

```text
images/train/frame_00012.jpg
labels/train/frame_00012.txt
```

YOLO 标签格式：

```text
class_id center_x center_y width height
```

坐标为归一化比例。

## 9. 训练单视频 YOLO 模型

使用 GPU 训练：

```bash
python scripts/train_single_video_yolo.py --device 0
```

RTX 4060 Ti 16GB 显存通常可以使用默认参数：

```text
epochs: 80
imgsz: 640
batch: 8
model: yolov8n.pt
```

如果显存占用较低，可以尝试增大 batch：

```bash
python scripts/train_single_video_yolo.py --device 0 --batch 16
```

如果出现显存不足，则降低 batch：

```bash
python scripts/train_single_video_yolo.py --device 0 --batch 4
```

训练输出目录：

```text
model_runs/end_cap_single_video/
  weights/
    best.pt
    last.pt
  results.csv
  results.png
```

训练完成后确认权重存在：

```bash
ls model_runs/end_cap_single_video/weights
```

应看到：

```text
best.pt
last.pt
```

## 10. 使用训练模型计数

训练完成后，使用 YOLO 模型重新对样例视频计数：

```bash
python scripts/count_with_yolo_model.py --device 0
```

默认读取：

```text
model_runs/end_cap_single_video/weights/best.pt
```

默认输出：

```text
model_count_output/
  count_result.json
  keyframes/
  annotated/
```

查看计数摘要：

```bash
python -c "import json; d=json.load(open('model_count_output/count_result.json')); print(d['estimated_count'], d['sampled_frame_count'], d['total_raw_detections'])"
```

## 11. 启动 Web Demo

启动服务：

```bash
python server.py
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

页面支持上传同类绕拍视频，并返回：

1. 估算数量。
2. 视频信息。
3. 每帧检测数量。
4. 标注关键帧。
5. 结果 JSON。

当前 Web demo 默认调用 YOLO 模型计数，默认权重路径为：

```text
model_runs/end_cap_single_video/weights/best.pt
```

如果模型权重不在默认位置，可以设置：

```bash
COUNTING_MODEL_WEIGHTS=/path/to/best.pt python server.py
```

## 12. 推荐完整运行顺序

首次在 Ubuntu + RTX 4060 Ti 上运行，建议按这个顺序：

```bash
cd /path/to/Test_12

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python scripts/extract_frames_for_labeling.py
# 完成人工标注后再训练
python scripts/train_single_video_yolo.py --device 0
python scripts/count_with_yolo_model.py --device 0
python server.py
```

## 13. 常见问题

### 13.1 `ensurepip is not available`

原因：系统缺少 venv 支持。

处理：

```bash
sudo apt install -y python3-venv
```

如果仍失败，按 Python 版本安装：

```bash
sudo apt install -y python3.12-venv
```

### 13.2 `torch.cuda.is_available()` 返回 `False`

优先检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

常见原因：

1. NVIDIA 驱动没有装好。
2. 安装了 CPU 版 PyTorch。
3. 驱动版本太旧，不支持当前 CUDA wheel。

处理建议：

1. 先确认 `nvidia-smi` 正常。
2. 卸载 CPU 版 PyTorch：

```bash
pip uninstall -y torch torchvision torchaudio
```

3. 重新安装 CUDA wheel：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 13.3 训练显存不足

降低 batch：

```bash
python scripts/train_single_video_yolo.py --device 0 --batch 4
```

或降低输入尺寸：

```bash
python scripts/train_single_video_yolo.py --device 0 --imgsz 512
```

### 13.4 下载 `yolov8n.pt` 很慢

首次训练时 Ultralytics 会下载基础模型权重。如果网络较慢，可以手动下载 `yolov8n.pt` 后放到项目根目录，再运行训练：

```bash
python scripts/train_single_video_yolo.py --model yolov8n.pt --device 0
```

### 13.5 单视频训练效果很好但换视频效果差

这是预期现象。当前数据集只来自一个视频，即使标签是人工标注，也不足以覆盖不同光照、角度、背景和遮挡情况。

这一步的目标是验证模型框架，而不是证明泛化能力。后续要提升泛化能力，需要：

1. 增加更多同类视频。
2. 持续补充和修正人工标签。
3. 建立训练集、验证集、测试集。
4. 记录每条视频的人工真实数量。
5. 用真实误差评估模型效果。

## 14. 参考

- PyTorch 官方安装说明：https://pytorch.org/get-started/locally/
- Ultralytics 官方安装说明：https://docs.ultralytics.com/quickstart/
