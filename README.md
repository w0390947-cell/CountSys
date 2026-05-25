# Multi-view Counting Demo

这个 demo 针对 `b7763a0682d156294de373ad97e2c544.mp4`，用于粗略统计“竹竿包裹的圆柱形零件”数量。

## 运行

```bash
python3 count_cylinder_parts_demo.py
```

默认输出到 `count_demo_output/`：

- `count_result.json`：计数结果、每帧检测数、相机运动估计质量、检测明细。
- `annotated/`：带检测圈的关键帧。
- `keyframes/`：抽取的关键帧。
- `projected_clusters.png`：跨帧投影后的聚类可视化。

## 技术流程

1. 每隔 `--frame-stride` 帧抽一个关键帧。
2. 根据竹竿颜色生成前景物资区域，默认只保留样例视频里的主前景堆。
3. 在前景区域内检测银灰色圆柱端面。
4. 使用 ORB 特征和单应矩阵估计相邻关键帧之间的相机运动。
5. 把不同帧的端面检测点投影到同一参考坐标系。
6. 使用 DBSCAN 生成全局投影聚类诊断图。
7. 主计数采用多视角关键帧中的强视角鲁棒融合，避免把绕拍视频里的三维视差错误压平成一个平面后产生过度合并。

## 重要说明

这是一个工程 demo，不是生产级盘点系统。它的目的是把“多视角重建 + 目标去重”的主流程跑通。当前检测器是针对样例视频调过的规则检测器，正式系统应替换为训练好的 YOLO/SAM/实例分割模型；当前相机运动是单应近似，正式系统应替换为 COLMAP、SLAM 或深度/三维重建。

注意：绕拍物体不是平面，单应矩阵只能作为近似诊断。`count_result.json` 里的 `global_projection_cluster_count_diagnostic` 不建议直接作为最终库存数，主结果看 `estimated_count`。

如果要把背景中的同类堆垛也纳入统计，可以运行：

```bash
python3 count_cylinder_parts_demo.py --include-background-piles
```

如果结果过分合并或重复，可以调整：

```bash
python3 count_cylinder_parts_demo.py --cluster-eps 60
```

如果石子误检较多，可以调严端面平滑度：

```bash
python3 count_cylinder_parts_demo.py --max-center-std 38
```

## Web demo

启动本地演示服务：

```bash
python3 server.py
```

然后在浏览器打开：

```text
http://127.0.0.1:8000/
```

页面支持上传同类绕拍视频，后端会保存到 `demo_runs/` 下的独立运行目录，并复用当前脚本逻辑返回估算数量、视频信息、每帧检测数量、标注关键帧和投影聚类诊断图。

## 单视频模型框架

如果要先用样例视频搭建模型训练和推理链路，请参考：

```text
MODEL_TRAINING.md
```

当前模型框架支持：

1. 从 `b7763a0682d156294de373ad97e2c544.mp4` 抽帧。
2. 使用现有规则检测器生成 YOLO 伪标签。
3. 训练 YOLO 端面检测模型。
4. 使用训练好的模型重新对视频计数。
