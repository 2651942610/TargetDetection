# YOLOv5 目标检测系统 🚀

一个基于 YOLOv5 的现代化目标检测系统，提供直观的图形用户界面，支持图像、视频和实时摄像头检测。

## ✨ 主要特性

- 🖼️ **图像检测**: 支持多种格式的静态图像目标检测
- 🎬 **视频检测**: 批量处理视频文件中的目标检测
- 📹 **实时检测**: 支持摄像头实时目标检测
- 🎨 **现代化界面**: 基于 PyQt5 的美观用户界面
- ⚡ **智能模型选择**: 根据硬件配置自动选择最适合的模型
- 🔧 **多模型支持**: 内置 YOLOv5n/s/m/l/x 五种不同规模的模型
- 💾 **结果保存**: 自动保存检测结果到指定目录

## 🏗️ 项目结构

```
TargetDetection/
├── UI.py                   # 主界面程序
├── detect.py              # 核心检测逻辑
├── models/                # 模型定义
│   ├── common.py          # 通用模型组件
│   ├── experimental.py    # 实验性模型
│   └── yolo.py           # YOLO模型定义
├── utils/                 # 工具函数
│   ├── general.py         # 通用工具函数
│   ├── dataloaders.py     # 数据加载器
│   ├── plots.py           # 绘图工具
│   ├── torch_utils.py     # PyTorch工具
│   └── myutil.py          # 自定义工具
├── resource/              # 资源文件
│   ├── UI.svg             # 界面图标
│   ├── yolov5n.pt         # YOLOv5 nano 模型
│   ├── yolov5s.pt         # YOLOv5 small 模型
│   ├── yolov5m.pt         # YOLOv5 medium 模型
│   ├── yolov5l.pt         # YOLOv5 large 模型
│   └── yolov5x.pt         # YOLOv5 extra large 模型
└── result/                # 检测结果输出目录
    ├── demo.mp4           # 示例视频检测结果
    ├── 1.png              # 示例图像检测结果
    └── 0.mp4              # 摄像头检测结果
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- CUDA 10.2+ (可选，用于 GPU 加速)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd TargetDetection
   ```

2. **安装依赖**
   ```bash
   pip install torch torchvision torchaudio
   pip install PyQt5
   pip install opencv-python
   pip install numpy
   pip install pillow
   pip install pyyaml
   pip install tqdm
   pip install matplotlib
   pip install seaborn
   pip install pandas
   ```

3. **运行程序**
   ```bash
   python UI.py
   ```

### 使用方法

#### 🖼️ 图像检测
1. 启动程序后，默认在"图像检测"选项卡
2. 点击"📁 加载图片"按钮选择要检测的图像
3. 点击"🔍 开始检测"按钮开始检测
4. 检测结果将显示在右侧面板，并自动保存到 `result/` 目录

#### 🎬 视频检测
1. 切换到"视频检测"选项卡
2. 点击"📁 加载视频"按钮选择视频文件
3. 点击"🔍 开始检测"按钮开始处理
4. 处理完成后结果视频将保存到 `result/` 目录

#### 📹 摄像头检测
1. 切换到"摄像头检测"选项卡
2. 点击"📹 开始检测"按钮启动实时检测
3. 实时检测画面将显示在界面中
4. 点击"⏹ 停止检测"按钮停止检测

## ⚙️ 模型配置

系统会根据您的硬件配置自动选择最适合的模型：

| 模型 | 大小 | 速度 | 精度 | 推荐配置 |
|------|------|------|------|----------|
| YOLOv5n | 1.9MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | CPU 或低端 GPU |
| YOLOv5s | 14.1MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 一般 GPU (4GB+) |
| YOLOv5m | 21.2MB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 中端 GPU (6GB+) |
| YOLOv5l | 46.5MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 高端 GPU (8GB+) |
| YOLOv5x | 86.7MB | ⚡ | ⭐⭐⭐⭐⭐ | 专业 GPU (8GB+) |

## 🔧 高级配置

### 自定义检测参数

在 `detect.py` 中可以调整检测参数：

```python
# 置信度阈值 (0-1, 默认 0.25)
conf_thres = 0.25

# IoU 阈值 (0-1, 默认 0.45)  
iou_thres = 0.45

# 最大检测数量 (默认 1000)
max_det = 1000

# 输入图像尺寸 (默认 640)
imgsz = 640
```