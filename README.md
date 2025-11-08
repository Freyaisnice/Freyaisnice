# 🐕 Dog Stool Detection Model (狗狗大便检测模型)

一个基于深度学习的狗狗大便分析系统，可以自动识别和分类大便的类型、状态、颜色，并评估健康状况。

## 📋 功能特点

- **多任务学习**: 同时识别四个维度的特征
  - 🔍 **大便类型** (Stool Type): 正常、软便、腹泻、便秘、带粘液、带血、含未消化食物
  - 💧 **大便状态** (Consistency): 硬/干燥、成形、软但成形、不成形、液体状
  - 🎨 **大便颜色** (Color): 健康棕色、深棕色、黄色、绿色、黑色、红色、白色/灰色、橙色
  - 💚 **健康评估** (Health): 健康、需观察、需注意、需就医

- **强大的模型架构**: 支持多种预训练backbone
  - EfficientNet (B0, B1)
  - ResNet50
  - MobileNet V3

- **完整的训练流程**:
  - 数据增强 (Albumentations)
  - 学习率调度
  - 早停机制
  - TensorBoard可视化

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone <repository-url>
cd Freyaisnice

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

#### 数据集结构

```
data/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels.json
├── val/
│   ├── images/
│   └── labels.json
└── test/
    ├── images/
    └── labels.json
```

#### 标签格式 (labels.json)

```json
{
  "img001.jpg": {
    "stool_type": "normal",
    "consistency": "formed",
    "color": "brown",
    "health": "healthy"
  },
  "img002.jpg": {
    "stool_type": "diarrhea",
    "consistency": "liquid",
    "color": "yellow",
    "health": "concerning"
  }
}
```

#### 创建示例数据集

```bash
python -c "from src.dataset import create_sample_dataset; create_sample_dataset('data/train')"
python -c "from src.dataset import create_sample_dataset; create_sample_dataset('data/val')"
```

### 3. 训练模型

```bash
# 使用默认配置训练
python src/train.py

# 使用自定义配置
python src/train.py --config your_config.yaml

# 指定设备
python src/train.py --device cuda
```

### 4. 推理预测

#### 单张图片预测

```bash
python src/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/image.jpg
```

#### 批量预测

```bash
python src/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image-dir path/to/images/ \
  --output results.json
```

#### Python API

```python
from src.inference import StoolDetector

# 初始化检测器
detector = StoolDetector(
    config_path='config.yaml',
    checkpoint_path='checkpoints/best_model.pth'
)

# 预测单张图片
results = detector.analyze_image('image.jpg')

# 批量预测
batch_results = detector.batch_predict('images_dir/', 'results.json')
```

## 📊 模型架构

```
Input Image (224x224x3)
        ↓
[Backbone CNN]
(EfficientNet/ResNet/MobileNet)
        ↓
  Feature Vector
        ↓
    ┌───┴───┬───────┬───────┐
    ↓       ↓       ↓       ↓
[Type]  [State] [Color] [Health]
  Head    Head    Head    Head
    ↓       ↓       ↓       ↓
  Output  Output  Output  Output
```

## ⚙️ 配置说明

主要配置参数在 `config.yaml` 中：

```yaml
# 数据配置
data:
  image_size: [224, 224]
  batch_size: 32

# 模型配置
model:
  backbone: "efficientnet_b0"
  pretrained: true
  dropout: 0.3

# 训练配置
training:
  epochs: 100
  learning_rate: 0.001
  scheduler: "cosine"
  early_stopping_patience: 10
```

## 📈 训练监控

使用 TensorBoard 监控训练过程：

```bash
tensorboard --logdir logs/
```

## 🏥 健康指标说明

| 健康等级 | 说明 | 建议 |
|---------|------|------|
| 健康 (Healthy) | 大便正常，无异常 | 保持当前饮食和护理 |
| 需观察 (Monitor) | 轻微异常 | 观察1-2天，注意变化 |
| 需注意 (Concerning) | 明显异常 | 建议调整饮食，持续观察 |
| 需就医 (Urgent) | 严重异常 | 建议立即咨询兽医 |

## 📁 项目结构

```
Freyaisnice/
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖列表
├── README.md               # 项目说明
├── src/                    # 源代码
│   ├── __init__.py
│   ├── model.py           # 模型定义
│   ├── dataset.py         # 数据集处理
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   └── utils.py           # 工具函数
├── data/                   # 数据目录
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/           # 模型检查点
├── logs/                  # 训练日志
└── notebooks/            # Jupyter notebooks
```

## 🔬 技术栈

- **深度学习框架**: PyTorch 2.0+
- **计算机视觉**: OpenCV, Albumentations
- **数据处理**: NumPy, Pandas
- **可视化**: TensorBoard, Matplotlib
- **模型导出**: ONNX

## 📝 使用示例

### 训练自定义模型

```python
import yaml
from src.train import Trainer

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 修改配置
config['model']['backbone'] = 'resnet50'
config['training']['epochs'] = 50

# 训练
trainer = Trainer(config)
trainer.train()
```

### 模型评估

```python
from src.inference import StoolDetector

detector = StoolDetector('config.yaml', 'checkpoints/best_model.pth')

# 单张图片详细分析
analysis = detector.analyze_image('test_image.jpg', verbose=True)

# 输出示例:
# ============================================================
# Analysis Results: test_image.jpg
# ============================================================
#
# 大便类型 (stool_type):
#   预测结果: normal
#   置信度: 95.32%
#   Top 3 可能性:
#     1. normal: 95.32%
#     2. soft: 3.45%
#     3. constipation: 0.89%
```

## ⚠️ 注意事项

1. **数据质量**: 确保图片质量良好，光照充足
2. **标注准确性**: 标签应由专业人员（兽医）审核
3. **模型限制**: 本模型仅供参考，不能替代专业兽医诊断
4. **隐私保护**: 请勿上传包含个人隐私的图片

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📧 联系方式

如有问题或建议，请提交 Issue。

## 🙏 致谢

- PyTorch团队
- Albumentations库开发者
- 所有贡献者

---

**免责声明**: 本项目仅用于研究和教育目的。任何医疗决策应咨询专业兽医。
