# PACS Data Preparation

本模块为项目提供统一的 Data Split 和 Dataset Interface，供组员读取 PACS 图片及对应 Label。

## 1. 项目结构

```text
项目文件夹/
├── pacs_dataset.py
├── example_load_data.py
├── project_splits/
│   ├── photo_train.txt
│   ├── photo_val.txt
│   ├── photo_test.txt
│   ├── art_painting_test.txt
│   ├── cartoon_test.txt
│   └── sketch_test.txt
└── data/
    ├── images/
    └── splits/
```

`data/images` 保存原始图片，`data/splits` 保存下载时附带的原始清单。

本项目使用 `project_splits` 中的清单开展实验。所有成员使用相同清单，不自行重新随机划分。

## 2. Data Split

| Split | 图片数量 | 用途 |
|---|---:|---|
| Photo Training | 1,349 | 训练 Classifier，生成增强图片 |
| Photo Validation | 150 | 选择参数和 Model |
| Photo Test | 171 | 最终评价 In-domain Accuracy |
| Art Painting Test | 2,048 | 最终评价 Cross-domain Accuracy |
| Cartoon Test | 2,344 | 最终评价 Cross-domain Accuracy |
| Sketch Test | 3,928 | 最终评价 Cross-domain Accuracy |

将原始 Photo crossval 的171张图片保留为 Photo Test。从原始 Photo Training 中按类别抽取约10%作为 Validation，Random Seed 为42。

这是项目自定义的 Photo-only 实验设置。只对 Photo Training 生成增强图片，Test Set 不参与训练或调参。

依据 Dassl PACS 参考实现，排除内容为“ERROR”的图片 `sketch/dog/n02103406_4068-1.png`。原始文件仍保留。

## 3. Label Mapping

| Class | Dataset 返回的 Label |
|---|---:|
| dog | 0 |
| elephant | 1 |
| giraffe | 2 |
| guitar | 3 |
| horse | 4 |
| house | 5 |
| person | 6 |

TXT 清单使用1～7，`PACSDataset` 内部减1，返回0～6。调用者不需要再次减1。

## 4. Dataset Interface

```python
from pacs_dataset import PACSDataset

dataset = PACSDataset(
    data_root=data_root,
    split_file=train_split,
    transform=transform,
)

image, label = dataset[0]
```

- `data_root`：本机数据目录，其下应有 `images` 文件夹。
- `split_file`：项目清单文件的路径。
- `transform`：图片预处理操作，不是文件路径。
- 返回值：处理后的 Image 和整数 Label。

清单中的图片路径相对于 `data/images`，不包含成员个人电脑的绝对路径。

## 5. 环境与运行

读取示例需要安装 PyTorch、torchvision 和 Pillow。

准备好数据并配置路径后，运行：

```bash
python example_load_data.py
```

正式实验中的 Resize、Normalize 和 Training Augmentation 由成员4统一确定，确保三组实验设置可比较。

## 6. 数据来源

- [PACS 作者项目页面](https://dali-dl.github.io/project_iccv2017.html)
- [Dassl.pytorch 数据说明](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#pacs)
- [数据下载](https://drive.google.com/file/d/1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE/view)