# PACS 数据与 AdaIN 风格增强

本仓库提供统一的 PACS 数据划分和读取接口，并提供 Photo 训练集的 AdaIN 风格增强。实验只在 Photo 上训练，再到 Art Painting、Cartoon、Sketch 上测试。所有成员使用 `project_splits` 里的固定清单，不另行随机划分。

类别顺序为 dog、elephant、giraffe、guitar、horse、house、person。清单里的标签是 1 到 7，`PACSDataset` 读取时减 1，返回 0 到 6。调用者不需要再减一次。

## 目录

```text
├── pacs_dataset.py              按清单读取图片和标签
├── example_load_data.py         检查清单或试读一个 batch
├── adain.py                     AdaIN 模型，接口 stylize(content, style, alpha)
├── adain_paths.py               路径、清单和四档强度
├── style_pool.py                读取外部风格图并按种子分配
├── generate_adain_samples.py    生成预览图
├── build_adain_aug.py           生成全量增强集，全量结果尚未生成
├── project_splits/              实验使用的固定划分
├── style_pool/                  10 张外部风格图
├── models/                      AdaIN 权重
├── figures/adain_flowchart.svg  风格增强流程图
└── data/
    ├── images/                  原始 PACS 图片
    ├── splits/                  下载包自带的官方清单，实验不使用
    └── aug_adain/samples/       已生成的预览
```

## 数据划分

| 清单 | 数量 | 用途 |
|---|---:|---|
| `photo_train.txt` | 1,349 | 训练分类器，并只对这一份做风格增强 |
| `photo_val.txt` | 150 | 选择参数和模型 |
| `photo_test.txt` | 171 | 域内准确率 |
| `art_painting_test.txt` | 2,048 | 跨域准确率 |
| `cartoon_test.txt` | 2,344 | 跨域准确率 |
| `sketch_test.txt` | 3,928 | 跨域准确率 |

Photo 测试集来自原始 Photo crossval 的 171 张。验证集从原始 Photo 训练集中按类别抽取约 10%，随机种子为 42。验证集、Photo 测试集和三个跨域测试集不参与风格化，也不用来选择生成参数。

坏图 `sketch/dog/n02103406_4068-1.png` 被排除，原始文件仍保留。

读取方式：

```python
from pacs_dataset import PACSDataset

dataset = PACSDataset(
    data_root=data_root,
    split_file=train_split,
    transform=transform,
)
image, label = dataset[0]
```

`data_root` 是包含 `images` 的目录。`split_file` 是清单路径。`transform` 是图片预处理，不是文件路径。清单中的图片路径相对于 `data/images`。

检查清单可以运行：

```bash
python example_load_data.py --check-files
```

试读一个 batch：

```bash
python example_load_data.py
```

需要安装 PyTorch、torchvision 和 Pillow。分类器训练时的 Resize、Normalize 由负责训练的成员统一确定。

## AdaIN 风格增强

风格图是 `style_pool/` 中的 10 张外部画作，不使用 PACS 里的图片，以免 Art、Cartoon、Sketch 的外观进入训练集。来源说明见 `style_pool/SOURCES.txt`。

配对由种子 42 固定。每张 Photo 训练图对应 1 张风格图。调用接口是：

```python
from adain import stylize

image = stylize(content, style, alpha)
```

`alpha` 取 0 到 1。主比较使用 1.0。代码里的四档强度是 0.25、0.5、0.75、1.0。需要按位置调节强度时，使用 `encode`、`mix`、`decode`。`mix` 可以接收一个数，也可以接收一张强度图。

预览已生成。前 8 张训练照片、四档强度，共 32 张，在 `data/aug_adain/samples/`。总览图是同目录下的 `alpha_overview.jpg`。这 32 张不进入训练集。

全量 1,349 张尚未生成。生成命令：

```bash
python build_adain_aug.py --alpha 1
```

这一命令只生成主比较用的 α = 1，输出在 `data/aug_adain/`。不传 `--alpha` 时，四档分别写到 `alpha_0.25`、`alpha_0.50`、`alpha_0.75`、`alpha_1.00`。

主比较的训练集是原始 1,349 张加上 α = 1 的 1,349 张，共 2,698 张。标签仍沿用内容图。

## 权重文件

`models/` 中有两份现成权重，来自公开的 PyTorch AdaIN 实现，不是本项目训练的。

| 文件 | 大小 | 作用 | GitHub |
|---|---:|---|---|
| `decoder.pth` | 约 14 MB | 把特征还原成图片 | 可以上传 |
| `vgg_normalised.pth` | 约 80 MB | VGG-19 编码器，计算内容特征和风格特征 | 网页无法上传 |

`vgg_normalised.pth` 约 80 MB，超过 GitHub 网页上传的 25 MB 上限，所以这个文件无法通过网页上传。它低于 Git 命令推送的 100 MB 上限，但超过 50 MB 时 GitHub 会给出大文件警告。

仓库里可以不提交这两份权重。`adain.py` 发现 `models/` 中缺少文件时，会从下面的地址自动下载：

- https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth
- https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth

克隆仓库的成员第一次运行风格化时，也会自动补上这两个文件。

## 数据来源

- PACS 作者项目页面：https://dali-dl.github.io/project_iccv2017.html
- Dassl.pytorch 的 PACS 说明：https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#pacs
- PACS 数据下载：https://drive.google.com/file/d/1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE/view
- AdaIN 权重与实现：https://github.com/naoto0804/pytorch-AdaIN
