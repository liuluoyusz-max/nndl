# 文件说明

## 根目录

| 路径 | 内容 |
|---|---|
| `pacs_dataset.py` | 按清单读取 PACS 图片和标签的数据集类 |
| `example_load_data.py` | 检查清单并试读一个 batch 的示例 |
| `adain.py` | AdaIN 模型。对外函数是 `stylize(content, style, alpha)` |
| `adain_paths.py` | 数据路径、Photo 训练清单和输出文件名的共用读取 |
| `style_pool.py` | 读取 `style_pool` 里的风格图，并按种子分配给内容图 |
| `generate_adain_samples.py` | 生成少量风格化样例和对比图 |
| `build_adain_aug.py` | 批量生成 Photo 训练集的 AdaIN 增强集。全量结果尚未生成 |
| `README.md` | 原有的 PACS 划分和读取说明 |
| `handoff_member2.md` | 本说明 |

## `project_splits/`

实验使用的固定划分。每行是「相对路径 + 标签」，标签为 1–7。

| 文件 | 内容 |
|---|---|
| `photo_train.txt` | Photo 训练集，1,349 张 |
| `photo_val.txt` | Photo 验证集，150 张 |
| `photo_test.txt` | Photo 测试集，171 张 |
| `art_painting_test.txt` | Art Painting 测试集，2,048 张 |
| `cartoon_test.txt` | Cartoon 测试集，2,344 张 |
| `sketch_test.txt` | Sketch 测试集，3,928 张 |
| `protocol.json` | 上述划分的数量、各类统计和校验值 |

## `data/`

| 路径 | 内容 |
|---|---|
| `data/images/photo/` | PACS 照片，其下按 dog、elephant、giraffe、guitar、horse、house、person 分文件夹 |
| `data/images/art_painting/` | PACS 油画 |
| `data/images/cartoon/` | PACS 卡通 |
| `data/images/sketch/` | PACS 素描 |
| `data/splits/` | 下载压缩包自带的原始划分，实验不使用 |
| `data/aug_adain/samples/` | 8 组 AdaIN 样例。普通 jpg 是风格化结果，`__compare.jpg` 从左到右是原图、风格图、结果 |
| `data/aug_adain/samples/samples.txt` | 每行对应一张样例：内容图路径、风格图文件名、输出文件名、alpha |

## `style_pool/`

10 张外部画作，用作 AdaIN 的风格图。没有使用 PACS 里的图片。

| 文件 | 内容 |
|---|---|
| `asheville.jpg` | 风格图 |
| `brushstrokes.jpg` | 风格图 |
| `contrast_of_forms.jpg` | 风格图 |
| `flower_of_life.jpg` | 风格图 |
| `la_muse.jpg` | 风格图 |
| `mondrian.jpg` | 风格图 |
| `picasso_self_portrait.jpg` | 风格图 |
| `scene_de_rue.jpg` | 风格图 |
| `the_resevoir_at_poitiers.jpg` | 风格图 |
| `woman_with_hat_matisse.jpg` | 风格图 |
| `SOURCES.txt` | 这 10 张图的来源和排除规则 |

## `models/`

| 文件 | 内容 |
|---|---|
| `vgg_normalised.pth` | AdaIN 编码器权重 |
| `decoder.pth` | AdaIN 解码器权重 |

## `figures/`

| 文件 | 内容 |
|---|---|
| `adain_flowchart.svg` | AdaIN 风格增强流程图 |
