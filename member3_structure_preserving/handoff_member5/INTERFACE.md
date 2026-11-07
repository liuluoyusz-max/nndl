# 结构保持模块接口：一次性交接约定

## 接收成员2的固定接口

接收版本：`nndl-part2（include_part1)` 提交 `416bdc9811d67aae80fd79e8fac1ae0f37d5f339`
中的 `adain.py`，当前仓库已放入 `member2_adain/adain.py`。
本模块只依赖这个固定模型接口，结构图、alpha规则、生成控制、审计和实验脚本由成员3独立实现。
不依赖成员2的 `style_pool.py`、批量脚本或个人数据目录，不需要反复重新对接。

| 成员2 API | 输入 | 返回与约定 |
|---|---|---|
| `AdaINModel(weights_dir=None, device=None, content_size=512, style_size=512)` | 权重/设备/短边尺寸 | 冻结的 normalized VGG-19 relu4_1 编码器与解码器 |
| `prepare_image(image, size)` | 路径、RGB PIL 图或 CHW/BCHW tensor | BCHW float；路径/PIL按短边缩放，tensor不自动缩放；值域 `[0,1]` |
| `encode(image)` | BCHW RGB float | 内容或风格特征，512通道；空间尺寸可不同 |
| `mix(Fc, Fs, alpha)` | 同batch/channel特征，标量或空间alpha | AdaIN统计匹配，再与内容特征插值 |
| `decode(features)` | 混合特征 | CPU BCHW RGB tensor，裁剪到 `[0,1]` |
| `stylize(content, style, alpha)` | 内容、风格及强度 | 单张 RGB PIL 图，可作普通AdaIN对照 |

## 成员3公开接口

```python
from member3_structure_preserving.structure_map import (
    StructureConfig, build_structure_map, spatial_alpha,
)
from member3_structure_preserving.stylizer import StructurePreservingStylizer
```

| API | 返回 |
|---|---|
| `build_structure_map(rgb, config=None)` | `B×1×H×W` 的 M，值域 `[0,1]`；输入为 RGB CHW/BCHW，值域 `[0,1]` |
| `spatial_alpha(M, alpha=0.5, beta=0.7)` | `alpha*(1-beta*M)`；alpha/beta需有限且在 `[0,1]` |
| `StructurePreservingStylizer(model=None, structure_config=None, **model_options)` | 可注入接收后的 AdaIN 模型；默认构造固定模型 |
| `prepare_content(content)` | `PreparedContent(image, features, structure)` |
| `encode_style(style)` | 风格特征；路径输入按路径缓存，减少重复编码 |
| `render(prepared, style_features, alpha=0.5, beta=0.7, with_baseline=False)` | `StylizationResult` |
| `stylize(content, style, alpha=0.5, beta=0.7, with_baseline=False)` | 编码、计算地图、特征混合和解码的一步封装 |

`StylizationResult` 字段：`image`（结构保持结果，PIL）、`content`（实际缩放后的内容图，PIL）、
`structure`（原尺度软M）、`alpha_map`（原尺度强度）、`feature_alpha`（面积平均后的特征尺度强度），
以及可选 `baseline`（相同输入/权重/尺寸下的普通AdaIN，PIL）。

生成器每次处理一张图，最小边至少32像素。原图与地图严格共享预处理。
强度图广播到512个通道，不做额外图像级贴回或后处理。
解码输出尺寸可能向上取整到8的倍数；两种方法保持一致。
alpha=0为网络重建，不能写成“与原图像素完全相同”；beta=0才是普通AdaIN对照。

## 无需再次协商的默认值

- 内容与风格短边均为512；保持纵横比。
- 灰度权重 `(0.299,0.587,0.114)`；预平滑sigma=1，3×3 Sobel，95%分位数归一化。
- 幅值不超过 `1e-6` 归零；分位数近零时改用最大幅值，分母至少 `1e-6`。
- 最大池化3×3、stride=1、保持尺寸；地图平滑sigma=2。
- alpha0=0.5；beta=0.7为初始值；JPEG质量95。
- 模型推理使用冻结权重和 `no_grad`；生成脚本保存代码、权重、输入与输出哈希。

## 调用示例

从仓库根目录运行；只从冻结配对表读取风格：

```python
import json
from member4_training_eval import config
from member3_structure_preserving.stylizer import StructurePreservingStylizer

pool = json.loads(config.STYLE_POOL_MANIFEST.read_text(encoding="utf-8"))
original = "photo/guitar/063_0024.jpg"
style = pool["pairing"][original]
style_path = config.STYLE_POOL_DIR / pool["pool"]["styles"][style]["file"]
generator = StructurePreservingStylizer()
result = generator.stylize(config.IMAGE_ROOT / original, style_path,
                           alpha=0.5, beta=0.7, with_baseline=True)
result.image.save("example_sp.jpg", quality=95)
```

## 给成员4的文件接口

数据根目录与成员4共用 `NNDL_DATA_ROOT`，默认 `<repo>/data`，原图在 `PACS/`。
增强集在 `aug_sp/images/photo/<class>/`。

六列 `aug_sp/manifest_ours.txt` 同步导出为 `augmented/manifest_ours.txt`：

```text
aug_sp/images/photo/dog/056_0022__scene_de_rue.jpg 1 photo/dog/056_0022.jpg scene_de_rue 0.5 0.7
```

列依次为输出相对DATA_ROOT路径、标签1～7、原图相对PACS路径、冻结风格名、alpha0、beta。
读取训练数据时由成员4减1。两列 `aug_sp/photo_train.txt` 的图片路径则相对于 `aug_sp/images/`。
`pairs.txt` 是输出/原图/风格/标签的tab表。
只生成 Photo Training 的1349张原图，每张一张；不得用测试图片作内容或风格源。
训练前两组清单运行成员4的公平性检查。批量CLI对部分预览使用单独目录，
不会发布伪装成全量数据的正式清单。

## 接口验收

已完成12项自动化测试；正式1349张图通过解码、标签、冻结配对和哈希检查；
七张真实beta=0输出与匹配的普通AdaIN文件完全相同。实际记录见
`evidence/delivery_check.json`，这不是分类准确率结果。
