# 成员3：结构保持风格增强工作区

已实现结构图、空间风格强度、结构保持 AdaIN、批量生成、对比图、alpha/beta 扫描和
成员4 manifest 对接。所有下列命令通过根目录 `nn.ps1` 使用本机现有 NN 环境。

## 实现入口

| 文件 | 用途 |
|---|---|
| `structure_map.py` | 灰度平滑、Sobel、归一化、软结构图及空间 alpha |
| `stylizer.py` | 在成员2 AdaIN 特征层进行结构保持混合 |
| `protocol.py` | 校验和读取成员4的冻结风格池及配对 |
| `generate.py` | `samples` / `build` / `sweep` 三个命令 |
| `analysis.py` | 标注对比图、固定 Canny EdgeScore、可选 LPIPS |
| `smoke_test.py` | 无需 PACS 的隔离合成图流程检查 |
| `tests/` | 数值、模型、清单、公平性、断点续跑与指标对齐测试 |
| `METHOD.md` | 方法公式、参数、实验约定和局限 |

方法为 `alpha(x,y) = alpha0 * (1 - beta * M(x,y))`。
默认 `alpha0=0.5`，`beta=0.7` 是尚未选优的起始值；`beta=0` 与普通 AdaIN 完全一致。
详见 [METHOD.md](METHOD.md)。完整的2.4、3.5、4.3、5.3稿件、示意图、接口和BibTeX已集中到
[给成员5的一次性交付包](handoff_member5/README.md)。`../report/sections/member3_method.tex`
是3.5稿的兼容入口；总报告仍由成员5统一合并。

## 当前本机交付状态

下面的数据路径描述已完成的本地生成结果。GitHub 中提交代码、七类对比图和
[结果审计文件](results/README.md)；原始 PACS、模型权重和1349张完整增强图沿用仓库约定，
不纳入 Git。其他成员可按本文命令生成相同目录，或另行接收完整增强集。

- PACS：`data/PACS/` 共9991张原图，全部解码成功；项目固定清单使用9990张。
- 正式增强集：`data/aug_sp/images/` 共1349张真实训练图增强，alpha=0.5、beta=0.7。
- 两列清单：`data/aug_sp/photo_train.txt`；六列清单：`data/aug_sp/manifest_ours.txt`。
- 成员4兼容清单：`data/augmented/manifest_ours.txt`，与上述六列清单完全相同。
- `data/aug_sp/delivery_check.json`：1349张增强图逐张解码、原图/输出哈希、标签、冻结配对、
  两列 Dataset 读取及兼容清单一致性均已通过检查。
- 七类对比样例、五组 beta 小样本扫描已保存；真实 beta=0 的7张输出与普通 AdaIN 文件完全一致。
- 成员4的结构指标脚本已计算全量 EdgeScore：均值0.3436，标准差0.0628，有效数1349。
  结果在 `data/aug_sp/structure_metrics.json`；LPIPS 未计算，分类训练尚未进行。
- 12项自动化测试通过。合成测试文件仍隔离在 `data/member3_smoke/`，未混入正式数据。

## 直接运行

在仓库根目录执行。没有 PACS 时也可运行：

```powershell
.\nn.ps1 -m member3_structure_preserving.smoke_test
.\nn.ps1 -m unittest discover -s member3_structure_preserving/tests -v
```

冒烟测试使用带 `SYNTHETIC FIXTURE` 标记的几何图，所有文件隔离在
`data/member3_smoke/`，不代表真实 PACS 实验结果。代码更新后重建旧冒烟结果时加 `--overwrite`。

放入真实 PACS 后，先生成7个类别各一张的对比样例：

```powershell
.\nn.ps1 -m member3_structure_preserving.generate samples --limit 7
```

正式生成1349张增强图（与成员2默认512短边一致）：

```powershell
.\nn.ps1 -m member3_structure_preserving.generate build --alpha 0.5 --beta 0.7 --threads 12
```

正式交付目录为 `data/aug_sp/`，包括增强图片、`photo_train.txt` 两列清单、
`pairs.txt` 配对表和 `manifest_ours.txt` 六列训练清单。
同时将六列清单导出到 `data/augmented/manifest_ours.txt`，成员4可直接读取，图片不重复复制。
断点后重跑相同命令会验证文件并续跑。
本机正式生成使用12个 CPU 线程；续跑时保持 `--threads 12`，其余机器可在首次生成前调整。
`--limit N` 只生成预览，不发布正式清单；`--diagnostics` 增加 M、alpha、对比图；
`--with-adain` 另存匹配的普通 AdaIN 参考图及清单。

扫描 beta；先用7张预览检查，真实完整扫描时去掉 `--limit 7`：

```powershell
.\nn.ps1 -m member3_structure_preserving.generate sweep --limit 7 --betas 0 0.3 0.5 0.7 1
.\nn.ps1 -m member3_structure_preserving.generate sweep --limit 7 --alphas 0.25 0.5 0.75 1 --betas 0 0.7
```

样例在 `data/aug_sp/previews/samples_n7/`，扫描在 `data/aug_sp/sweeps/sweep_n7/`。
每个参数目录有 `generation.json`、manifest 和 `analysis.json`，扫描父目录另有 `sweep_summary.json`。
默认用 EdgeScore 分析样例和扫描，不自动选择最优 beta；`--lpips` 可增加 LPIPS，
`--no-metrics` 可关闭图像指标。目标域测试集不参与参数选择。

## 已准备的内容

- `../member2_adain/adain.py`：成员2原版 AdaIN，支持标量或空间 alpha。
- `../member2_adain/models/decoder.pth`：解码器权重（本地保留，Git 忽略）。
- `../member2_adain/models/vgg_normalised.pth`：已补齐的编码器权重（Git 忽略）。
- 成员2的8组参考样例位于上游 `nndl-part2（include_part1)` 分支；当前成员3的七类真实对比图在 `figures/`。
- `../member4_training_eval/style_pool/`：10张冻结风格图、哈希和1349条内容图→风格配对。
- `../member4_training_eval/project_splits/`：与训练流程一致的数据清单。
- `../member4_training_eval/GENERATOR_INTERFACE.md`：正式输出的接口约定。
- `../member4_training_eval/structure_metrics.py`：LPIPS / EdgeScore 分析脚本。

## 统一的数据路径

默认路径相对于仓库根目录，与成员4共用：

```text
Project/
├── data/                         # 不提交到 Git
│   ├── PACS/
│   │   ├── photo/
│   │   ├── art_painting/
│   │   ├── cartoon/
│   │   └── sketch/
│   ├── aug_sp/
│   │   ├── images/photo/<class>/
│   │   ├── photo_train.txt
│   │   ├── pairs.txt
│   │   ├── manifest_ours.txt
│   │   └── generation.json
│   └── augmented/manifest_ours.txt  # 成员4兼容清单，引用 aug_sp/images
├── member2_adain/
├── member3_structure_preserving/
└── member4_training_eval/
```

数据在其他位置时，在运行程序的 PowerShell 中设置（填 PACS 的父目录）：

```powershell
$env:NNDL_DATA_ROOT = 'D:\你的数据目录'
```

PyCharm 运行时也应使用同样的环境变量；不设置时默认 `Project/data`。
成员1的原始加载器仍按其上游约定读取 `images/`，本工作区使用成员4的训练接口。

## Python 环境

本次运行复用已有 Conda `NN` 环境，已安装
PyTorch、torchvision、Pillow、NumPy 和 SciPy。
根目录 `nn.ps1` 从 `$env:USERPROFILE\.conda\envs\NN\python.exe` 启动解释器；
其他平台或环境可直接用 `python -m member3_structure_preserving.generate ...`。
当前 PyTorch 是 CPU 版本；OpenCV 已安装，可选的 LPIPS 尚未安装。

从仓库根目录，在准备使用的环境中执行（已有依赖无需重新安装）：

```powershell
conda activate NN
python -m pip install -r member3_structure_preserving/requirements.txt
```

LPIPS 需要时另装 `python -m pip install lpips`。

## 在成员3代码里复用接口

从仓库根目录运行，或将后续脚本以 `python -m member3_structure_preserving.<模块名>` 启动：

```python
from member3_structure_preserving.stylizer import StructurePreservingStylizer
from member4_training_eval import config

content_path = config.IMAGE_ROOT / "photo/dog/056_0022.jpg"
style_path = config.STYLE_POOL_IMAGES / "scene_de_rue.jpg"
generator = StructurePreservingStylizer()
result = generator.stylize(content_path, style_path, alpha=0.5, beta=0.7,
                          with_baseline=True)
result.image.save("ours.jpg")
# result.structure / alpha_map / feature_alpha 为结构图和强度图。
# result.baseline 为普通 AdaIN 对照。
```

`AdaINModel()` 默认从 `member2_adain/models/` 加载权重。
`vgg_normalised.pth` 不在上游 Git 分支内，本工作区已从成员2代码指定的
`naoto0804/pytorch-AdaIN` GitHub release 下载。以后换电脑时，模型会尝试自动补齐缺失权重。

本机使用 `NN` 环境已通过：预训练编码器/解码器推理、beta=0 与成员2输出一致、
结构响应及参数边界、冻结配对、成员4读取器与公平性检查、损坏文件续跑及指标一致性检查。
PACS 已下载到 `data/PACS/`：原始9991张，项目清单使用9990张；
六份清单校验和全部原图解码检查通过。压缩包及来源记录在 `data/downloads/`。
真实七类对比样例已生成，五组 beta 的7张小样本扫描已跑通；尚未运行分类训练。

## 正式生成时的约定

- 原图只使用 `config.SPLIT_FILES["photo_train"]`，共1349张。
- 读取 `config.STYLE_POOL_MANIFEST` 的 `pairing`，不重新随机分配风格。
- 基础 alpha 为0.5，每张原图生成一张图；beta 为成员3方法参数。
- 输出图片放到 `config.DATA_ROOT / "aug_sp/images/photo/<class>"`。
- manifest 每行：`<图片相对DATA_ROOT路径> <标签1..7> <原图相对PACS路径> <风格名> <alpha> <beta>`。

例如：

```text
aug_sp/images/photo/dog/056_0022__scene_de_rue.jpg 1 photo/dog/056_0022.jpg scene_de_rue 0.5 0.7
```

## 数据准备后的检查

从仓库根目录执行：

```powershell
.\nn.ps1 member4_training_eval/verify_splits.py
```

两组正式 manifest 到位后运行公平性检查；使用实际文件路径：

```powershell
.\nn.ps1 member4_training_eval/fairness_check.py --adain data/augmented/manifest_adain.txt --ours data/augmented/manifest_ours.txt
```

使用外部数据目录时，上述 manifest 参数也应指向外部目录。
