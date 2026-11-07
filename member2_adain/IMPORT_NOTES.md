# 成员2代码导入记录

- 来源仓库：https://github.com/liuluoyusz-max/nndl
- 来源分支：`nndl-part2（include_part1)`
- 来源提交：`416bdc9811d67aae80fd79e8fac1ae0f37d5f339`
- 来源目录：`nndl-main - part2/`

为成员3工作区导入的内容：

本次 GitHub 提交包含未改动算法的 `adain.py` 与本说明，作为成员3的运行依赖。
下表中的权重、成员2参考样例、流程图和上游交接文件为本地参考内容，未一并提交；
仍可从上述成员2分支获取。成员3的真实对比图和结果记录另放在成员3目录。

| 上游路径 | 本地路径 |
|---|---|
| `adain.py` | `member2_adain/adain.py`，原样导入 |
| `models/decoder.pth` | `member2_adain/models/decoder.pth`，原样导入，Git 忽略 |
| `data/aug_adain/samples/*` | `member2_adain/examples/*`，内容不变 |
| `figures/*` | `member2_adain/figures/*` |
| `handoff_member2.md` | `member2_adain/UPSTREAM_HANDOFF.md`，上游历史说明 |

上游批量生成脚本依赖未提交的 `style_pool.py`，且使用 `data/images` 和两列清单。
当前只复用成员3需要的 AdaIN 核心；生成数据时使用成员4的冻结风格池、配对表及 manifest 约定。
原始生成脚本仍可从已拉取的远程分支查阅。

上游未提交 `vgg_normalised.pth`；本地已从 `adain.py` 中指定的地址补齐：
https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth
（80,102,481字节，Git 忽略）。已使用本机 `NN` 环境成功加载编码器和解码器，
并通过小尺寸合成输入的标量/空间 alpha 推理验证。
上游8组样例的 alpha=1.0，正式实验协议的基础 alpha=0.5。

当前工作区将成员4 `config.py` 和 `verify_splits.py` 的默认数据根目录
从成员4个人电脑路径改成 `<仓库>/data`，仍支持 `NNDL_DATA_ROOT` 覆盖；
训练超参数、数据划分和冻结配对均未修改。
