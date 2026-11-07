# 已运行的 beta 小样本检查

在真实 PACS Photo Training 的前7张图片上，用同一套冻结内容→风格配对运行：

```powershell
.\nn.ps1 -m member3_structure_preserving.generate sweep --limit 7 --betas 0 0.3 0.5 0.7 1
```

固定 alpha=0.5，内容/风格短边512，JPEG质量95。7张图均来自 dog 类；
这是流程和参数行为检查，不是全类别统计或参数选优实验。

| beta | 平均 alpha | 平均 EdgeScore | 有效图片数 |
|---:|---:|---:|---:|
| 0.0 | 0.5000 | 0.3589 | 7 |
| 0.3 | 0.4469 | 0.3692 | 7 |
| 0.5 | 0.4115 | 0.3752 | 7 |
| 0.7 | 0.3761 | 0.3794 | 7 |
| 1.0 | 0.3231 | 0.3913 | 7 |

来源：`data/aug_sp/sweeps/sweep_n7/sweep_summary.json`。每个参数目录都包含实际生成图、
对应的普通 AdaIN 参考图、训练清单、生成记录和 `analysis.json`。

这批图随 beta 增大，边缘重叠指标提高，同时平均风格强度下降。
该结果不能证明跨域分类准确率提高，也不足以将变化独立归因于空间保护。
正式默认 beta 仍为预先设置的0.7，未根据这7张图选择“最优值”。LPIPS 尚未计算。

另有七类各一张的结构图、alpha图和对比面板，保存在
`data/aug_sp/previews/samples_n7/a0p5_b0p7/figures/`；该批样例与普通 AdaIN 参考
通过了成员4的 `fairness_check.py --subset` 检查。
