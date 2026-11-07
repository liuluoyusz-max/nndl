# Structure-Preserving Style Augmentation for Cross-Domain Image Classification (PACS)

NNDL 课程小组项目仓库。Topic: single-source (Photo-only) cross-domain image
classification on PACS with structure-preserving style augmentation.

## Member 3 implementation

Member 3 uses member 2's AdaIN core from commit
`416bdc9811d67aae80fd79e8fac1ae0f37d5f339`. See
[`member3_structure_preserving/README.md`](member3_structure_preserving/README.md)
for setup and [`member2_adain/IMPORT_NOTES.md`](member2_adain/IMPORT_NOTES.md)
for the imported files and their provenance.

Member 3 and member 4 share `data/PACS/` for original images. Member 3 writes
its deliverable to `data/aug_sp/` and exports a member-4-compatible manifest
at `data/augmented/manifest_ours.txt`, pointing to the same generated images.
Set `NNDL_DATA_ROOT` to use another data directory. The dataset itself is not
included in the repository.

Member 3's checked-in [results](member3_structure_preserving/results/README.md)
contain generation manifests, audit records, structural metrics, and the beta
pilot. Seven real PACS comparison panels are in `member3_structure_preserving/figures/`.

The [one-time handoff to member 5](member3_structure_preserving/handoff_member5/README.md)
collects report drafts for 2.4, 3.5, 4.3 and 5.3, formulas, standalone figures,
the frozen interface contract, BibTeX, evidence, and a compiled PDF preview.

## Repository layout — one folder per member

| Folder | Owner | Contents |
|---|---|---|
| `member1_data/` | Member 1 (data) | PACS splits (`project_splits/`), `PACSDataset` loader, data README |
| `member2_adain/` | Member 2 | AdaIN generator, style-pool reader, sample generation |
| `member3_structure_preserving/` | Member 3 | Structure-preserving stylization, structure map, β experiments |
| `member4_training_eval/` | Member 4 | Unified training/evaluation/fairness code, frozen style pool, run results |
| `report/` | Member 5 (integration) | LaTeX progress report, section drafts, figures |

## Where things are (member 4's part)

- **Shared experiment protocol**: `member4_training_eval/config.py`
  (step budget 660, batch 64, SGD, cosine, label smoothing, geometric-only
  augmentation, seeds {0,1,2}, final-checkpoint reporting).
- **Frozen style pool**: `member4_training_eval/style_pool/`
  (10 external paintings + sha256 manifest + frozen content→style pairing,
  pool_hash `72cdbadb8970`). Verified byte-disjoint from PACS — Art/Cartoon/
  Sketch are test domains and must never supply style images.
- **Training / evaluation**: `train.py`, `run_all.py`, `evaluate.py`,
  `summarize_results.py`.
- **Fairness checks**: `fairness_check.py` (manifest constraints, style-pool
  leak guard, config-hash verification), `verify_splits.py`.
- **Generator interface** (for members 2/3): `GENERATOR_INTERFACE.md`.
- **Results so far**: `member4_training_eval/runs_results/` — JSON summaries
  only (checkpoints `.pt` are 86–556 MB each and are not committed;
  rerun `run_all.py --group original` to reproduce).

## Baseline result (experiment 1, complete)

Original group, 3 seeds, 660 steps, final checkpoint:

| | Photo | Art | Cartoon | Sketch | Avg |
|---|---|---|---|---|---|
| Original | 99.0±0.3 | 63.2±3.7 | 25.1±1.8 | 31.5±4.2 | **39.9±2.7** |

## Data setup

The PACS images live outside this repo (9,990 usable images ≈ hundreds of MB).
Download PACS, then point the code at your copy:

```
set NNDL_DATA_ROOT=<path>\to\data     # parent of the PACS folder
```

`member1_data/README.md` documents the dataset interface and the six splits.

## Report

`report/progress_report.tex` (IEEEtran). Member 4's sections live in
`report/sections/member4_*.tex`; regenerate the concatenated file with
`member4_training_eval/_sync_sections.py` if you edit them.
