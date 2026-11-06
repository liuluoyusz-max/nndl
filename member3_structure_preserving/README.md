# Member 3 — Structure-Preserving Style Augmentation

Owner: member 3. Structure-preserving stylization (structure map M,
adaptive style strength α(x,y), β parameter).

Expected contents (member 3 to fill in):

- Structure-preserving generator (uses member 2's `stylize` interface with a
  spatial alpha map — see `adain.py` in `../member2_adain/`)
- Structure map construction and β-parameter experiment scripts
- Figures: M-map illustration, α(x,y) illustration

## Interface contract (agreed with members 2 and 4)

See `../member4_training_eval/GENERATOR_INTERFACE.md`. Key points:

- Read the frozen content→style pairing from
  `../member4_training_eval/style_pool/style_pool.json` — do not recompute it.
- Style images come from `../member4_training_eval/style_pool/images/`
  (10 external paintings; nothing from PACS).
- Manifest format: `<img> <label 1..7> <original> <style> <alpha> <beta>`,
  written under `data_root/augmented/ours/`.
