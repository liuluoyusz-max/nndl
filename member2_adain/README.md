# Member 2 — AdaIN Style Augmentation

Owner: member 2. AdaIN generator and style-transfer implementation.

Expected contents (member 2 to fill in):

- `adain.py` — AdaIN model, fixed interface `stylize(content, style, alpha)`
- `style_pool.py` — external style-pool reader
- `build_adain_aug.py` — batch generation of the Photo-training augmented set
- `generate_adain_samples.py` — side-by-side sample generation
- `models/` — `vgg_normalised.pth`, `decoder.pth` (large weights: do not commit)
- `figures/` — AdaIN flowchart, generated samples

## Interface contract (agreed with member 4)

See `../member4_training_eval/GENERATOR_INTERFACE.md` for the manifest format
and the frozen style-pool / pairing rules. Key points:

- The style pool is **10 external paintings** (`../member4_training_eval/style_pool/images/`),
  NOT from PACS. Art/Cartoon/Sketch are test domains and must never be style sources.
- The content→style pairing is frozen in
  `../member4_training_eval/style_pool/style_pool.json` (seed 42).
- Manifest format: `<img> <label 1..7> <original> <style> <alpha> [beta]`,
  written under `data_root/augmented/<group>/`.
- Member 4's `convert_aug_manifest.py` converts this folder's `pairs.txt`
  output into that manifest format.
- Default generation alpha is **0.5** (member 2's scripts default to 1.0 —
  pass `--alpha 0.5` explicitly).
