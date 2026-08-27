# DINOv2 ViT-Large Experiment Progress

This is the source-branch template. Completed E0, E1 and E2 reports are generated after evaluation and pushed to the `experiment-reports` branch; no checkpoint, data, log or result JSON is committed to Git.

## Protocol

| Phase | Encoder | Loss | Batch size | Maximum epochs |
|---|---|---|---:|---:|
| E0 | Frozen ViT-Large | BCE | 8 | 50 |
| E1 | Frozen ViT-Large | BCE + Dice | 8 | 50 |
| E2 | Full ViT-Large fine-tuning | BCE + Dice | 1 | 50 |

Use test Dice and Jaccard/IoU, each reported as mean ± population standard deviation over test slices, with min/max and the best validation Dice. Include seed, code commit, duration, peak VRAM, checkpoint path and a statement that paper comparisons require matching data/split/protocol.

## Report lifecycle

1. Train writes atomic `last`, `best`, periodic and `final` checkpoints on the server.
2. The evaluator writes the per-slice test JSON on the server.
3. The runner appends the phase summary here in the external report worktree and pushes only this Markdown to `experiment-reports`.
