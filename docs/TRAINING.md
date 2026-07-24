# PBS-VL training

PBS-VL is trained bottom-up while the DinoBloom vision backbone and downstream Qwen2.5-VL backbone remain frozen.

| Phase | Config | Input | Trainable modules | Output used by |
|---|---|---|---|---|
| 1. Cell representation | `01_cell_representation.yaml` | cell crop + morphology caption | cell Q-Former | phases 2 and 3 |
| 2. Cell–patch alignment | `02_cell_patch_alignment.yaml` | cell crop + containing patch | patch Perceiver and projections | phase 4 |
| 3. Cell QA | `03_cell_qa.yaml` | cell crop + question/answer | cell Q-Former and projector | cell inference |
| 4. Slide QA | `04_slide_qa.yaml` | patch features + slide question/answer | slide Perceiver and projector | slide inference |

Phase 1 uses BLIP-2 image-text contrastive, image-text matching, and language-modeling losses. Phase 2 uses global cosine and local token-level contrastive alignment. Phases 3 and 4 use Qwen2.5-VL causal language-modeling loss.

```text
cell representation checkpoint ──┬──> cell–patch alignment
                                 └──> cell QA
cell–patch aligned patch features ──> slide QA
```

Run a phase with `pbsbench-train --config <config>`. All paths are portable. Private model downloads must use provider login/environment mechanisms; never put access tokens in YAML.

Download the DinoBloom-L checkpoint expected by the default configs:

```bash
python -m pip install --upgrade huggingface_hub
hf download MarrLab/DinoBloom pytorch_model_l.bin --local-dir weights
```

The loader uses the upstream DINOv2 ViT-L/14 architecture without DINOv2
pretrained weights, then loads `weights/pytorch_model_l.bin`. It also accepts
the legacy DinoBloom trainer checkpoint format containing a `teacher` state.

Phase 3 uses the complete released cell-QA set: 20,985 training
questions rather than the later three-questions-per-cell downsample. Its 3,240
validation questions remain separate and are evaluated after every epoch.

Phase 4 likewise preserves the source slide split: 1,142 training questions
from 192 slides and 144 validation questions from 24 disjoint slides. The
validation set is evaluated after every epoch.

Phase 2 reuses `data/PBSInstr/cell_captions.jsonl`, whose records contain both
the cell `image_path` and containing `patch_path`. The frozen patch backbone
feeds a trainable Perceiver, as described in the paper.

Before Phase 4, tile each WSI and encode its retained patches with the Phase-2
checkpoint:

```bash
pbsbench-tile \
  --slides data/images/slides/S-BIAD440 \
  --output data/processed/patches

pbsbench-extract-features \
  --config configs/02_cell_patch_alignment.yaml \
  --checkpoint checkpoints/cell_patch_alignment \
  --patches data/processed/patches \
  --output data/processed/patch_features
```

For a directory `data/processed/patches/III_1/`, feature extraction writes
`data/processed/patch_features/III_1.pt`, which is the `feature_path` referenced
by slide QA records. Training checkpoints contain only trainable PBS-VL
parameters; frozen DinoBloom, BLIP-2, and Qwen weights are reloaded from their
official sources.
