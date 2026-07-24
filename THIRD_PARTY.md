# Third-party components

PBSBench interfaces with external software, models, and image datasets. Those
components are not relicensed by this repository; obtain them from their
original sources and follow their respective terms.

## Bundled code

- [LAVIS / BLIP-2](https://github.com/salesforce/LAVIS) — selected files under
  `pbsbench/models/blip2/` are derived from LAVIS and retain the upstream
  BSD-3-Clause copyright and license notices. The retained license is available
  at `pbsbench/models/blip2/LICENSE.txt`.

## External models and preprocessing software

- [DinoBloom](https://huggingface.co/MarrLab/DinoBloom) — model checkpoints are
  downloaded separately under the Apache-2.0 license.
- [DINOv2](https://github.com/facebookresearch/dinov2) — the matching vision
  transformer architecture is loaded from the official upstream Torch Hub
  repository with `pretrained=False`; no DINOv2 checkpoint is used.
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
  — model files are downloaded separately under the Apache-2.0 license.
- [Haemorasis](https://github.com/josegcpa/haemorasis)
  ([paper](https://www.nature.com/articles/s41467-023-39676-y)) — preprocessing
  code is used under the Apache-2.0 license. The paper's quality-control network
  is released separately in
  [`josegcpa/quality-net`](https://github.com/josegcpa/quality-net).
- [Cellpose](https://www.cellpose.org/)
  ([repository](https://github.com/mouseland/cellpose)) — installed separately
  under its BSD-3-Clause license.

## External image datasets

The repository publishes curated QA annotations and metadata, not copies of the
source images. Download source images from the providers below and comply with
their access, attribution, citation, and reuse requirements.

- [S-BIAD440 / Haemorasis](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD440)
- [AML-Cytomorphology LMU](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)
  — released by TCIA under CC BY 3.0; cite the dataset as requested by TCIA.
- [Acute Promyelocytic Leukemia (APL)](https://www.kaggle.com/datasets/eugeneshenderov/acute-promyelocytic-leukemia-apl)
  — consult the Kaggle data card and terms shown at download time; the public
  page does not expose a machine-readable license notice.
- [WBC LISC v3](https://universe.roboflow.com/wbcs/wbc-lisc/dataset/3) — CC BY 4.0.

PBSBench's curated QA annotations and associated metadata are released under
[CC BY 4.0](data/LICENSE); that license does not extend to these source images.
