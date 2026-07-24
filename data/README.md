# PBSInstr and PBSBench annotations

This directory contains publication-ready QA annotations. Source images are not relicensed here.

```text
data/
├── PBSInstr/
│   ├── cell_captions.jsonl
│   ├── cell_train.jsonl
│   ├── cell_val.jsonl
│   ├── slide_train.jsonl
│   └── slide_val.jsonl
└── PBSBench/
    ├── cell_id_test.jsonl
    ├── cell_ood_test.jsonl
    └── slide_id_test.jsonl
```

## Released annotation counts

| File | QA pairs |
|---|---:|
| `PBSInstr/cell_train.jsonl` | 20,985 |
| `PBSInstr/cell_val.jsonl` | 3,240 |
| `PBSInstr/slide_train.jsonl` | 1,142 |
| `PBSInstr/slide_val.jsonl` | 144 |
| `PBSBench/cell_id_test.jsonl` | 3,289 |
| `PBSBench/cell_ood_test.jsonl` | 4,495 |
| `PBSBench/slide_id_test.jsonl` | 138 |

`PBSInstr/cell_captions.jsonl` contains 28,970 Phase-1 cell-caption records.
The same manifest carries each cell's containing `patch_path` and is therefore
also the Phase-2 cell-patch pair manifest.

The released QA counts differ slightly from those reported in the paper
because several annotation errors were corrected during final data
preparation, which shifted the distribution across the published splits.

Each JSON line contains stable `id`, human-readable `image_id`, relative `image_path`, dataset-native `source`, `level`, `split`, `domain`, `task`, `question_type`, `question`, `answer`, and optional `options`. Question types are `true_false`, `multiple_choice`, `fill_blank`, and `open`.

## Split granularity

The cell ID benchmark is a **cell-crop-level holdout**. Its cell images are
disjoint from PBSInstr cell images, but their parent S-BIAD440 WSIs are not:
all 189 source slides represented in the cell ID test set also contribute
different cell crops to cell training. It must not be described as a
patient/slide-level holdout.

The slide ID benchmark is a **WSI-level holdout** defined by
`slide_split.csv`: PBSInstr contains 192 training and 24 validation slides,
while 23 different slides form PBSBench. No WSI appears in more than one of
these splits.

## Linking QA records to images

`id` identifies a QA pair; `image_id` identifies the image shared by one or more QA pairs. For an in-domain cell, the identifier has the form:

```text
S-BIAD440:<slide>:<patch-x>_<patch-y>:<Cellpose-object-index>
```

Its `source` object records the S-BIAD440 WSI path, level-0 patch coordinates, patch size, object index, and curated bounding box. `pbsbench-prepare-data` uses this locator to generate the record's `image_path` exactly. Slide records use `S-BIAD440:<slide>` and point directly to the corresponding downloaded WSI.

OOD records use provider-native identifiers rather than local download
wrappers: TCIA records use paths such as `NGS/NGS_5602.tiff`, Kaggle APL
records use paths below `All/All/`, and WBC LISC records use the unique source
filename. The materializer searches below the user-supplied extraction root
and recognizes filenames rewritten by Roboflow exports.

Example:

```json
{
  "image_id": "S-BIAD440:III_1:18432_95744:0016",
  "image_path": "cells/S-BIAD440/III_1/18432_95744_0016.png",
  "source": {
    "dataset": "S-BIAD440",
    "path": "images/III_1.tiff",
    "patch_x": 18432,
    "patch_y": 95744,
    "patch_size": 512,
    "object_index": 16,
    "bbox": {"left": 382, "top": 30, "right": 442, "bottom": 99}
  }
}
```

The released PBSInstr and PBSBench QA manifests are versioned directly in this
repository under `data/PBSInstr/` and `data/PBSBench/`. Source images are
acquired separately and are not redistributed here:

- [S-BIAD440 / Haemorasis](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD440)
  supplies the in-domain whole-slide images.
- [AML-Cytomorphology LMU](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/),
  [Acute Promyelocytic Leukemia (APL)](https://www.kaggle.com/datasets/eugeneshenderov/acute-promyelocytic-leukemia-apl),
  and [WBC LISC v3](https://universe.roboflow.com/wbcs/wbc-lisc/dataset/3)
  supply the out-of-domain cell images.

See [docs/PREPROCESSING.md](../docs/PREPROCESSING.md) for path-based
materialization commands and [`THIRD_PARTY.md`](../THIRD_PARTY.md) for upstream
licenses and notices.

## License

The curated PBSInstr and PBSBench question-answer annotations and associated
metadata are available under [CC BY 4.0](LICENSE). This license does not cover
source microscopy images or other third-party assets.
