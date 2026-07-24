# Reconstructing published images

Install the OpenSlide native library and Python bindings in the PBSBench Conda
environment before running the WSI commands:

```bash
conda install --channel conda-forge openslide openslide-python -y
```

Download and extract [S-BIAD440](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD440). Its `images/` directory must contain WSI files such as `III_1.tiff`.

```bash
pbsbench-prepare-data \
  --annotations data/PBSInstr/cell_captions.jsonl \
                data/PBSInstr/cell_train.jsonl data/PBSInstr/cell_val.jsonl \
                data/PBSInstr/slide_train.jsonl data/PBSInstr/slide_val.jsonl \
                data/PBSBench/cell_id_test.jsonl data/PBSBench/slide_id_test.jsonl \
  --source S-BIAD440=/path/to/extracted/S-BIAD440 \
  --output data/images
```

> [!NOTE]
> The default configs expect `data/images`. If you use a different
> `--output`, override `data.image_root` during training or pass
> `--image-root` during cell evaluation.

The annotations store the original slide filename, `512 × 512` patch origin, curated Cellpose-SAM object index, and bounding box. The command reads the region directly from the WSI and reproduces the contextual `224 × 224` cell crop used during QA curation. It also saves the corresponding patch and symlinks slide-level images. Use `--link-mode copy` if symlinks are unsuitable.

OOD records refer to images distributed by their source datasets. Provide one root for every source named in the selected annotations, for example:

```bash
pbsbench-prepare-data --annotations data/PBSBench/cell_ood_test.jsonl \
  --source AML-Cytomorphology_LMU=/path/to/AML-Cytomorphology_LMU \
  --source APL=/path/to/APL \
  --source WBC_LISC=/path/to/WBC_LISC \
  --output data/images
```

The root may be either the extracted dataset directory or a parent directory
introduced by the download tool. The materializer first checks the
provider-native relative path, then searches nested extraction directories by
the unique source filename. It also recognizes Roboflow export names such as
`Baso_46-1__8_bmp.rf.<hash>.jpg` for the published source locator
`Baso_46-1__8.bmp`. Ambiguous or missing matches stop with an error rather than
silently selecting an image.

Generated images are ignored by Git. The source datasets retain their own licenses.

## WSI patches and Phase-2 features

After materializing the S-BIAD440 slides, create non-overlapping 512-pixel
patches. Pass `--qc-model` to apply an available Haemorasis QC model; without
it, all patches are retained.

```bash
pbsbench-tile \
  --slides data/images/slides/S-BIAD440 \
  --output data/processed/patches \
  --patch-size 512

pbsbench-extract-features \
  --config configs/02_cell_patch_alignment.yaml \
  --checkpoint checkpoints/cell_patch_alignment \
  --patches data/processed/patches \
  --output data/processed/patch_features
```

The extractor preserves slide names: patches under `III_1/` become
`III_1.pt`. Each file stores fixed-length Phase-2 Perceiver tokens and the
ordered patch filenames. These generated files are ignored by Git.
