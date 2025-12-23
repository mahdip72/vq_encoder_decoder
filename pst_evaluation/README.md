# Simplified Scripts (lite embeddings)

These commands assume:
- Python env: `source /path/to/StructTokenBench/bin/activate`
- Python: `python`
- H5 embeddings: `./inference_embed_results/.../embeddings`
- Data roots:
  - Functional: `./pst_evaluation/struct_token_bench_release_data/data/functional/local`
  - Structural: `./pst_evaluation/struct_token_bench_release_data/data/structural`
  - Physicochemical (ATLAS): `./pst_evaluation/struct_token_bench_release_data/data/physicochemical`
  - Sensitivity (Apo/Holo): `./pst_evaluation/struct_token_bench_release_data/data/sensitivity`

Adjust paths if your local layout is different.

Optional shorthand:
```bash
export EMB_DIR=./inference_embed_results/.../embeddings
```

## Datasets
- StructTokenBench evaluation data (labels/splits used by eval scripts via `--data-root`): download and extract into `./pst_evaluation/struct_token_bench_release_data/`
  - [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/mpngf_umsystem_edu/IQDppFECYFI4Rbo9WnqYODhcAcpo6SCLLInO2vJ_m8jcy9A?e=9PKz9r)
- PST raw CIF/H5 structures (used to generate embeddings): download and extract into, then run `inference_embed.py` (repo root) to produce H5 embeddings for the eval scripts (`--h5`)
  - [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/mpngf_umsystem_edu/IQDRerYYUdb1Ra73OmACyIILAfrlSyUV_aqFNMHk-3GH8UA?e=DjtgC7)

## BioLIP2 binding (lite)
```bash
python pst_evaluation/biolip2_binding_eval.py \
  --h5 "$EMB_DIR/vq_embed_biolip2_binding_lite_model.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/functional/local
```

## BioLIP2 catalytic (lite)
```bash
python pst_evaluation/biolip2_catalytic_eval.py \
  --h5 "$EMB_DIR/vq_embed_biolip2_catalytic_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/functional/local
```

## ProteinShake binding site (lite)
```bash
python pst_evaluation/proteinshake_binding_eval.py \
  --h5 "$EMB_DIR/vq_embed_proteinshake_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/functional/local
```

## InterPro binding (lite)
```bash
python pst_evaluation/interpro_binding_eval.py \
  --h5 "$EMB_DIR/vq_embed_interpro_binding_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/functional/local
```

## InterPro active site (lite)
```bash
python pst_evaluation/interpro_activesite_eval.py \
  --h5 "$EMB_DIR/vq_embed_interpro_activesite_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/functional/local
```

## InterPro conserved site (lite)
```bash
python pst_evaluation/interpro_conserved_eval.py \
  --h5 "$EMB_DIR/vq_embed_conserved_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/functional/local
```

## ProteinGLUE epitope region (lite)
```bash
python pst_evaluation/proteinglue_epitope_region_eval.py \
  --h5 "$EMB_DIR/vq_embed_proteinglue_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/functional/local
```

## ATLAS RMSF (lite)
```bash
python pst_evaluation/atlas_rmsf_eval.py \
  --h5 "$EMB_DIR/vq_embed_atlas_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/physicochemical
```

## ATLAS B-factor (lite)
```bash
python pst_evaluation/atlas_bfactor_eval.py \
  --h5 "$EMB_DIR/vq_embed_atlas_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/physicochemical
```

## ATLAS NEQ (lite)
```bash
python pst_evaluation/atlas_neq_eval.py \
  --h5 "$EMB_DIR/vq_embed_atlas_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/physicochemical
```

## Remote homology (lite)
```bash
python pst_evaluation/remote_homology_eval.py \
  --h5 "$EMB_DIR/vq_embed_remote_homology_train_tst_valid.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/structural \
  --progress
```

## Apo/Holo + Fold Switching (unsupervised, lite)
```bash
python pst_evaluation/apolo_unsupervised_eval.py \
  --h5 "$EMB_DIR/vq_embed_apolo_lite.h5" \
  --data-root ./pst_evaluation/struct_token_bench_release_data/data/sensitivity \
  --target-field tm_score
```
