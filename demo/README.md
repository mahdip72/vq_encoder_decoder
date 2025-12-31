# Demo Evaluation (Standalone)

This demo runs a minimal inference + evaluation loop without bells and whistles.
It scans a directory for PDB/CIF/mmCIF files,
processes them in memory, encodes tokens, optionally saves embeddings, and
optionally writes reconstructed PDBs plus TM-score/RMSD evaluation.

Structure preprocessing is required for GCP-VQVAE because most experimental PDB
files contain noise, missing residues, and multiple chains that must be cleaned
or addressed first. This has been the main root of performance degradation for our model and
many other models when not handled properly. We have already trimmed the
dataloader path as much as possible; removing more of it potentially hurts
model performance.

Dependencies
1) Use the same Python environment as the main project.
2) If you do not have one, install demo-only dependencies. First create and activate a Python environment, then run this script:

```cmd
   bash demo/install_demo.sh
```

Usage
1) Edit `demo/demo_eval_config.yaml`:
   - `data_dir`: directory containing PDB/CIF/mmCIF files (recursively scanned)
   - `trained_model_dir`: directory with checkpoint + saved training YAMLs
   - Set these based on your need: `save_indices_csv`, `save_embeddings_h5`, `save_pdb_and_evaluate`
2) Run:
```cmd
   python demo/demo_evaluation.py
```

Notes
- Both PDB and CIF files are handled automatically.
- Outputs are saved under `output_base_dir/<timestamp>/`.
