# Simplified Evaluation Demo

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

Example demo config (`demo/demo_eval_config.yaml`):
```yaml
trained_model_dir: "/abs/path/to/trained_model"  # Directory of model checkpoint result in timestamped 
checkpoint_path: "checkpoints/best_valid.pth"  # Relative to trained_model_dir
config_vqvae: "config_vqvae.yaml"  # Saved training config name
config_encoder: "config_gcpnet_encoder.yaml"  # Saved encoder config name
config_decoder: "config_geometric_decoder.yaml"  # Saved decoder config name

data_dir: "/abs/path/to/pdb_cif_dir"  # Root of PDB/CIF/mmCIF files (recursive)
output_base_dir: "demo_results"  # Output parent dir (timestamped subdir created)

batch_size: 2  # Inference batch size
num_workers: 0  # DataLoader workers
max_task_samples: 0  # 0 means no limit
tqdm_progress_bar: true  # Show progress bars

alignment_strategy: "kabsch"  # kabsch or no (type of alignment before evaluation)
mixed_precision: "bf16"  # no, fp16, bf16

save_indices_csv: true  # Write discrete tokens into vq_indices.csv
save_embeddings_h5: false  # Write continuous VQ embeddings into vq_embed.h5
save_pdb_and_evaluate: true  # Write original and reconstrued coordinates in PDB format and evaluate TM-score/RMSD
```

Notes
- Both PDB and CIF files are handled automatically.
- Outputs are saved under `output_base_dir/<timestamp>/`.
