# gcp-vqvae

Minimal inference wrapper for GCP-VQVAE with encode, embed, and decode modes.

## Install

```bash
pip install -e /path/to/vq_encoder_decoder/gcp-vqvae
```

## Usage

```python
import torch
from gcp_vqvae import GCPVQVAE

model = GCPVQVAE(
    trained_model_dir="/path/to/trained_run",
    checkpoint_path="checkpoints/best_valid.pth",
    config_vqvae="config_vqvae.yaml",
    config_encoder="config_gcpnet_encoder.yaml",
    config_decoder="config_geometric_decoder.yaml",
    mode="encode",
    device="cuda",
    mixed_precision="bf16",
)

records = model.encode(
    data_path="/path/to/h5_dir",
    batch_size=8,
    num_workers=0,
    shuffle=False,
)
```

### Encode from raw PDB/CIF directory

```python
records = model.encode(
    pdb_dir="/path/to/pdb_dir",
    batch_size=2,
)
```

### Embed

```python
embedder = GCPVQVAE(
    trained_model_dir="/path/to/trained_run",
    checkpoint_path="checkpoints/best_valid.pth",
    config_vqvae="config_vqvae.yaml",
    config_encoder="config_gcpnet_encoder.yaml",
    config_decoder="config_geometric_decoder.yaml",
    mode="embed",
    device="cuda",
    mixed_precision="bf16",
)

embeddings = embedder.embed(
    data_path="/path/to/h5_dir",
    batch_size=4,
    keep_missing_tokens=False,
)
```

### Decode

```python
decoder = GCPVQVAE(
    trained_model_dir="/path/to/trained_run",
    checkpoint_path="checkpoints/best_valid.pth",
    config_vqvae="config_vqvae.yaml",
    config_encoder="config_gcpnet_encoder.yaml",
    config_decoder="config_geometric_decoder.yaml",
    mode="decode",
    device="cuda",
    mixed_precision="bf16",
)

coords = decoder.decode([[1, 2, 3, 4], [5, 6, 7]], batch_size=2)
```

## Notes
- Raw PDB/CIF inputs are preprocessed using the same logic as the `demo/` pipeline.
- `torch-geometric` and `torch-cluster` require matching CUDA wheels; install them per the PyG docs.
- Mixed precision defaults to `bf16` on CUDA; set `mixed_precision="no"` to disable or pass `dtype` explicitly.
