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

model = GCPVQVAE(mode="encode")  # Uses Hugging Face default model.

model = GCPVQVAE(
    mode="encode",
    hf_model_id="Mahdip72/gcp-vqvae-large",
)

records = model.encode(
    pdb_dir="/path/to/pdb_dir",
    batch_size=8,
)
```

### Encode (local checkpoint/configs)

```python
encoder = GCPVQVAE(
    trained_model_dir="/path/to/trained_run",
    checkpoint_path="checkpoints/best_valid.pth",
    config_vqvae="config_vqvae.yaml",
    config_encoder="config_gcpnet_encoder.yaml",
    config_decoder="config_geometric_decoder.yaml",
    use_hf=False,
    mode="encode",
    device="cuda",
    mixed_precision="bf16",
)

records = encoder.encode(
    pdb_dir="/path/to/pdb_dir",
    batch_size=8,
    num_workers=0,
    shuffle=False,
    show_progress=False,
)
```

### Embed

```python
embedder = GCPVQVAE(
    hf_model_id="Mahdip72/gcp-vqvae-large",
    use_hf=True,
    mode="embed",
    device="cuda",
    mixed_precision="bf16",
)

embeddings = embedder.embed(
    pdb_dir="/path/to/pdb_dir",
    batch_size=4,
    keep_missing_tokens=False,
    show_progress=False,
)
```

### Embed (local checkpoint/configs)

```python
embedder = GCPVQVAE(
    trained_model_dir="/path/to/trained_run",
    checkpoint_path="checkpoints/best_valid.pth",
    config_vqvae="config_vqvae.yaml",
    config_encoder="config_gcpnet_encoder.yaml",
    config_decoder="config_geometric_decoder.yaml",
    use_hf=False,
    mode="embed",
    device="cuda",
    mixed_precision="bf16",
)

embeddings = embedder.embed(
    pdb_dir="/path/to/pdb_dir",
    batch_size=4,
    keep_missing_tokens=False,
    show_progress=False,
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
    use_hf=False,
    mode="decode",
    device="cuda",
    mixed_precision="bf16",
)

decoded = decoder.decode([[1, 2, 3, 4], [5, 6, 7]], batch_size=2)
coords = decoded["coords"]
plddt = decoded["plddt"]  # None if the model has no pLDDT head
```

### Decode (HF model)

```python
decoder = GCPVQVAE(
    mode="decode",
    hf_model_id="Mahdip72/gcp-vqvae-large",
    use_hf=True,
    device="cuda",
    mixed_precision="bf16",
)

decoded = decoder.decode([[1, 2, 3, 4], [5, 6, 7]], batch_size=2)
coords = decoded["coords"]
plddt = decoded["plddt"]  # None if the model has no pLDDT head
```

### All modes in one object

```python
model = GCPVQVAE(mode="all")  # Loads encode/embed + decode paths.
records = model.encode(pdb_dir="/path/to/pdb_dir", batch_size=4)
decoded = model.decode(["1 2 3 4 5"], batch_size=1)
coords = decoded["coords"]
plddt = decoded["plddt"]
```

## Notes
- Raw PDB/CIF inputs are preprocessed using the same logic as the `demo/` pipeline.
- Encode/embed operate on `pdb_dir` inputs; the wrapper handles preprocessing internally.
- Encode/embed outputs include a `plddt` field (None when no pLDDT head is present).
- Decode returns a dict of batched outputs with `coords`, `mask`, and `plddt` (when available).
- Progress bars are available via `show_progress=True` (default: false).
- External logging is suppressed by default; set `suppress_logging=False` if you need verbose logs.
- `torch-geometric` and `torch-cluster` require matching CUDA wheels; install them per the PyG docs.
- Mixed precision defaults to `bf16` on CUDA; set `mixed_precision="no"` to disable or pass `dtype` explicitly.
- Hugging Face downloads are cached; pass `hf_cache_dir` or `hf_revision` if needed.
- Use `use_hf=False` with local checkpoint/config paths to override the Hugging Face default.
