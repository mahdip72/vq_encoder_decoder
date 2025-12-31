import math
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from models.gcpnet.models.base import instantiate_module, load_encoder_config

from data_utils import find_structure_files, process_structure_file

BOND_LENGTHS = {
    "N-CA": 1.458,
    "CA-C": 1.525,
    "C-O": 1.231,
    "C-N": 1.329,
}

PREPROCESS_MIN_LEN = 25
PREPROCESS_MAX_MISSING_RATIO = 0.2
PREPROCESS_MAX_CONSECUTIVE_MISSING = 15
PREPROCESS_USE_GAP_ESTIMATION = True
PREPROCESS_GAP_THRESHOLD = 5
PREPROCESS_SIMILARITY_THRESHOLD = 0.90
PREPROCESS_INCLUDE_FILE_INDEX = True


def merge_features_and_create_mask(features_list, max_length=512):
    padded_tensors = []
    mask_tensors = []
    for t in features_list:
        if t.size(0) < max_length:
            size_diff = max_length - t.size(0)
            pad = torch.zeros(size_diff, t.size(1), device=t.device)
            t_padded = torch.cat([t, pad], dim=0)
            mask = torch.cat([
                torch.ones(t.size(0), dtype=torch.bool, device=t.device),
                torch.zeros(size_diff, dtype=torch.bool, device=t.device),
            ], dim=0)
        else:
            t_padded = t
            mask = torch.ones(t.size(0), dtype=torch.bool, device=t.device)
        padded_tensors.append(t_padded.unsqueeze(0))
        mask_tensors.append(mask.unsqueeze(0))

    result = torch.cat(padded_tensors, dim=0)
    mask = torch.cat(mask_tensors, dim=0)
    return result, mask


def amino_acid_to_tensor(sequence, max_length, letter_to_num=None):
    letter_to_num = letter_to_num or {
        'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
        'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
        'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
        'N': 2, 'Y': 18, 'M': 12, 'X': 20,
    }

    pad_index = letter_to_num.get('X', 20)
    encoded = [letter_to_num.get(aa, pad_index) for aa in sequence]
    if len(encoded) < max_length:
        encoded.extend([pad_index] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length]
    return torch.tensor(encoded, dtype=torch.long)


def enforce_backbone_bonds(coords: torch.Tensor, changed: torch.Tensor = None) -> torch.Tensor:
    n_res = coords.size(0)
    for i in range(n_res):
        for (a, b, key) in ((0, 1, "N-CA"), (1, 2, "CA-C"), (2, 3, "C-O")):
            if changed is None or changed[i, a] or changed[i, b]:
                if torch.isnan(coords[i, b]).any():
                    if a > 0:
                        v = coords[i, a] - coords[i, a - 1]
                    else:
                        v = coords[i, a + 1] - coords[i, a]
                else:
                    v = coords[i, b] - coords[i, a]
                norm = v.norm(dim=-1, keepdim=True)
                if norm.item() > 1e-6:
                    coords[i, b] = coords[i, a] + v / norm * BOND_LENGTHS[key]
        if i < n_res - 1:
            if changed is None or changed[i, 2] or changed[i + 1, 0]:
                v = coords[i + 1, 0] - coords[i, 2]
                norm = v.norm(dim=-1, keepdim=True)
                if norm > 1e-6:
                    coords[i + 1, 0] = coords[i, 2] + v / norm * BOND_LENGTHS["C-N"]
    return coords


def enforce_ca_spacing(coords: torch.Tensor, changed: torch.Tensor = None, ideal: float = 3.8) -> torch.Tensor:
    n_res = coords.size(0)
    if changed is not None:
        res_changed = changed.any(dim=1)
    else:
        res_changed = torch.ones(n_res, dtype=torch.bool, device=coords.device)

    for i in range(n_res - 1):
        if changed is not None and not res_changed[i] and not res_changed[i + 1]:
            continue
        ca_i = coords[i, 1]
        ca_j = coords[i + 1, 1]
        v = ca_j - ca_i
        d = v.norm()
        if d > 1e-6:
            delta = v * ((d - ideal) / d)
            if changed is not None and not res_changed[i]:
                coords[i + 1] = coords[i + 1] - delta
            elif changed is not None and not res_changed[i + 1]:
                coords[i] = coords[i] + delta
            else:
                coords[i] = coords[i] + delta / 2
                coords[i + 1] = coords[i + 1] - delta / 2
    return coords


def _normalize(tensor, dim=-1, eps=1e-8):
    return torch.nan_to_num(tensor / torch.norm(tensor, dim=dim, keepdim=True).clamp_min(eps))


class DemoStructureDataset(Dataset):
    def __init__(
        self,
        data_dir,
        *,
        max_length,
        encoder_config_path,
        max_task_samples=0,
        progress=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.max_length = max_length
        self.letter_to_num = {
            'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
            'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
            'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
            'N': 2, 'Y': 18, 'M': 12, 'X': 20,
        }

        encoder_cfg = load_encoder_config(encoder_config_path)
        self.pretrained_featuriser = instantiate_module(encoder_cfg.get("features"))

        task_cfg = encoder_cfg.get("task")
        self.pretrained_task_transform = (
            instantiate_module(task_cfg.get("transform"))
            if isinstance(task_cfg, dict)
            else None
        )

        files = find_structure_files(data_dir)
        if not files:
            raise RuntimeError(f"No structure files found under {data_dir}")

        stats = Counter()
        samples = []
        iterator = tqdm(
            enumerate(files),
            total=len(files),
            desc="Processing structures",
            disable=not progress,
        )

        for file_index, file_path in iterator:
            try:
                file_samples, file_stats = process_structure_file(
                    file_index,
                    file_path,
                    max_len=max_length,
                    min_len=PREPROCESS_MIN_LEN,
                    similarity_threshold=PREPROCESS_SIMILARITY_THRESHOLD,
                    gap_threshold=PREPROCESS_GAP_THRESHOLD,
                    use_gap_estimation=PREPROCESS_USE_GAP_ESTIMATION,
                    max_missing_ratio=PREPROCESS_MAX_MISSING_RATIO,
                    max_consecutive_missing=PREPROCESS_MAX_CONSECUTIVE_MISSING,
                    include_file_index=PREPROCESS_INCLUDE_FILE_INDEX,
                )
            except Exception:
                stats['errors'] += 1
                continue

            stats.update(file_stats)
            samples.extend(file_samples)

            if max_task_samples and len(samples) >= max_task_samples:
                break

        self.stats = stats
        self.samples = samples

    @staticmethod
    def handle_nan_coordinates(coords: torch.Tensor):
        nan_mask = torch.isnan(coords).any(dim=-1)
        if not nan_mask.any():
            return coords, coords.new_ones(coords.size(0), dtype=torch.bool)

        coords = coords.clone()
        n_res, n_atoms, _ = coords.shape

        for atom in range(n_atoms):
            flat = coords[:, atom].clone()
            mask = nan_mask[:, atom]
            valid = torch.where(~mask)[0]
            if valid.numel() == 0:
                flat[:] = 0.0
            else:
                first, last = valid[0].item(), valid[-1].item()
                flat[:first] = flat[first]
                flat[last + 1:] = flat[last]
                for a, b in zip(valid[:-1], valid[1:]):
                    gap = b - a - 1
                    if gap > 0:
                        v = flat[b] - flat[a]
                        dist = v.norm().item()
                        l_target = gap * 3.8
                        if l_target > dist:
                            u = v / dist
                            temp = torch.tensor([1, 0, 0], device=v.device, dtype=v.dtype)
                            if abs((u * temp).sum()) > 0.9:
                                temp = torch.tensor([0, 1, 0], device=v.device, dtype=v.dtype)
                            n = torch.linalg.cross(u, temp)
                            n = n / n.norm()

                            h_squared = (l_target / math.pi) ** 2 - (dist / 2) ** 2
                            if h_squared < 0:
                                w = torch.linspace(0, 1, gap + 2, device=flat.device, dtype=flat.dtype)[1:-1].unsqueeze(1)
                                flat[a + 1:b] = flat[a] * (1 - w) + flat[b] * w
                                continue

                            h = h_squared ** 0.5
                            for j in range(1, gap + 1):
                                theta = j * math.pi / (gap + 1)
                                d_j = dist * j / (gap + 1)
                                h_j = h * math.sin(theta)
                                flat[a + j] = flat[a] + d_j * u + h_j * n
                        else:
                            w = torch.linspace(0, 1, gap + 2, device=flat.device, dtype=flat.dtype)[1:-1].unsqueeze(1)
                            flat[a + 1:b] = flat[a] * (1 - w) + flat[b] * w
            coords[:, atom] = flat

        coords = enforce_ca_spacing(coords, changed=nan_mask)
        coords = enforce_backbone_bonds(coords, changed=nan_mask)
        return coords, ~nan_mask.any(dim=1)

    @staticmethod
    def recenter_coordinates(coordinates):
        num_amino_acids, num_atoms, _ = coordinates.shape
        flattened_coordinates = coordinates.view(-1, 3)
        center_of_mass = flattened_coordinates.mean(dim=0)
        return coordinates - center_of_mass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_sequence = sample['seq'].replace('U', 'X').replace('O', 'X').replace('B', 'X').replace('Z', 'X')

        coords_list = torch.tensor(sample['coords'], dtype=torch.float32)
        coords_list, nan_mask = self.handle_nan_coordinates(coords_list)

        nan_mask = torch.cat([nan_mask, nan_mask.new_zeros(self.max_length)], dim=0)[:self.max_length]

        coords_list = self.recenter_coordinates(coords_list).tolist()

        sample_dict = {'name': sample['pid'], 'coords': coords_list, 'seq': raw_sequence}
        feature = self._featurize_as_graph(sample_dict)

        plddt_scores = torch.tensor(sample['plddt_scores'], dtype=torch.float16) / 100
        plddt_scores = plddt_scores[: self.max_length]
        if plddt_scores.shape[0] < self.max_length:
            pad_len = self.max_length - plddt_scores.shape[0]
            pad = torch.full((pad_len,), float("nan"), dtype=plddt_scores.dtype)
            plddt_scores = torch.cat([plddt_scores, pad], dim=0)

        coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
        coords_tensor = coords_tensor[:self.max_length, ...]
        coords_tensor = coords_tensor.reshape(1, -1, 12)

        coords, masks = merge_features_and_create_mask(coords_tensor, self.max_length)

        input_coordinates = coords.clone()
        coords = coords[..., :9]

        coords = coords.squeeze(0)
        masks = masks.squeeze(0)

        sample_weight = 1.0
        inverse_folding_labels = amino_acid_to_tensor(raw_sequence, self.max_length, self.letter_to_num)

        esm_input_ids = None
        esm_attention_mask = None

        return [
            feature,
            raw_sequence,
            plddt_scores,
            sample['pid'],
            coords,
            masks,
            input_coordinates,
            inverse_folding_labels,
            nan_mask,
            sample_weight,
            esm_input_ids,
            esm_attention_mask,
        ]

    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'], dtype=torch.float32)
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']], dtype=torch.long)

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            x_ca = coords[:, 1]

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(x_ca)
            sidechains = self._sidechains(coords)
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)

            node_s, node_v = map(torch.nan_to_num, (node_s, node_v))

        data = Data(
            x=x_ca,
            x_bb=coords[:, :3],
            seq=seq,
            name=name,
            h=node_s,
            chi=node_v,
            mask=mask,
        )
        return data

    @staticmethod
    def _dihedrals(x, eps=1e-7):
        x = torch.reshape(x[:, :3], [3 * x.shape[0], 3])
        dx = x[1:] - x[:-1]
        u = _normalize(dx, dim=-1)
        u_2 = u[:-2]
        u_1 = u[1:-1]
        u_0 = u[2:]

        n_2 = _normalize(torch.linalg.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.linalg.cross(u_1, u_0), dim=-1)

        cos_d = torch.sum(n_2 * n_1, -1)
        cos_d = torch.clamp(cos_d, -1 + eps, 1 - eps)
        d = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cos_d)

        d = F.pad(d, [1, 2])
        d = torch.reshape(d, [-1, 3])
        d_features = torch.cat([torch.cos(d), torch.sin(d)], 1)
        return d_features

    @staticmethod
    def _orientations(x):
        forward = _normalize(x[1:] - x[:-1])
        backward = _normalize(x[:-1] - x[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    @staticmethod
    def _sidechains(x):
        n, origin, c = x[:, 0], x[:, 1], x[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.linalg.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec
