import glob
import math
import os
from collections import Counter

from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio import pairwise2

DEFAULT_AA_MAP = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    'ASX': 'B', 'GLX': 'Z', 'PYL': 'O', 'SEC': 'U',
}


def find_structure_files(directory_path):
    patterns = [
        os.path.join(directory_path, '**', '*.pdb'),
        os.path.join(directory_path, '**', '*.cif'),
        os.path.join(directory_path, '**', '*.mmcif'),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files


def _select_parser(file_path):
    lower_path = file_path.lower()
    if lower_path.endswith(('.cif', '.mmcif')):
        return MMCIFParser(QUIET=True, auth_chains=False)
    return PDBParser(QUIET=True)


def sequence_similarity(seq1, seq2):
    alignment = pairwise2.align.globalxx(seq1, seq2)[0]
    return alignment[2] / min(len(seq1), len(seq2))


def check_chains(structure, stats, min_len):
    ppb = PPBuilder()
    chains = [chain for model in structure for chain in model]
    sequences = {}
    for chain in chains:
        sequence = ''.join([str(pp.get_sequence()) for pp in ppb.build_peptides(chain)])
        if len(sequence) < min_len:
            stats['chains_too_short'] += 1
            continue
        sequences[chain.id] = sequence
    return sequences


def filter_best_chains(chain_sequences, structure, similarity_threshold=0.90):
    processed_chains = {}
    sequence_to_chain = {}

    for chain_id, sequence in chain_sequences.items():
        model = structure[0]
        chain = model[chain_id]

        ca_count = sum(1 for residue in chain if 'CA' in residue)

        is_similar = False
        for existing_sequence in sequence_to_chain.keys():
            if sequence_similarity(sequence, existing_sequence) > similarity_threshold:
                is_similar = True
                existing_chain_id, existing_ca_count = sequence_to_chain[existing_sequence]
                if ca_count > existing_ca_count:
                    sequence_to_chain[existing_sequence] = (chain_id, ca_count)
                break

        if not is_similar:
            sequence_to_chain[sequence] = (chain_id, ca_count)

    for sequence, (chain_id, ca_count) in sequence_to_chain.items():
        processed_chains[sequence] = (chain_id, ca_count)

    processed_chains = {v[0]: (k, v[0]) for k, v in processed_chains.items()}
    return processed_chains


def estimate_missing_from_distance(prev_ca_coord, next_ca_coord, ideal_ca_ca=3.8):
    try:
        x1, y1, z1 = prev_ca_coord
        x2, y2, z2 = next_ca_coord
        if any(math.isnan(v) for v in (x1, y1, z1, x2, y2, z2)):
            return None
    except Exception:
        return None

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    est_missing = max(0, int(math.floor((dist / ideal_ca_ca) * 1.2) - 1))
    return est_missing


def evaluate_missing_content(pos, max_missing_ratio=0.2, max_consecutive_missing=15):
    total = len(pos)
    if total == 0:
        return False, 'missing_ratio_exceeded'

    missing_flags = []
    for residue in pos:
        ca_coords = residue[1] if len(residue) > 1 else []
        if len(ca_coords) != 3:
            missing_flags.append(True)
            continue
        missing_flags.append(any(math.isnan(v) for v in ca_coords))

    missing_count = sum(missing_flags)
    if missing_count / total > max_missing_ratio:
        return False, 'missing_ratio_exceeded'

    longest_run = 0
    current_run = 0
    for is_missing in missing_flags:
        if is_missing:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    if longest_run > max_consecutive_missing:
        return False, 'missing_block_exceeded'

    return True, ''


def propagate_nan_residues(pos):
    updated_count = 0
    for i, residue_coords in enumerate(pos):
        is_fully_nan = True
        has_any_missing = False
        for atom_coords in residue_coords:
            if len(atom_coords) != 3:
                has_any_missing = True
                continue
            any_nan = any(math.isnan(v) for v in atom_coords)
            if any_nan:
                has_any_missing = True
            else:
                is_fully_nan = False
        if has_any_missing and not is_fully_nan:
            pos[i] = [[math.nan, math.nan, math.nan] for _ in range(4)]
            updated_count += 1
    return updated_count


def process_structure_file(
    file_index,
    file_path,
    *,
    max_len,
    min_len,
    similarity_threshold,
    gap_threshold,
    use_gap_estimation,
    max_missing_ratio,
    max_consecutive_missing,
    include_file_index,
    aa_map=None,
):
    stats = Counter()
    aa_map = aa_map or DEFAULT_AA_MAP

    parser = _select_parser(file_path)
    try:
        structure = parser.get_structure('protein', file_path)
    except Exception:
        fallback = MMCIFParser(QUIET=True, auth_chains=False) if isinstance(parser, PDBParser) else PDBParser(QUIET=True)
        structure = fallback.get_structure('protein', file_path)

    chain_sequences = check_chains(structure, stats, min_len)

    if len(chain_sequences) > 1:
        stats['protein_complex_prededup'] += 1

    best_chains = filter_best_chains(chain_sequences, structure, similarity_threshold)

    if len(best_chains) > 1:
        stats['protein_complex'] += 1
    if 'A' not in list(best_chains.keys()):
        stats['no_chain_id_a'] += 1

    samples = []
    for chain_id, sequence in best_chains.items():
        model = structure[0]
        chain = model[chain_id]

        residues = [res for res in chain if res.id[0] == ' ']
        if not residues:
            continue

        protein_seq = ''
        pos = []
        plddt_scores = []
        for residue in residues:
            if residue.resname in aa_map:
                protein_seq += aa_map[residue.resname]
            else:
                protein_seq += 'X'
            try:
                plddt_scores.append(residue['CA'].get_bfactor())
            except KeyError:
                plddt_scores.append(math.nan)

            coords = []
            for key in ['N', 'CA', 'C', 'O']:
                if key in residue:
                    coords.append(list(residue[key].coord))
                else:
                    coords.append([math.nan, math.nan, math.nan])
            pos.append(coords)

        for i in range(len(residues) - 1, 0, -1):
            current_res_id = residues[i].id
            prev_res_id = residues[i - 1].id
            if current_res_id[1] > prev_res_id[1] + 1:
                numeric_gap_size = current_res_id[1] - prev_res_id[1] - 1

                insert_count = numeric_gap_size
                if use_gap_estimation and numeric_gap_size > gap_threshold:
                    prev_ca = pos[i - 1][1]
                    next_ca = pos[i][1]
                    est_missing = estimate_missing_from_distance(prev_ca, next_ca)
                    if est_missing is not None:
                        insert_count = min(numeric_gap_size, est_missing)
                    else:
                        insert_count = gap_threshold

                if insert_count <= 0:
                    continue

                x_padding = 'X' * insert_count
                nan_coord_padding = [[math.nan, math.nan, math.nan] for _ in range(4)]
                nan_plddt_padding = [math.nan] * insert_count
                nan_pos_padding = [nan_coord_padding] * insert_count
                protein_seq = protein_seq[:i] + x_padding + protein_seq[i:]
                pos[i:i] = nan_pos_padding
                plddt_scores[i:i] = nan_plddt_padding
                stats['missing_residues'] += insert_count

        nan_residue_count = propagate_nan_residues(pos)
        if nan_residue_count > 0:
            stats['missing_coordinates'] += nan_residue_count

        final_len = len(protein_seq)
        if final_len < min_len:
            stats['chains_too_short'] += 1
            continue
        if final_len > max_len:
            stats['chains_too_long'] += 1
            continue

        is_valid, reason = evaluate_missing_content(
            pos,
            max_missing_ratio=max_missing_ratio,
            max_consecutive_missing=max_consecutive_missing,
        )
        if not is_valid:
            stats[reason] += 1
            continue

        basename = os.path.splitext(os.path.basename(file_path))[0]
        if len(best_chains) > 1:
            pid = f"{basename}_chain_id_{chain_id}"
        else:
            pid = basename
        if include_file_index:
            pid = f"{file_index}_{pid}"

        stats['samples_processed'] += 1
        samples.append(
            {
                'pid': pid,
                'seq': protein_seq,
                'coords': pos,
                'plddt_scores': plddt_scores,
                'source_path': file_path,
            }
        )

    return samples, stats
