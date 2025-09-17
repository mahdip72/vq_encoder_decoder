import argparse
import glob
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio import pairwise2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.utils import write_chain_to_pdb


def find_structure_files(directory_path, use_cif):
    patterns = [os.path.join(directory_path, "**", "*.pdb")]
    if use_cif:
        patterns = [os.path.join(directory_path, "**", "*.cif")]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files


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


def check_chains(structure, report_dict, min_len):
    ppb = PPBuilder()
    chains = [chain for model in structure for chain in model]
    sequences = {}
    for chain in chains:
        sequence = ''.join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))
        if len(sequence) < min_len:
            report_dict['chains_too_short'] += 1
            continue
        sequences[chain.id] = sequence
    return sequences


def sequence_similarity(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = alignments[0]
    similarity = best_alignment[2] / min(len(seq1), len(seq2))
    return similarity


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
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 0
    if longest_run > max_consecutive_missing:
        return False, 'missing_block_exceeded'

    return True, ''


def preprocess_file(file_index, file_path, max_len, min_len, save_path, dictn, report_dict, use_cif, with_file_index, gap_threshold):
    parser = MMCIFParser(QUIET=True, auth_chains=False) if use_cif else PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)

    chain_sequences = check_chains(structure, report_dict, min_len)

    had_multichain_pre_dedup = len(chain_sequences) > 1
    if had_multichain_pre_dedup:
        report_dict['protein_complex_prededup'] += 1

    best_chains = filter_best_chains(chain_sequences, structure)

    if len(best_chains) > 1:
        report_dict['protein_complex'] += 1
    if 'A' not in list(best_chains.keys()):
        report_dict['no_chain_id_a'] += 1

    for chain_id in best_chains.keys():
        model = structure[0]
        chain = model[chain_id]

        residues = [res for res in chain if res.id[0] == ' ']
        if not residues:
            continue

        protein_seq = ''
        pos = []
        plddt_scores = []
        for residue in residues:
            protein_seq += dictn.get(residue.resname, 'X')
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
                if numeric_gap_size > gap_threshold:
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
                report_dict['missing_residues'] += insert_count

        final_len = len(protein_seq)
        if final_len < min_len:
            report_dict['chains_too_short'] += 1
            continue
        if final_len > max_len:
            report_dict['chains_too_long'] += 1
            continue

        is_valid, reason = evaluate_missing_content(pos)
        if not is_valid:
            report_dict[reason] += 1
            continue

        basename = os.path.splitext(os.path.basename(file_path))[0]
        chain_suffix = f"chain_id_{chain_id}"
        if with_file_index:
            outputfile = os.path.join(save_path, f"{file_index}_{basename}_{chain_suffix}.pdb")
        else:
            outputfile = os.path.join(save_path, f"{basename}_{chain_suffix}.pdb")

        write_chain_to_pdb(structure, chain_id, outputfile, model_id=0, include_hetero=False, output_chain_id="A")
        report_dict['pdb_written'] += 1


def main():
    parser = argparse.ArgumentParser(description='Split complexes into monomeric PDB files.')
    parser.add_argument('--data', default='./test_data', help='Path to structure files.')
    parser.add_argument('--max_len', default=2048, type=int, help='Max sequence length to consider.')
    parser.add_argument('--save_path', default='./save_test/', help='Path to output PDB files.')
    parser.add_argument('--max_workers', default=16, type=int,
                        help='Number of workers for parallel processing.')
    parser.add_argument('--min_len', default=25, type=int,
                        help='Minimum sequence length for chains to process.')
    parser.add_argument('--use_cif', action='store_true',
                        help='Use CIF/mmCIF input instead of PDB.')
    parser.add_argument('--with_file_index', action='store_true',
                        help='Include file index prefix in output filenames.')
    parser.add_argument('--gap_threshold', default=5, type=int,
                        help='Numeric residue-numbering gaps larger than this are reduced using CA distance estimates.')
    args = parser.parse_args()

    data_path = find_structure_files(args.data, args.use_cif)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dictn = {
        'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
        'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
        'ASX': 'B', 'GLX': 'Z', 'PYL': 'O', 'SEC': 'U',
    }

    with Manager() as manager:
        report_dict = manager.dict({
            'protein_complex': 0,
            'protein_complex_prededup': 0,
            'no_chain_id_a': 0,
            'pdb_written': 0,
            'chains_too_short': 0,
            'chains_too_long': 0,
            'error': 0,
            'missing_residues': 0,
            'missing_ratio_exceeded': 0,
            'missing_block_exceeded': 0,
        })
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    preprocess_file,
                    i,
                    file_path,
                    args.max_len,
                    args.min_len,
                    args.save_path,
                    dictn,
                    report_dict,
                    args.use_cif,
                    args.with_file_index,
                    args.gap_threshold,
                ): file_path for i, file_path in enumerate(data_path)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred while processing {file_path}: {exc} {type(exc)}")
                    report_dict['error'] += 1
        print(dict(report_dict))


if __name__ == '__main__':
    main()
