import argparse
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio import pairwise2
import math
import h5py
import glob
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import Manager


def find_pdb_files(directory_path):
    # Pattern to match PDB files in the directory and all subdirectories
    pdb_files_pattern = directory_path + '/**/*.pdb'

    # Find all PDB files matching the pattern
    pdb_files = glob.glob(pdb_files_pattern, recursive=True)

    return pdb_files


def find_structure_files(directory_path, use_cif):
    """
    Find structure files recursively.

    If use_cif is True, searches for .cif files.
    Otherwise, searches for .pdb files.
    """
    patterns = [directory_path + '/**/*.pdb']
    if use_cif:
        patterns = [directory_path + '/**/*.cif']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files


def write_h5_file(file_path, pad_seq, n_ca_c_o_coord, plddt_scores):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('seq', data=pad_seq)
        f.create_dataset('N_CA_C_O_coord', data=n_ca_c_o_coord)
        f.create_dataset('plddt_scores', data=plddt_scores)


def estimate_missing_from_distance(prev_ca_coord, next_ca_coord, ideal_ca_ca=3.8):
    """
    Estimate the number of missing residues between two residues using the
    straight-line distance between their CA atoms.

    The estimate assumes an average CA-CA distance of `ideal_ca_ca` angstroms.
    If the two residues were adjacent (no missing), distance ~ ideal_ca_ca,
    which yields ~0 missing residues. For a gap of N residues, distance is
    roughly (N+1) * ideal_ca_ca, so N ~= floor(distance / ideal_ca_ca) - 1.

    Returns an integer >= 0.
    """
    # Validate coords (any NaN -> cannot estimate)
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
    # Convert to missing count using floor; ensure non-negative
    est_missing = max(0, int(math.floor((dist / ideal_ca_ca)*1.2) - 1))
    return est_missing


def check_chains(structure, report_dict, min_len):
    """
    Extracts sequences from each chain and filters them by minimum length.

    This function processes each chain in the given PDB structure, extracts its sequence, and applies the following filter:
    1. Removes chains with sequences shorter than min_len amino acids.

    Args:
        structure (Bio.PDB.Structure.Structure): The PDB structure object to process.
        report_dict (multiprocessing.Manager.dict): Dictionary to log processing metrics.
        min_len (int): Minimum sequence length for a chain to be processed.

    Returns:
        dict: Mapping of chain IDs to their sequences that meet the min_len criterion.
    """
    ppb = PPBuilder()
    chains = [chain for model in structure for chain in model]
    sequences = {}
    for chain in chains:
        sequence = ''.join([str(pp.get_sequence()) for pp in ppb.build_peptides(chain)])
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
    """
    Filters chains to retain only the unique sequences and selects the chain with the most resolved Cα atoms
    for each unique sequence.

    This function processes the chain sequences from a PDB structure and retains only the chains with
    unique sequences or the best representative chain for similar sequences. The "best" chain is defined as
    the one with the highest number of resolved Cα atoms.

    The procedure is as follows:
    1. Initialize dictionaries to keep track of processed chains and the best representative chain for each sequence.
    2. Iterate through each chain in the structure:
        a. Retrieve the sequence for the current chain.
        b. Count the number of resolved Cα atoms in the current chain.
        c. Compare the current sequence with already processed sequences using a similarity threshold.
        d. If a similar sequence is found, compare the number of resolved Cα atoms:
            - If the current chain has more resolved Cα atoms than the existing one, update the best representative.
        e. If no similar sequence is found, add the current chain to the processed chains.
    3. Build and return the final dictionary of unique sequences with the chain ID and the count of resolved Cα atoms.

    Args:
        chain_sequences (dict): Dictionary where keys are chain IDs and values are their sequences.
        structure (Bio.PDB.Structure.Structure): The parsed PDB structure object containing the chains.
        similarity_threshold (float, optional): Threshold for considering two sequences as similar. Defaults to 0.95.

    Returns:
        dict: Dictionary of unique sequences with their best representative chain ID.

    Example:
        Suppose you have the following chain sequences and counts of resolved Cα atoms:
        - Chain 'A': Sequence 'MKT' with 5 resolved Cα atoms.
        - Chain 'B': Sequence 'MKT' with 7 resolved Cα atoms.
        - Chain 'C': Sequence 'MKV' with 6 resolved Cα atoms.
        - Chain 'D': Sequence 'GVA' with 4 resolved Cα atoms.

        With a similarity threshold of 0.95, the function would return:
        - 'MKT': ('B', 7)  # Chain 'B' has the most resolved Cα atoms for the similar sequences 'MKT' and 'MKV'.
        - 'GVA': ('D', 4)  # Chain 'D' is the only chain with the sequence 'GVA'.
    """
    processed_chains = {}
    sequence_to_chain = {}

    for chain_id, sequence in chain_sequences.items():
        model = structure[0]
        chain = model[chain_id]

        ca_count = sum(1 for residue in chain if 'CA' in residue)

        # Filter out similar sequences
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

    # swap keys and values
    processed_chains = {v[0]: (k, v[0]) for k, v in processed_chains.items()}
    return processed_chains


def evaluate_missing_content(pos, max_missing_ratio=0.2, max_consecutive_missing=15):
    """Return (is_valid, reason_key) based on missing residue statistics."""
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


def preprocess_file(file_index, file_path, max_len, min_len, save_path, dictn, report_dict, use_cif, no_file_index, gap_threshold):
    """
    Processes a structure file (PDB/mmCIF) into HDF5 format, handling insertion codes and numeric gaps.

    This function parses the PDB structure, filters chains below min_len or above max_len, and
    selects representative chains. It extracts residue sequences, backbone coordinates, and pLDDT scores
    for all present residues, handles insertion codes naturally, and post-processes numeric gaps by
    inserting 'X' and NaN paddings. Finally, it writes the data to HDF5 files.

    Args:
        file_index (int): Index of the file for naming outputs.
        file_path (str): Path to the input structure file.
        max_len (int): Maximum sequence length allowed for processing.
        min_len (int): Minimum sequence length required for processing.
        save_path (str): Directory to save output HDF5 files.
        dictn (dict): Mapping from three-letter to one-letter amino acid codes.
        report_dict (multiprocessing.Manager.dict): Dictionary for logging processing statistics.
    """
    parser = MMCIFParser(QUIET=True) if use_cif else PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)

    chain_sequences = check_chains(structure, report_dict, min_len)

    best_chains = filter_best_chains(chain_sequences, structure)

    if len(best_chains) > 1:
        report_dict['protein_complex'] += 1
    if 'A' not in list(best_chains.keys()):
        report_dict['no_chain_id_a'] += 1

    for chain_id, sequence in best_chains.items():
        model = structure[0]
        chain = model[chain_id]

        # New, robust direct residue iteration logic
        # Get a list of all residues directly from the chain (includes insertion codes)
        residues = [res for res in chain if res.id[0] == ' ']
        # Skip if empty; max_len will be enforced after gap handling
        if not residues:
            continue

        protein_seq = ''
        pos = []
        plddt_scores = []
        # Iterate directly over residues
        for residue in residues:
            # assign standard residue code or 'X' for non-standard
            if residue.resname in dictn:
                protein_seq += dictn[residue.resname]
            else:
                protein_seq += 'X'
            # pLDDT score
            try:
                plddt_scores.append(residue['CA'].get_bfactor())
            except KeyError:
                plddt_scores.append(math.nan)
            # backbone coordinates
            coords = []
            for key in ['N', 'CA', 'C', 'O']:
                if key in residue:
                    coords.append(list(residue[key].coord))
                else:
                    coords.append([math.nan, math.nan, math.nan])
            pos.append(coords)

        # --- Gap handling ---
        # Apply numeric gap handling for both PDB and CIF inputs. For CIF, residue numbering
        # (auth_seq_id) can be non-contiguous; large gaps are reduced using CA-CA distance
        # to avoid runaway padding.
        for i in range(len(residues) - 1, 0, -1):
            current_res_id = residues[i].id
            prev_res_id = residues[i-1].id
            if current_res_id[1] > prev_res_id[1] + 1:
                numeric_gap_size = current_res_id[1] - prev_res_id[1] - 1

                insert_count = numeric_gap_size
                if numeric_gap_size > gap_threshold:
                    # Estimate from CA-CA distance
                    prev_ca = pos[i-1][1]
                    next_ca = pos[i][1]
                    est_missing = estimate_missing_from_distance(prev_ca, next_ca)
                    if est_missing is not None:
                        insert_count = min(numeric_gap_size, est_missing)
                    else:
                        # Fallback: clamp to threshold to avoid runaway padding
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
        # --- END Gap handling ---
        # Enforce final length constraints before writing
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
        if len(best_chains) > 1:
            if no_file_index:
                outputfile = os.path.join(save_path, f"{basename}_chain_id_{chain_id}.h5")
            else:
                outputfile = os.path.join(save_path, f"{file_index}_{basename}_chain_id_{chain_id}.h5")
        else:
            if no_file_index:
                outputfile = os.path.join(save_path, f"{basename}.h5")
            else:
                outputfile = os.path.join(save_path, f"{file_index}_{basename}.h5")
        report_dict['h5_processed'] += 1
        write_h5_file(outputfile, protein_seq, pos, plddt_scores)


def main():
    parser = argparse.ArgumentParser(description='Processing PDB files.')
    parser.add_argument('--data', default='./test_data', help='Path to PDB files.')
    parser.add_argument('--max_len', default=2048, type=int, help='Max sequence length to consider.')
    parser.add_argument('--save_path', default='./save_test/', help='Path to output.')
    parser.add_argument('--max_workers', default=16, type=int,
                        help='Set the number of workers for parallel processing.')
    parser.add_argument('--min_len', default=25, type=int,
                        help='Minimum sequence length for chains to process.')
    parser.add_argument('--use_cif', action='store_true',
                        help='Use CIF/mmCIF input instead of PDB (default: PDB).')
    parser.add_argument('--no_file_index', action='store_true',
                        help='Omit file index prefix in output filenames.')
    parser.add_argument('--gap_threshold', default=5, type=int,
                        help='For both PDB and CIF: if a numeric residue-numbering gap exceeds this value, '
                             'reduce the inserted missing residues to an estimate based on CA-CA distance.')
    args = parser.parse_args()

    data_path = find_structure_files(args.data, args.use_cif)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dictn = {
        'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
        'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
        'ASX': 'B', 'GLX': 'Z', 'PYL': 'O', 'SEC': 'U',  # 'UNK': 'X'
    }

    with Manager() as manager:
        report_dict = manager.dict({'protein_complex': 0, 'no_chain_id_a': 0, 'h5_processed': 0,
                                    'chains_too_short': 0, 'chains_too_long': 0, 'error': 0, 'missing_residues': 0,
                                    'missing_ratio_exceeded': 0, 'missing_block_exceeded': 0})
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(preprocess_file, i, file_path, args.max_len, args.min_len, args.save_path, dictn, report_dict, args.use_cif, args.no_file_index, args.gap_threshold): file_path for i, file_path in enumerate(data_path)}
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
