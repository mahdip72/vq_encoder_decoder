import argparse
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
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


def write_h5_file(file_path, pad_seq, n_ca_c_o_coord, plddt_scores):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('seq', data=pad_seq)
        f.create_dataset('N_CA_C_O_coord', data=n_ca_c_o_coord)
        f.create_dataset('plddt_scores', data=plddt_scores)


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


def filter_best_chains(chain_sequences, structure, similarity_threshold=0.95):
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


def preprocess_file(file_index, file_path, max_len, min_len, save_path, dictn, report_dict):
    """
    Processes a PDB file into HDF5 format, filtering and iterating chains by length.

    This function parses the PDB structure, filters chains below min_len or above max_len,
    selects representative chains, extracts residue sequences, backbone coordinates, and pLDDT scores,
    then writes the data to HDF5 files.

    Args:
        file_index (int): Index of the file for naming outputs.
        file_path (str): Path to the input PDB file.
        max_len (int): Maximum sequence length allowed for processing.
        min_len (int): Minimum sequence length required for processing.
        save_path (str): Directory to save output HDF5 files.
        dictn (dict): Mapping from three-letter to one-letter amino acid codes.
        report_dict (multiprocessing.Manager.dict): Dictionary for logging processing statistics.
    """
    parser = PDBParser(QUIET=True)
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
        # Skip if empty or exceeds max length
        if not residues or len(residues) > max_len:
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

        basename = os.path.splitext(os.path.basename(file_path))[0]
        if len(best_chains) > 1:
            outputfile = os.path.join(save_path, f"{file_index}_{basename}_chain_id_{chain_id}.h5")
        else:
            outputfile = os.path.join(save_path, f"{file_index}_{basename}.h5")
        # count missing (padded) residues marked as 'X'
        missing_count = protein_seq.count('X')
        report_dict['missing_residues'] = report_dict['missing_residues'] + missing_count
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
    args = parser.parse_args()

    data_path = find_pdb_files(args.data)
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
                                    'chains_too_short': 0, 'error': 0, 'missing_residues': 0})
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(preprocess_file, i, file_path, args.max_len, args.min_len, args.save_path, dictn, report_dict): file_path for i, file_path in enumerate(data_path)}
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
