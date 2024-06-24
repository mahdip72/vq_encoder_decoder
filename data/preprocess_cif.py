import argparse
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

from Bio.PDB.Polypeptide import PPBuilder
from Bio import pairwise2
import math
import h5py
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import Manager


def write_h5_file(file_path, pad_seq, n_ca_c_o_coord, plddt_scores):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('seq', data=pad_seq)
        f.create_dataset('N_CA_C_O_coord', data=n_ca_c_o_coord)
        f.create_dataset('plddt_scores', data=plddt_scores)


def check_chains(structure, report_dict):
    """
    Extracts sequences from each chain in the given structure and filters them based on criteria.

    This function processes each chain in the given PDB structure, extracts its sequence, and applies the following filter:
    1. Removes chains with sequences consisting of fewer than 2 amino acids.

    Args:
        structure (Bio.PDB.Structure.Structure): The PDB structure object to process.

    Returns:
        dict: A dictionary with chain IDs as keys and their corresponding sequences as values,
              filtered based on the specified criteria.
    """
    ppb = PPBuilder()
    chains = [chain for model in structure for chain in model]
    sequences = {}
    for chain in chains:
        sequence = ''.join([str(pp.get_sequence()) for pp in ppb.build_peptides(chain)])
        if len(sequence) < 2:
            report_dict['single_amino_acid'] += 1
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


def preprocess_file(file_path, max_len, save_path, dictn, report_dict):
    #parser = PDBParser(QUIET=True)
    parser = MMCIFParser()
    structure = parser.get_structure('protein', file_path)

    chain_sequences = check_chains(structure, report_dict)

    best_chains = filter_best_chains(chain_sequences, structure)

    if len(best_chains) > 1:
        report_dict['protein_complex'] += 1
    if 'A' not in list(best_chains.keys()):
        report_dict['no_chain_id_a'] += 1

    for chain_id, sequence in best_chains.items():
        model = structure[0]
        chain = model[chain_id]

        protein_seq = ''
        pos = []
        plddt_scores = []
        has_ambiguous_residue = False

        for residue_index, residue in enumerate(chain):
            if residue.id[0] == ' ' and residue.resname in dictn:
                if residue.resname not in dictn:
                    has_ambiguous_residue = True
                    break

                protein_seq += dictn[residue.resname]

                pos_N_CA_C_O = []
                try:
                    plddt_scores.append(residue["CA"].get_bfactor())
                except KeyError:
                    plddt_scores.append(0)

                for key in ['N', 'CA', 'C', 'O']:
                    if key in residue:
                        pos_N_CA_C_O.append(list(residue[key].coord))
                    else:
                        pos_N_CA_C_O.append([math.nan, math.nan, math.nan])
                pos.append(pos_N_CA_C_O)

                if len(pos) == max_len:
                    break
            elif residue.id[0] != ' ':
                continue

        if has_ambiguous_residue:
            continue

        pad_seq = protein_seq[:max_len]
        if len(best_chains) > 1:
            outputfile = os.path.join(save_path, os.path.splitext(os.path.basename(file_path))[0] + f"_chain_id_{chain_id}.h5")
        else:
            outputfile = os.path.join(save_path, os.path.splitext(os.path.basename(file_path))[0] + ".h5")
        report_dict['h5_processed'] += 1
        write_h5_file(outputfile, pad_seq, pos, plddt_scores)


def main():
    parser = argparse.ArgumentParser(description='Processing PDB files.')
    parser.add_argument('--data', default='./test_data', help='Path to PDB files.')
    parser.add_argument('--max_len', default=1024, type=int, help='Max sequence length to consider.')
    parser.add_argument('--save_path', default='./save_test/', help='Path to output.')
    parser.add_argument('--max_workers', default=16,type=int,
                        help='Set the number of workers for parallel processing.')
    args = parser.parse_args()

    data_path = [os.path.join(args.data, path) for path in os.listdir(args.data) if os.path.isfile(os.path.join(args.data, path)) and path.endswith('cif')]
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
                                    'single_amino_acid': 0, 'error': 0})
        with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
            futures = {executor.submit(preprocess_file, file_path, args.max_len, args.save_path, dictn, report_dict): file_path for file_path in data_path}
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


