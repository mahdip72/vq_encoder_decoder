import argparse
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import math
import h5py
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import Manager


def write_h5_file(file_path, pad_seq, n_ca_c_o_coord, plddt_scores):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('seq', data=pad_seq)
        f.create_dataset('N_CA_C_O_coord', data=n_ca_c_o_coord)
        f.create_dataset('plddt_scores', data=plddt_scores)


def extract_coordinates_plddt(chain, max_amino_acid_index, max_len):
    """Returns a list of N, C-alpha, C and O (backbone atoms) coordinates for a chain"""
    pos = []
    plddt_scores = []
    for row, residue in enumerate(chain):
        if row == max_amino_acid_index:
            break
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
    return pos, plddt_scores


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
        if len(sequence) >= 2:
            sequences[chain.id] = sequence
        else:
            report_dict['single_amino_acid'] += 1
    return sequences


def filter_best_chains(chain_sequences, structure):
    """
    Filters chains to retain only the unique sequences and selects the chain with the most resolved Cα atoms
    for each unique sequence.

    Args:
        chain_sequences (dict): Dictionary of chain IDs and their sequences.
        structure (Structure): Parsed structure from the PDB file.

    Returns:
        dict: Dictionary of unique sequences with the chain ID and the count of resolved Cα atoms.

    Example:
        Suppose you have the following chain sequences and counts of resolved Cα atoms:

        - Chain 'A': Sequence 'MKT' with 5 resolved Cα atoms.
        - Chain 'B': Sequence 'MKT' with 7 resolved Cα atoms.
        - Chain 'C': Sequence 'MKT' with 6 resolved Cα atoms.
        - Chain 'D': Sequence 'GVA' with 4 resolved Cα atoms.

        The `filter_best_chains` function would return:

        - 'MKT': ('B', 7)  # Chain 'B' has the most resolved Cα atoms for the sequence 'MKT'.
        - 'GVA': ('D', 4)  # Chain 'D' is the only chain with the sequence 'GVA'.
    """
    processed_chains = {}
    for chain_id, sequence in chain_sequences.items():
        model = structure[0]
        chain = model[chain_id]

        # Count the number of resolved Cα atoms
        ca_count = sum(1 for residue in chain if 'CA' in residue)
        if sequence in processed_chains:
            if ca_count > processed_chains[sequence][1]:
                processed_chains[sequence] = (chain_id, ca_count)
        else:
            processed_chains[sequence] = (chain_id, ca_count)

    return processed_chains


def preprocess_file(file_path, max_len, save_path, dictn, report_dict):
    parser = PDBParser(QUIET=True)  # This is from Bio.PDB
    structure = parser.get_structure('protein', file_path)

    # Extract chains and their sequences
    chain_sequences = check_chains(structure, report_dict)
    if len(chain_sequences) > 1:
        report_dict['protein_complex'] += 1

    if 'A' not in list(chain_sequences.keys()):
        report_dict['no_chain_id_a'] += 1

    # Filter the chains to retain only the best representative chains
    best_chains = filter_best_chains(chain_sequences, structure)

    for sequence, (chain_id, ca_count) in best_chains.items():
        if len(sequence) < 2:
            continue
        model = structure[0]
        chain = model[chain_id]

        protein_seq = ''
        max_amino_acid_index = 0
        for residue_index, residue in enumerate(chain):
            if residue.id[0] == ' ' and residue.resname in dictn:
                protein_seq += dictn[residue.resname]
                max_amino_acid_index = residue_index
            elif residue.id[0] == ' ' and residue.resname not in dictn:
                report_dict['wrong_amino_acid'] += 1
                break
            elif residue.id[0] != ' ':
                break
            else:
                pass

        n_ca_c_o_coord, plddt_scores = extract_coordinates_plddt(chain, max_amino_acid_index, max_len)
        pad_seq = protein_seq[:max_len]
        if len(chain_sequences) > 1:
            outputfile = os.path.join(save_path, os.path.splitext(os.path.basename(file_path))[0] + f"_chain_id_{chain_id}.h5")
        else:
            outputfile = os.path.join(save_path, os.path.splitext(os.path.basename(file_path))[0] + ".h5")
        report_dict['h5_processed'] += 1
        write_h5_file(outputfile, pad_seq, n_ca_c_o_coord, plddt_scores)


def main():
    parser = argparse.ArgumentParser(description='Processing PDB files.')
    parser.add_argument('--data', default='./test_data', help='Path to PDB files.')
    parser.add_argument('--max_len', default=1024, type=int, help='Max sequence length to consider.')
    parser.add_argument('--save_path', default='./save_test/', help='Path to output.')
    parser.add_argument('--max_workers', default=16,
                        help='Set the number of workers for parallel processing.')
    args = parser.parse_args()

    data_path = []
    for path in os.listdir(args.data):
        if os.path.isfile(os.path.join(args.data, path)) and path[-3:] == 'pdb':
            data_path.append(os.path.join(args.data, path))

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
                                    'wrong_amino_acid': 0, 'single_amino_acid': 0, 'error': 0})
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
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
