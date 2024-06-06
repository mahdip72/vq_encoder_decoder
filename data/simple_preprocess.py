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


def check_single_protein_chain(structure):
    ppb = PPBuilder()
    chains = [chain for model in structure for chain in model]
    sequences = [pp.get_sequence() for chain in chains for pp in ppb.build_peptides(chain)]
    return len(sequences) == 1, chains[0].id if sequences else None


def preprocess_file(file_path, max_len, save_path, dictn, report_dict):
    if os.path.basename(file_path) == '1H7M.pdb':
        pass

    parser = PDBParser(QUIET=True)  # This is from Bio.PDB
    structure = parser.get_structure('protein', file_path)

    # Check for a single protein chain
    single_chain, chain_id = check_single_protein_chain(structure)
    if not single_chain:
        report_dict['protein_complex'] += 1
        return
    if chain_id != 'A':
        report_dict['no_chain_id_a'] += 1

    model = structure[0]
    chain = model[chain_id]

    sequence = []
    protein_seq = ''
    max_amino_acid_index = 0
    for residue_index, residue in enumerate(chain):
        if residue_index == 95:
            pass
        if residue.id[0] == ' ' and residue.resname in dictn:
            sequence.append(residue.resname)
            amino_acid = [dictn[triple] for triple in sequence]
            protein_seq = "".join(amino_acid)
            max_amino_acid_index = residue_index
        elif residue.id[0] != ' ':
            break
        else:
            pass

    n_ca_c_o_coord, plddt_scores = extract_coordinates_plddt(chain, max_amino_acid_index, max_len)
    pad_seq = protein_seq[:max_len]
    outputfile = os.path.join(save_path, os.path.splitext(os.path.basename(file_path))[0] + ".h5")
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

    dictn = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    with Manager() as manager:
        report_dict = manager.dict({'protein_complex': 0, 'no_chain_id_a': 0})
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(preprocess_file, file_path, args.max_len, args.save_path, dictn, report_dict): file_path for file_path in data_path}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred while processing {file_path}: {exc} {type(exc)}")
        print(dict(report_dict))


if __name__ == '__main__':
    main()
