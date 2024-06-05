import argparse
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
import math
import h5py
from concurrent.futures import as_completed, ProcessPoolExecutor


def write_h5_file(file_path, pad_seq, n_ca_c_o_coord, plddt_scores):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('seq', data=pad_seq)
        f.create_dataset('N_CA_C_O_coord', data=n_ca_c_o_coord)
        f.create_dataset('plddt_scores', data=plddt_scores)


def extract_coordinates_plddt(chain, max_len):
    """Returns a list of C-alpha coordinates for a chain"""
    pos = []
    plddt_scores = []
    for row, residue in enumerate(chain):
        pos_N_CA_C_O = []
        plddt_scores.append(residue["CA"].get_bfactor())
        for key in ['N', 'CA', 'C', 'O']:
            if key in residue:
                pos_N_CA_C_O.append(list(residue[key].coord))
            else:
                pos_N_CA_C_O.append([math.nan, math.nan, math.nan])

        pos.append(pos_N_CA_C_O)
        if len(pos) == max_len:
            break

    return pos, plddt_scores


def preprocess_file(file_path, max_len, save_path, dictn):
    parser = PDBParser()  # This is from Bio.PDB
    structure = parser.get_structure('protein', file_path)
    model = structure[0]
    sequence = []
    chain = model['A']
    protein_seq = ''
    for residue_index, residue in enumerate(chain):
        sequence.append(residue.resname)
        amino_acid = [dictn[triple] for triple in sequence]
        protein_seq = "".join(amino_acid)

    n_ca_c_o_coord, plddt_scores = extract_coordinates_plddt(chain, max_len)
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

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(preprocess_file, file_path, args.max_len, args.save_path, dictn): file_path for
                   file_path in data_path}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"An error occurred while processing {file_path}: {exc}")


if __name__ == '__main__':
    main()
