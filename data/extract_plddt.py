import glob
import argparse
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
import pandas as pd
from concurrent.futures import as_completed, ProcessPoolExecutor


dictn = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    'ASX': 'B', 'GLX': 'Z', 'PYL': 'O', 'SEC': 'U',  # 'UNK': 'X'
}


def find_pdb_files(directory_path):
    # Pattern to match PDB files in the directory and all subdirectories
    pdb_files_pattern = directory_path + '/**/*.pdb'

    # Find all PDB files matching the pattern
    pdb_files = glob.glob(pdb_files_pattern, recursive=True)

    return pdb_files


def preprocess_file(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    model = structure[0]  # Assuming there's only one model
    chain = model['A']  # As specified, only chain A is present

    plddt_scores = []
    sequence = []
    for residue in chain:
        if residue.id[0] == ' ' and residue.resname:
            sequence.append(dictn[residue.resname])
            try:
                plddt_score = residue["CA"].get_bfactor()
                plddt_scores.append(str(plddt_score))
            except KeyError:
                continue  # Skip residues without CA atom
    pdb_directory, subdirectory = os.path.split(os.path.dirname(file_path))

    data = [{
        # 'pdb_directory': file_path,
        # 'subdirectory': subdirectory,
        "prot_id": os.path.basename(file_path).split(".")[0],
        "sequence": ''.join(sequence),
        # 'plddt': round(sum(plddt_scores) / len(plddt_scores), 2)
        'plddt': ','.join(plddt_scores)
    }
    ]
    return data


def main():
    parser = argparse.ArgumentParser(description='Processing PDB files.')
    parser.add_argument('--data', default='/mnt/hdd8/mehdi/datasets/vqvae/unirpot_swissprot_pdbs/test_case_b',
                        help='Path to PDB files.')
    parser.add_argument('--save_path', default='./save_test/', help='Path to output.')
    parser.add_argument('--max_workers', default=64, type=int,
                        help='Set the number of workers for parallel processing.')
    args = parser.parse_args()

    data_path = find_pdb_files(args.data)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    aggregated_data = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(preprocess_file, file_path): file_path for file_path in
                   data_path}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                result = future.result()
                aggregated_data.extend(result)
            except Exception as exc:
                file_path = futures[future]
                print(f"An error occurred while processing {file_path}: {exc} {type(exc)}")

    # Convert aggregated data to DataFrame
    df = pd.DataFrame(aggregated_data)

    # Save DataFrame to CSV
    csv_file_path = os.path.join(args.save_path, 'test_set_b.csv')
    df.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    main()
