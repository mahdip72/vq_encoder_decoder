import math
import h5py


def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        plddt_scores = f['plddt_scores'][:]
    return seq, n_ca_c_o_coord, plddt_scores


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


if __name__ == '__main__':
    print('for test')
    prot_seq, coords, scores = load_h5_file('./save_test/AF-G5EB01-F1-model_v4.h5')
    print(prot_seq, coords)
