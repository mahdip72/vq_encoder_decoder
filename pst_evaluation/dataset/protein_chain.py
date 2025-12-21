import numpy as np

import biotite.structure as bs
from biotite.structure.io.pdbx import CIFFile, convert
from Bio.Data import PDBData


class ProteinChain:
    def __init__(self, sequence, residue_index, chain_id):
        self.sequence = sequence
        self.residue_index = residue_index
        self.chain_id = chain_id

    def __len__(self):
        return len(self.sequence)

    @classmethod
    def from_cif(cls, path, chain_id="detect"):
        atom_array = convert.get_structure(
            CIFFile.read(path),
            model=1,
            extra_fields=["b_factor"],
        )
        if chain_id == "detect":
            chain_id = atom_array.chain_id[0]
        if not (atom_array.chain_id == chain_id).any():
            atom_array = convert.get_structure(
                CIFFile.read(path),
                model=1,
                extra_fields=["b_factor"],
                use_author_fields=False,
            )

        atom_array = atom_array[
            bs.filter_amino_acids(atom_array)
            & ~atom_array.hetero
            & (atom_array.chain_id == chain_id)
        ]

        sequence = "".join(
            (
                r
                if len(r := PDBData.protein_letters_3to1.get(m[0].res_name, "X")) == 1
                else "X"
            )
            for m in bs.residue_iter(atom_array)
        )
        residue_index = np.full([len(sequence)], -1, dtype=np.int64)

        for i, res in enumerate(bs.residue_iter(atom_array)):
            residue_index[i] = res[0].res_id

        return cls(
            sequence=sequence,
            residue_index=residue_index,
            chain_id=chain_id,
        )
