import torch
from typing import Tuple, NewType, Union
from jaxtyping import Float
from torch import Tensor

# Positions
AtomTensor = NewType("AtomTensor", Float[Tensor, "residues 37 3"])
"""
``torch.float[-1, 37, 3]``

Tensor of atom coordinates. The first dimension is the length of the protein,
the second the number of canonical atom types. The last dimension contains the
x,y,z positions.

.. seealso:: :class:`ResidueTensor` :class:`CoordTensor`
"""

BackboneFrameTensor = NewType(
    "BackboneFrameTensor", Float[Tensor, "residues 3 3"]
)
"""
``torch.float[-1, 3, 3]``

Tensor of backbone frames as rotation matrices. The first dimension is the
length of the protein, the second and third dimensions specify a rotation matrix
relative to an idealised residue.

.. seealso::

    :meth:`graphein.protein.tensor.reconstruction.get_ideal_backbone_coords`
    :meth:`graphein.protein.tensor.representation.get_backbone_frames`
"""

BackboneTensor = NewType("BackboneTensor", Float[Tensor, "residues 4 3"])
"""
``torch.float[-1, 4, 3]``

Tensor of backbone atomic coordinates. The first dimension is the length of the
protein (or batch), the second dimension corresponds to ``[N, Ca, C, O]`` atoms
and the last dimension contains the x,y,z coordinates.

.. seealso:: :ref:`graphein.protein.tensor.types.AtomTensor

"""

CoordTensor = NewType("CoordTensor", Float[Tensor, "nodes 3"])
"""
``torch.float[-1, 3]``

Tensor of coordinates. The first dimension is the length of the protein
(or batch of proteins), the last dimension contains the x,y,z positions."""

RotationMatrix2D = NewType("RotationMatrix2D", Float[Tensor, "2 2"])
"""
``torch.float[2, 2]``

Specifies a 2D rotation matrix.

.. seealso:: :class:`RotationMatrix3D` :class:`RotationMatrix` :class:`RotationTensor`
"""

RotationMatrix3D = NewType("RotationMatrix3D", Float[Tensor, "3 3"])
"""
``torch.float[3, 3]``

Specifies a 3D rotation matrix.

.. seealso:: :class:`RotationMatrix2D` :class:`RotationMatrix` :class:`RotationTensor`
"""

RotationMatrix = NewType(
    "RotationMatrix", Union[RotationMatrix2D, RotationMatrix3D]
)
"""
``Union[RotationMatrix3D, RotationMatrix2D]``

Specifies a rotation matrix in either 2D or 3D.

.. seealso:: :class:`RotationMatrix2D` :class:`RotationMatrix3D` :class:`RotationTensor`
"""


def get_full_atom_coords(
        atom_tensor: AtomTensor, fill_value: float = 1e-5
) -> Tuple[CoordTensor, torch.Tensor, torch.Tensor]:
    """Converts an ``AtomTensor`` to a full atom representation.

    Return tuple of coords ``(N_atoms x 3)``, residue_index ``(N_atoms)``,
    atom_type ``(N_atoms x [0-36])``


    :param atom_tensor: AtomTensor of shape
        ``(N_residues, N_atoms (default is 37), 3)``
    :type atom_tensor: graphein.protein.tensor.AtomTensor
    :param fill_value: Value used to fill missing values. Defaults to ``1e-5``.
    :return: Tuple of coords, residue_index, atom_type
    :rtype: Tuple[CoordTensor, torch.Tensor, torch.Tensor]
    """
    # Get number of atoms per residue
    filled = atom_tensor[:, :, 0] != fill_value
    nz = filled.nonzero()
    residue_index = nz[:, 0]
    atom_type = nz[:, 1]
    coords = atom_tensor.reshape(-1, 3)
    coords = coords[coords != fill_value].reshape(-1, 3)
    return coords, residue_index, atom_type


def get_c_alpha(x: AtomTensor, index: int = 1) -> CoordTensor:
    """Returns tensor of C-alpha atoms: ``(L x 3)``

    :param x: Tensor of atom positions of shape:
        ``(N_residues, N_atoms (default=37), 3)``
    :type x: graphein.protein.tensor.types.AtomTensor
    :param index: Index of C-alpha atom in dimension 1 of the AtomTensor.
    :type index: int

    .. seealso:: :func:`get_backbone`
    """
    return x if x.ndim == 2 else x[:, index, :]


def get_center(
        x: Union[AtomTensor, CoordTensor],
        ca_only: bool = True,
        fill_value: float = 1e-5,
) -> CoordTensor:
    """
    Returns the center of a protein.

    .. code-block:: python
        import torch

        x = torch.rand((10, 37, 3))
        get_center(x)


    .. seealso::

        :meth:`center_protein`


    :param x: Point Cloud to Center. Torch tensor of shape ``(Length , 3)`` or
        ``(Length, num atoms, 3)``.
    :param ca_only: If ``True``, only the C-alpha atoms will be used to compute
        the center. Only relevant with AtomTensor inputs. Default is ``False``.
    :type ca_only: bool
    :param fill_value: Value used to denote missing atoms. Default is )``1e-5)``.
    :type fill_value: float
    :return: Torch tensor of shape ``(N,D)`` -- Center of Point Cloud
    :rtype: Union[graphein.protein.tensor.types.AtomTensor, graphein.protein.tensor.types.CoordTensor]
    """
    if x.ndim != 3:
        return x.mean(dim=0)
    if ca_only:
        return get_c_alpha(x).mean(dim=0)

    x_flat, _, _ = get_full_atom_coords(x, fill_value=fill_value)
    return x_flat.mean(dim=0)


def kabsch(
        A: Union[AtomTensor, CoordTensor],
        B: Union[AtomTensor, CoordTensor],
        ca_only: bool = True,
        fill_value: float = 1e-5,
        return_transformed: bool = True,
        allow_reflections: bool = False,
) -> Union[
    CoordTensor,
    Tuple[BackboneFrameTensor, torch.Tensor],
    Tuple[RotationMatrix, torch.Tensor],
]:
    """
    Computes registration between two (2D or 3D) point clouds with known
    correspondences using Kabsch algorithm for optimal rigid body alignment.

    This implementation handles both AtomTensor (full protein structures) and
    CoordTensor (coordinate arrays) inputs. When using AtomTensor with ca_only=True,
    the algorithm extracts C-alpha coordinates for alignment computation but
    applies the resulting transformation to the entire protein structure.

    Registration occurs in the zero centered coordinate system, and then
    must be transported back.

    .. see:: https://en.wikipedia.org/wiki/Kabsch_algorithm

    .. note::

        Based on implementation by Guillaume Bouvier (@bougui505):
        https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8

        Enhanced to handle AtomTensor inputs with proper coordinate extraction
        and improved reflection prevention for protein structure alignment.

    :param A: Point Cloud to Align (source). Can be AtomTensor of shape
        ``(N_residues, 37, 3)`` or CoordTensor of shape ``(N_points, 3)``
    :type A: Union[AtomTensor, CoordTensor]
    :param B: Reference Point Cloud (target). Same format as A.
    :type B: Union[AtomTensor, CoordTensor]
    :param ca_only: Whether to use only C-alpha atoms for alignment when input
        is AtomTensor. If True, alignment is computed using C-alpha atoms but
        transformation is applied to the full structure. Defaults to ``True``.
    :type ca_only: bool
    :param fill_value: Value used to denote missing atoms in AtomTensor.
        Only relevant when ``ca_only=False``. Defaults to ``1e-5``.
    :type fill_value: float
    :param return_transformed: If True, returns the transformed coordinates.
        If False, returns the rotation matrix and translation vector.
        Defaults to ``True``.
    :type return_transformed: bool
    :param allow_reflections: Whether to allow reflections in the transformation.
        If False, ensures proper rotations (det(R) = +1) to preserve chirality.
        Important for protein structures. Defaults to ``False``.
    :type allow_reflections: bool
    :return: Either the aligned point cloud (if return_transformed=True) or
        a tuple of (rotation_matrix, translation_vector). For AtomTensor inputs
        with ca_only=True, returns the full transformed protein structure.
    :rtype: Union[CoordTensor, AtomTensor, Tuple[RotationMatrix, torch.Tensor]]

    .. note::
        When ca_only=True and input is AtomTensor:
        - Alignment is computed using only C-alpha atoms (index 1)
        - Resulting transformation is applied to the full protein structure
        - This preserves the protein's complete atomic structure while using
          C-alpha atoms for optimal alignment computation
    """

    # Extract the coordinates to use for alignment
    if ca_only and A.ndim == 3:  # AtomTensor case
        coords_A = get_c_alpha(A)  # Shape: [residues, 3]
        coords_B = get_c_alpha(B)  # Shape: [residues, 3]
    elif not ca_only and A.ndim == 3:  # Full atom case
        coords_A, _, _ = get_full_atom_coords(A, fill_value=fill_value)
        coords_B, _, _ = get_full_atom_coords(B, fill_value=fill_value)
    else:  # Already CoordTensor
        coords_A = A
        coords_B = B

    # Get center of mass
    centroid_A = coords_A.mean(dim=0)
    centroid_B = coords_B.mean(dim=0)

    # Center the coordinates
    AA = coords_A - centroid_A
    BB = coords_B - centroid_B

    # Covariance matrix
    H = AA.T @ BB
    U, _, Vt = torch.svd(H)

    if not allow_reflections:
        with torch.no_grad():
            det = torch.det(U) * torch.det(Vt.T)
            if det < 0.0:
                Vt[-1, :] *= -1

    R = Vt @ U.T
    t = centroid_B - R @ centroid_A

    if return_transformed:
        # Apply transformation to the original coordinates
        if ca_only and A.ndim == 3:  # Transform the full AtomTensor
            A_transformed = A.clone()
            A_transformed[:, 1, :] = (R @ coords_A.T).T + t
            return A_transformed
        else:  # Transform CoordTensor or full atoms
            return (R @ coords_A.T).T + t
    else:
        return (R, t)
