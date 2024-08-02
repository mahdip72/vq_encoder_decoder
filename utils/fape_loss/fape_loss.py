import torch
from typing import Optional
from utils.fape_loss.rigid_utils import Rotation, Rigid
from utils.custom_losses import rigid_from_3_points_batch


def compute_frame_aligned_point_error(
        pred_frames: Rigid,
        target_frames: Rigid,
        frames_mask: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        positions_mask: torch.Tensor,
        length_scale: float,
        pair_mask: Optional[torch.Tensor] = None,
        l1_clamp_distance: Optional[float] = None,
        eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            pair_mask:
                [*,  N_frames, N_pts] mask to use for
                separating intra- from inter-chain losses.
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
            transformed predicted coordinates
            transformed target coordinates
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    # Replace NaNs with zeros. This is extra steps compared to openfold code.
    local_pred_pos = torch.nan_to_num(local_pred_pos, nan=0.0)
    local_target_pos = torch.nan_to_num(local_target_pos, nan=0.0)

    # TODO: decide whether averaging across frames or selecting one frame is better
    # # Average transformed coordinates across all frames
    # transformed_pred_coords = torch.mean(local_pred_pos, dim=1).detach()
    # transformed_true_coords = torch.mean(local_target_pos, dim=1).detach()

    # Get the transformed coordinates corresponding to the first frame
    transformed_pred_coords = local_pred_pos[:, 0, :, :].detach()
    transformed_true_coords = local_target_pos[:, 0, :, :].detach()

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))

        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        norm_factor = torch.sum(mask, dim=(-2, -1))

        normed_error = normed_error / (eps + norm_factor)
    else:
        # FP16-friendly averaging. Roughly equivalent to:
        #
        # norm_factor = (
        #     torch.sum(frames_mask, dim=-1) *
        #     torch.sum(positions_mask, dim=-1)
        # )
        # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
        #
        # ("roughly" because eps is necessarily duplicated in the latter)
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = (
                normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        )
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error, transformed_pred_coords, transformed_true_coords


def compute_fape_loss(x_predicted, x_true, r_predicted, r_true, masks):
    batch_size, num_amino_acids, _, _ = x_true.shape

    t_true = x_true[:, :, 1, :]
    t_predicted = x_predicted[:, :, 1, :]

    # Compute the rigid transformation using the first three amino acids
    # r_true, t_true = rigid_from_3_points_batch(x_true[:, :, 0, :],
    #                                            x_true[:, :, 1, :],
    #                                            x_true[:, :, 2, :])
    # r_predicted, t_predicted = rigid_from_3_points_batch(x_predicted[:, :, 0, :],
    #                                                      x_predicted[:, :, 1, :],
    #                                                      x_predicted[:, :, 2, :])

    rot_pred_object = Rotation(rot_mats=r_predicted, quats=None)
    rot_true_object = Rotation(rot_mats=r_true, quats=None)
    pre_rigid = Rigid(rots=rot_pred_object, trans=t_predicted)
    true_rigid = Rigid(rots=rot_true_object, trans=t_true)

    loss_value, transformed_pred_coords, transformed_true_coords = compute_frame_aligned_point_error(
        pred_frames=pre_rigid, target_frames=true_rigid, frames_mask=masks,
        pred_positions=t_predicted, target_positions=t_true, positions_mask=masks,
        length_scale=10.0, l1_clamp_distance=10.0, eps=1e-4
    )
    return loss_value, transformed_pred_coords, transformed_true_coords