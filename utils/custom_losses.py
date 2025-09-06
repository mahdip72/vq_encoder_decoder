import torch
import torch.nn.functional as F
from utils.alignment import kabsch
import torch.distributed as dist
from typing import Optional


def compute_grad_norm(loss, parameters, norm_type=2):
    """
    Compute the gradient norm for a given loss and model parameters without altering existing gradients.

    Args:
        loss (torch.Tensor): The loss tensor.
        parameters (iterable): Iterable of model parameters.
        norm_type (float): The type of norm (default 2 for L2 norm).

    Returns:
        torch.Tensor: The gradient norm.
    """
    grads = torch.autograd.grad(
        loss,
        [p for p in parameters if p.requires_grad],
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    grads = [g for g in grads if g is not None]
    if not grads:
        return torch.tensor(0.0)
    norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return norm


def adjust_coeff_by_grad(coeff, grad_norm, decrease_factor=0.9, increase_factor=1.1,
                         upper_thresh=0.5, lower_thresh=0.05):
    """
    Adjust a coefficient based on gradient norm magnitude.

    Args:
        coeff (float): Current coefficient value
        grad_norm (float): Gradient norm of corresponding loss component
        decrease_factor (float): Multiplier when grad_norm > upper_thresh
        increase_factor (float): Multiplier when grad_norm < lower_thresh
        upper_thresh (float): Upper threshold for coefficient reduction
        lower_thresh (float): Lower threshold for coefficient increase

    Returns:
        float: Adjusted coefficient
    """
    if grad_norm > upper_thresh:
        return coeff * decrease_factor
    elif grad_norm < lower_thresh:
        return coeff * increase_factor
    else:
        return coeff


def aggregate_grad_norms(local_grad_norms, accelerator):
    """
    Aggregate gradient norms across all ranks for global signal.

    Args:
        local_grad_norms (dict): Local gradient norms per rank
        accelerator: HuggingFace Accelerator

    Returns:
        dict: Globally averaged gradient norms
    """
    global_grad_norms = {}
    for key, local_norm in local_grad_norms.items():
        gathered_norms = accelerator.gather_for_metrics(local_norm)
        global_grad_norms[key] = gathered_norms.mean().item()
    return global_grad_norms


def broadcast_coefficients(adaptive_loss_coeffs, accelerator):
    """
    Broadcast updated coefficients to all ranks.

    Args:
        adaptive_loss_coeffs (dict): Dictionary of adaptive coefficients
        accelerator: HuggingFace Accelerator

    Returns:
        dict: Updated coefficients (same across all ranks)
    """
    if accelerator.num_processes > 1:
        # Convert dict to list for broadcasting
        coeff_list = [adaptive_loss_coeffs]
        dist.broadcast_object_list(coeff_list, src=0)
        adaptive_loss_coeffs = coeff_list[0]
    return adaptive_loss_coeffs


def adjust_adaptive_coefficients(adaptive_loss_coeffs, global_grad_norms, configs):
    """
    Adjust adaptive loss coefficients based on global gradient norms.

    Args:
        adaptive_loss_coeffs (dict): Current adaptive coefficients.
        global_grad_norms (dict): Global gradient norms for each loss component.
        configs: Configuration object containing adaptive coefficient settings.

    Returns:
        dict: Updated adaptive coefficients.
    """
    # Get individual adaptive coefficient settings for each loss
    mse_adaptive = configs.train_settings.losses.mse.adaptive_coefficient
    backbone_distance_adaptive = configs.train_settings.losses.backbone_distance.adaptive_coefficient
    backbone_direction_adaptive = configs.train_settings.losses.backbone_direction.adaptive_coefficient
    binned_direction_adaptive = configs.train_settings.losses.binned_direction_classification.adaptive_coefficient
    binned_distance_adaptive = configs.train_settings.losses.binned_distance_classification.adaptive_coefficient
    ntp_adaptive = configs.train_settings.losses.next_token_prediction.adaptive_coefficient
    vq_adaptive = configs.train_settings.losses.vq.adaptive_coefficient

    # Adjust each coefficient based on its global grad norm only if adaptive is enabled for that loss
    if 'mse' in global_grad_norms and mse_adaptive:
        adaptive_loss_coeffs['mse'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['mse'], global_grad_norms['mse']
        )

    if 'backbone_distance' in global_grad_norms and backbone_distance_adaptive:
        adaptive_loss_coeffs['backbone_distance'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['backbone_distance'], global_grad_norms['backbone_distance']
        )

    if 'backbone_direction' in global_grad_norms and backbone_direction_adaptive:
        adaptive_loss_coeffs['backbone_direction'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['backbone_direction'], global_grad_norms['backbone_direction']
        )

    if 'binned_direction_classification' in global_grad_norms and binned_direction_adaptive:
        adaptive_loss_coeffs['binned_direction_classification'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['binned_direction_classification'],
            global_grad_norms['binned_direction_classification']
        )

    if 'binned_distance_classification' in global_grad_norms and binned_distance_adaptive:
        adaptive_loss_coeffs['binned_distance_classification'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['binned_distance_classification'], global_grad_norms['binned_distance_classification']
        )

    if 'commit' in global_grad_norms and vq_adaptive:
        adaptive_loss_coeffs['commit'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs['commit'], global_grad_norms['commit']
        )

    if 'ntp' in global_grad_norms and ntp_adaptive:
        adaptive_loss_coeffs['ntp'] = adjust_coeff_by_grad(
            adaptive_loss_coeffs.get('ntp', 1.0), global_grad_norms['ntp']
        )

    return adaptive_loss_coeffs


def log_per_loss_components(writer, loss_dict, global_step):
    """
    Log individual loss components to TensorBoard with hierarchical naming.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        loss_dict (dict): Dictionary containing individual loss components.
        global_step (int): Current global training step.
    """
    # Log each loss component with hierarchical naming
    for loss_name, loss_value in loss_dict.items():
        if torch.is_tensor(loss_value) and loss_value.numel() == 1:
            writer.add_scalar(f'step_loss/{loss_name}', loss_value.item(), global_step)


def log_gradient_norms_and_coeffs(writer, global_grad_norms, adaptive_loss_coeffs, global_step):
    """
    Log gradient norms and adaptive coefficients to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        global_grad_norms (dict): Global gradient norms for each loss component.
        adaptive_loss_coeffs (dict): Current adaptive coefficients.
        global_step (int): Current global training step.
    """
    # Log gradient norms
    for key, norm in global_grad_norms.items():
        writer.add_scalar(f'gradient norm/{key}', norm, global_step)

    # Log adaptive coefficients
    for coeff_name, coeff_val in adaptive_loss_coeffs.items():
        writer.add_scalar(f'adaptive_coeff/{coeff_name}', coeff_val, global_step)


def log_per_loss_grad_norms(loss_dict, net, configs, writer, accelerator, global_step, adaptive_loss_coeffs):
    """
    Log per-loss gradient norms, individual loss components, and adjust adaptive coefficients based on global gradient norms.
    Only activates when adaptive mode is enabled and warmup period is complete.
    Logs gradient norms, adaptive coefficients, and individual loss components to TensorBoard with hierarchical naming.

    Args:
        loss_dict (dict): Loss components from calculate_decoder_loss
        net (torch.nn.Module): The model
        configs: Configuration object
        writer (SummaryWriter): TensorBoard writer
        accelerator: HuggingFace Accelerator
        global_step (int): Current global training step
        adaptive_loss_coeffs (dict): Current adaptive coefficients

    Returns:
        dict: Updated adaptive coefficients
    """
    # Early return if not logging step or not sync gradients
    if not (accelerator.sync_gradients and
            global_step % configs.train_settings.gradient_norm_logging_freq == 0 and global_step > 0):
        return adaptive_loss_coeffs

    # Compute local gradient norms for each enabled loss with adaptive coefficients
    local_grad_norms = {}

    # Get individual adaptive coefficient settings for each loss
    mse_adaptive = configs.train_settings.losses.mse.adaptive_coefficient
    backbone_distance_adaptive = configs.train_settings.losses.backbone_distance.adaptive_coefficient
    backbone_direction_adaptive = configs.train_settings.losses.backbone_direction.adaptive_coefficient
    binned_direction_adaptive = configs.train_settings.losses.binned_direction_classification.adaptive_coefficient
    binned_distance_adaptive = configs.train_settings.losses.binned_distance_classification.adaptive_coefficient
    ntp_adaptive = configs.train_settings.losses.next_token_prediction.adaptive_coefficient

    if configs.train_settings.losses.mse.enabled and mse_adaptive:
        local_grad_norms['mse'] = compute_grad_norm(loss_dict['mse_loss'], net.parameters())

    if configs.train_settings.losses.backbone_distance.enabled and backbone_distance_adaptive:
        local_grad_norms['backbone_distance'] = compute_grad_norm(
            loss_dict['backbone_distance_loss'], net.parameters()
        )

    if configs.train_settings.losses.backbone_direction.enabled and backbone_direction_adaptive:
        local_grad_norms['backbone_direction'] = compute_grad_norm(
            loss_dict['backbone_direction_loss'], net.parameters()
        )

    if configs.train_settings.losses.binned_direction_classification.enabled and binned_direction_adaptive:
        local_grad_norms['binned_direction_classification'] = compute_grad_norm(
            loss_dict['binned_direction_classification_loss'], net.parameters()
        )

    if configs.train_settings.losses.binned_distance_classification.enabled and binned_distance_adaptive:
        local_grad_norms['binned_distance_classification'] = compute_grad_norm(
            loss_dict['binned_distance_classification_loss'], net.parameters()
        )

    if getattr(configs.train_settings.losses, 'next_token_prediction', None) and \
            configs.train_settings.losses.next_token_prediction.enabled and \
            ntp_adaptive and ('ntp_loss' in loss_dict):
        local_grad_norms['ntp'] = compute_grad_norm(loss_dict['ntp_loss'], net.parameters())

    if configs.model.vqvae.vector_quantization.enabled:
        local_grad_norms['commit'] = compute_grad_norm(loss_dict.get('commit', torch.tensor(0.0)), net.parameters())

    zero = torch.tensor(0.0, device=loss_dict['rec_loss'].device)
    total_loss_unscaled = loss_dict['rec_loss'] + loss_dict.get('commit', zero)
    local_grad_norms['total_unscaled'] = compute_grad_norm(total_loss_unscaled, net.parameters())

    # Aggregate across ranks for global signal
    global_grad_norms = aggregate_grad_norms(local_grad_norms, accelerator)

    # Adjust coefficients and broadcast (only after warmup and if any adaptive coefficients are enabled)
    if (accelerator.is_main_process and
            global_step > configs.optimizer.decay.warmup and
            len(local_grad_norms) > 0):  # Only proceed if we have any losses with adaptive coefficients

        adaptive_loss_coeffs = adjust_adaptive_coefficients(
            adaptive_loss_coeffs, global_grad_norms, configs
        )

    if accelerator.is_main_process:
        # Log gradient norms, coefficients, and individual loss components
        if configs.tensorboard_log:
            log_gradient_norms_and_coeffs(writer, global_grad_norms, adaptive_loss_coeffs, global_step)
            log_per_loss_components(writer, loss_dict, global_step)

    if (global_step > configs.optimizer.decay.warmup and
            len(local_grad_norms) > 0):  # Only broadcast if we have any losses with adaptive coefficients
        adaptive_loss_coeffs = broadcast_coefficients(adaptive_loss_coeffs, accelerator)

    return adaptive_loss_coeffs


def calculate_aligned_mse_loss(x_predicted, x_true, masks, alignment_strategy):
    """
    Calculates the MSE loss between x_predicted and x_true after performing alignment,
    applying the provided masks.

    Parameters:
    x_predicted (torch.Tensor): Predicted coordinates of shape [batch_size, seq_len, num_atoms, 3].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.
    alignment_strategy (str): Strategy for alignment. Options: 'kabsch', 'no'.
                              Use 'no' for no alignment (absolute position error).

    Returns:
    torch.Tensor: The computed MSE loss for each batch element.
    torch.Tensor: The predicted coordinates.
    torch.Tensor: The aligned true coordinates.
    """
    batch_size = x_predicted.shape[0]
    loss_list = []
    x_true_aligned_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_pred = x_predicted[i]  # [seq_len, num_atoms, 3]
        x_tru = x_true[i]  # [seq_len, num_atoms, 3]

        with torch.no_grad():
            if alignment_strategy == 'kabsch':
                # Extract valid residues (each with multiple atoms) based on mask
                x_tru_valid = x_tru[mask]  # Shape: (num_valid_residues, num_atoms, 3)
                x_pred_valid = x_pred[mask]
                # Clone full true coords to fill in aligned residues
                x_true_aligned = x_tru.clone()
                # Flatten atoms across residues for Kabsch input
                coords_true_flat = x_tru_valid.reshape(-1, 3)
                coords_pred_flat = x_pred_valid.reshape(-1, 3)
                # Perform Kabsch alignment on flattened points (using all coordinates)
                aligned_flat = kabsch(
                    coords_true_flat, coords_pred_flat,
                    return_transformed=True,
                    allow_reflections=False
                ).detach()
                # Reshape back to per-residue atom layout
                aligned_valid = aligned_flat.reshape_as(x_tru_valid)
                # Assign aligned residues back into the full tensor
                x_true_aligned[mask] = aligned_valid

            elif alignment_strategy == 'no':
                # No alignment, use original true coordinates directly
                x_true_aligned = x_tru.detach()

            x_true_aligned_list.append(x_true_aligned)

        # Compute MSE loss using the masked areas
        loss = torch.mean(((x_pred[mask] - x_true_aligned[mask]) ** 2))
        loss_list.append(loss)

    return torch.stack(loss_list), x_predicted, torch.stack(x_true_aligned_list)


def calculate_backbone_distance_loss(x_predicted, x_true, masks):
    """
    Calculates the backbone distance loss between x_predicted and x_true after applying masks.

    Parameters:
    x_predicted (torch.Tensor): Predicted coordinates of shape [batch_size, seq_len, num_atoms, 3].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed backbone distance loss for each batch element.
    """
    batch_size = x_predicted.shape[0]
    loss_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_pred = x_predicted[i][mask]  # [num_valid, num_atoms, 3]
        x_tru = x_true[i][mask]

        # Extract the backbone atoms (N, CA, C)
        x_pred_backbone = x_pred[:, :3, :].reshape(-1, 3)
        x_tru_backbone = x_tru[:, :3, :].reshape(-1, 3)

        # Compute pairwise L2 distance matrices
        D_pred = torch.cdist(x_pred_backbone, x_pred_backbone, p=2)
        D_true = torch.cdist(x_tru_backbone, x_tru_backbone, p=2)

        # Compute the squared differences
        distance_diff = (D_pred - D_true) ** 2

        # Clamp the maximum error to (5 Å)^2
        clamped_diff = torch.clamp(distance_diff, max=5 ** 2)

        # Take the mean of the clamped differences
        loss = clamped_diff.mean()
        loss_list.append(loss)

    return torch.stack(loss_list)


def compute_vectors(coords):
    """
    Computes the six vectors described in the pseudocode.

    Args:
        coords (torch.Tensor): The input coordinates of shape (num_points, num_atoms, 3).

    Returns:
        torch.Tensor: The computed vectors of shape (6 * num_points, 3).
    """
    # Compute the vectors
    N_to_CA = coords[:, 1, :] - coords[:, 0, :]
    CA_to_C = coords[:, 2, :] - coords[:, 1, :]
    C_to_Nnext = F.pad(
        coords[1:, 0, :] - coords[:-1, 2, :],
        (0, 0, 0, 1),
        value=0.0
    )

    Cprev_to_N = F.pad(
        coords[1:, 0, :] - coords[:-1, 2, :],
        (0, 0, 1, 0),
        value=0.0
    )

    # Compute the normal vectors
    nCA = -torch.cross(N_to_CA, CA_to_C, dim=-1)
    nN = torch.cross(Cprev_to_N, N_to_CA, dim=-1)
    nC = torch.cross(CA_to_C, C_to_Nnext, dim=-1)

    # Concatenate all vectors
    vectors = torch.cat([N_to_CA, CA_to_C, C_to_Nnext, nCA, nN, nC], dim=0)

    return vectors


def calculate_backbone_direction_loss(x_predicted, x_true, masks):
    """
    Calculates the backbone direction loss between x_predicted and x_true after applying masks.

    Parameters:
    x_predicted (torch.Tensor): Predicted coordinates of shape [batch_size, seq_len, num_atoms, 3].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed backbone direction loss for each batch element.
    """
    batch_size = x_predicted.shape[0]
    loss_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_pred = x_predicted[i][mask]  # [num_valid, num_atoms, 3]
        x_tru = x_true[i][mask]

        # Compute vectors for predicted and true coordinates
        V_pred = compute_vectors(x_pred)
        V_true = compute_vectors(x_tru)

        # Compute pairwise dot products
        D_pred = torch.matmul(V_pred, V_pred.transpose(0, 1))
        D_true = torch.matmul(V_true, V_true.transpose(0, 1))

        # Compute squared differences
        E = (D_pred - D_true) ** 2

        # Clamp the maximum error to 20
        E = torch.clamp(E, max=20)

        # Take the mean of the clamped differences
        loss = E.mean()
        loss_list.append(loss)

    return torch.stack(loss_list)


def calculate_binned_direction_classification_loss(dir_loss_logits, x_true, masks):
    """
    Calculates the binned direction classification loss.

    Parameters:
    dir_loss_logits (torch.Tensor): Logits of shape [batch_size, seq_len, seq_len, 6, 16].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed binned direction classification loss for each batch element.
    """
    batch_size, _, _, _ = x_true.shape
    loss_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_tru = x_true[i][mask]  # [num_valid, num_atoms, 3]

        # Compute vectors for true coordinates
        CA_to_C = x_tru[:, 2, :] - x_tru[:, 1, :]
        CA_to_N = x_tru[:, 0, :] - x_tru[:, 1, :]
        nCA = torch.cross(CA_to_C, CA_to_N, dim=-1)

        # Normalize vectors to unit length
        CA_to_C = F.normalize(CA_to_C, dim=-1)
        CA_to_N = F.normalize(CA_to_N, dim=-1)
        nCA = F.normalize(nCA, dim=-1)

        # Compute pairwise dot products
        dot_products = torch.stack([
            torch.matmul(CA_to_N, CA_to_N.transpose(0, 1)),
            torch.matmul(CA_to_N, CA_to_C.transpose(0, 1)),
            torch.matmul(CA_to_N, nCA.transpose(0, 1)),
            torch.matmul(CA_to_C, CA_to_C.transpose(0, 1)),
            torch.matmul(CA_to_C, nCA.transpose(0, 1)),
            torch.matmul(nCA, nCA.transpose(0, 1))
        ], dim=-1)  # Shape: [num_valid, num_valid, 6]

        # Bin the dot products into 16 bins
        bins = torch.linspace(-1, 1, 16, device=x_true.device)
        dot_products = torch.clamp(dot_products, min=-0.9999, max=0.9999)
        labels = torch.bucketize(dot_products, bins)  # Shape: [num_valid, num_valid, 6]

        # Compute cross-entropy loss
        logits = dir_loss_logits[i][mask][:, mask]  # Shape: [num_valid, num_valid, 6, 16]
        loss = F.cross_entropy(logits.view(-1, 16), labels.view(-1), reduction='mean')
        loss_list.append(loss)

    return torch.stack(loss_list)


def calculate_binned_distance_classification_loss(dist_loss_logits, x_true, masks):
    """
    Calculates the binned distance classification loss.

    Parameters:
    dist_loss_logits (torch.Tensor): Logits of shape [batch_size, seq_len, seq_len, 64].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed binned distance classification loss for each batch element.
    """
    batch_size, _, _, _ = x_true.shape
    loss_list = []

    # Define the bin edges
    bin_edges = torch.tensor([(2.3125 + 0.3075 * i) ** 2 for i in range(63)], device=x_true.device)

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_tru = x_true[i][mask]  # [num_valid, num_atoms, 3]

        # Compute Cβ coordinates
        N_to_CA = x_tru[:, 1, :] - x_tru[:, 0, :]
        CA_to_C = x_tru[:, 2, :] - x_tru[:, 1, :]
        n = torch.cross(N_to_CA, CA_to_C, dim=-1)

        a = -0.58273431
        b = 0.56802827
        c = -0.54067466

        C_beta = a * n + b * N_to_CA + c * CA_to_C + x_tru[:, 1, :]

        # Compute pairwise distances
        pairwise_distances = torch.cdist(C_beta, C_beta, p=2) ** 2

        # Bin the distances into 64 bins
        labels = torch.bucketize(pairwise_distances, bin_edges)  # Shape: [num_valid, num_valid]

        # Compute cross-entropy loss
        logits = dist_loss_logits[i][mask][:, mask]  # Shape: [num_valid, num_valid, 64]
        loss = F.cross_entropy(logits.view(-1, 64), labels.view(-1), reduction='mean')
        loss_list.append(loss)

    return torch.stack(loss_list)


def calculate_ntp_loss(ntp_logits: Optional[torch.Tensor], indices: Optional[torch.Tensor],
                       masks: torch.Tensor) -> torch.Tensor:
    """
    Vectorized next-token prediction loss using VQ code indices as labels.

    Steps:
    - Apply valid mask to indices; set invalid positions to ignore_index (-100)
    - Shift labels by one (labels[t] = indices[t+1]) and pad last with ignore_index
    - Compute cross-entropy with ignore_index=-100
    - Return per-sample mean loss over non-ignored positions (zeros if none)
    """
    if ntp_logits is None or indices is None:
        B = masks.size(0)
        return torch.zeros(B, device=masks.device)

    ignore_index = -100
    device = ntp_logits.device

    # Ensure dtypes
    indices = indices.to(dtype=torch.long, device=device)
    masks = masks.to(dtype=torch.bool, device=device)

    B, L, K = ntp_logits.shape

    # Mask invalid positions to ignore_index
    labels_masked = indices.masked_fill(~masks, ignore_index)

    # Shift left by one and pad last as ignore
    pad_col = torch.full((B, 1), ignore_index, dtype=torch.long, device=device)
    labels = torch.cat([labels_masked[:, 1:], pad_col], dim=1)  # (B, L)

    # Flatten for CE
    logits_flat = ntp_logits.reshape(B * L, K)
    labels_flat = labels.reshape(B * L)

    # Per-position loss (ignored positions contribute 0 with 'none' + manual mask)
    loss_flat = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='none')
    loss = loss_flat.view(B, L)

    # Per-sample mean over non-ignored positions
    valid_pos = (labels != ignore_index)
    denom = valid_pos.sum(dim=1).clamp(min=1)
    per_sample_loss = (loss.sum(dim=1) / denom)

    return per_sample_loss


def calculate_decoder_loss(x_predicted, x_true, masks, configs, seq=None, dir_loss_logits=None, dist_loss_logits=None,
                           alignment_strategy='kabsch', adaptive_loss_coeffs=None, ntp_logits=None, indices=None,
                           valid_mask=None):
    # Compute aligned MSE foundation
    mse_raw, x_pred_aligned, x_true_aligned = calculate_aligned_mse_loss(
        x_predicted, x_true, masks, alignment_strategy=alignment_strategy)
    device = x_predicted.device

    # Initialize adaptive coefficients (defaults to 1.0 if not provided)
    adaptive = adaptive_loss_coeffs or {
        'mse': 1.0,
        'backbone_distance': 1.0,
        'backbone_direction': 1.0,
        'binned_direction_classification': 1.0,
        'binned_distance_classification': 1.0,
        'ntp': 1.0
    }

    # Prepare loss dict with weighted components or 0.0 if disabled
    loss_dict = {}
    # MSE reconstruction
    if configs.train_settings.losses.mse.enabled:
        w = configs.train_settings.losses.mse.weight
        mse_coeff = adaptive.get('mse', 1.0)
        loss_dict['mse_loss'] = mse_raw.mean() * w * mse_coeff
    else:
        loss_dict['mse_loss'] = torch.tensor(0.0, device=device)
    # Backbone distance
    if configs.train_settings.losses.backbone_distance.enabled:
        w = configs.train_settings.losses.backbone_distance.weight
        backbone_distance_coeff = adaptive.get('backbone_distance', 1.0)
        loss_dict['backbone_distance_loss'] = calculate_backbone_distance_loss(
            x_pred_aligned, x_true_aligned, masks).mean() * w * backbone_distance_coeff
    else:
        loss_dict['backbone_distance_loss'] = torch.tensor(0.0, device=device)
    # Backbone direction
    if configs.train_settings.losses.backbone_direction.enabled:
        w = configs.train_settings.losses.backbone_direction.weight
        backbone_direction_coeff = adaptive.get('backbone_direction', 1.0)
        loss_dict['backbone_direction_loss'] = calculate_backbone_direction_loss(
            x_pred_aligned, x_true_aligned, masks).mean() * w * backbone_direction_coeff
    else:
        loss_dict['backbone_direction_loss'] = torch.tensor(0.0, device=device)
    # Binned direction classification
    if configs.train_settings.losses.binned_direction_classification.enabled:
        w = configs.train_settings.losses.binned_direction_classification.weight
        binned_direction_coeff = adaptive.get('binned_direction_classification', 1.0)
        val = calculate_binned_direction_classification_loss(
            dir_loss_logits, x_true_aligned, masks).mean() if dir_loss_logits is not None else torch.tensor(0.0,
                                                                                                            device=device)
        loss_dict['binned_direction_classification_loss'] = val * w * binned_direction_coeff
    else:
        loss_dict['binned_direction_classification_loss'] = torch.tensor(0.0, device=device)
    # Binned distance classification
    if configs.train_settings.losses.binned_distance_classification.enabled:
        w = configs.train_settings.losses.binned_distance_classification.weight
        binned_distance_coeff = adaptive.get('binned_distance_classification', 1.0)
        val = calculate_binned_distance_classification_loss(
            dist_loss_logits, x_true_aligned, masks).mean() if dist_loss_logits is not None else torch.tensor(0.0,
                                                                                                              device=device)
        loss_dict['binned_distance_classification_loss'] = val * w * binned_distance_coeff
    else:
        loss_dict['binned_distance_classification_loss'] = torch.tensor(0.0, device=device)

    if configs.train_settings.losses.next_token_prediction.enabled:
        w = configs.train_settings.losses.next_token_prediction.weight
        ntp_coeff = adaptive.get('ntp', 1.0)
        ntp_per_sample = calculate_ntp_loss(ntp_logits, indices, valid_mask)
        loss_dict['ntp_loss'] = ntp_per_sample.mean() * w * ntp_coeff
    else:
        loss_dict['ntp_loss'] = torch.tensor(0.0, device=device)

    # Sum reconstruction components
    valid_losses = [v for k, v in loss_dict.items() if 'loss' in k and not torch.isnan(v)]
    if not valid_losses:
        loss_dict['rec_loss'] = torch.tensor(0.0, device=device)
    else:
        loss_dict['rec_loss'] = sum(valid_losses)
    return loss_dict, x_pred_aligned, x_true_aligned
