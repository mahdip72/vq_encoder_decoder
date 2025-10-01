import torch
import torchmetrics
from typing import Dict, Any, Optional, cast
from utils.metrics import GDTTS, TMScore, update_perplexity_from_ntp


def init_metrics(configs, accelerator) -> Dict[str, Any]:
    """Initialize metric objects used in training/validation.

    Inputs:
    - configs: Hydrated config object with train_settings.losses and NTP settings.
    - accelerator: accelerate.Accelerator with a target device for metric state.

    Behavior:
    - Always creates MAE, RMSD, GDTTS, and TMScore metrics on the correct device.
    - Optionally creates a Perplexity metric if next_token_prediction is enabled.

    Returns:
    - Dict[str, Any]:
        {
          'mae': torchmetrics.MeanAbsoluteError,
          'rmsd': torchmetrics.MeanSquaredError(squared=False),
          'gdtts': utils.metrics.GDTTS,
          'tm_score': utils.metrics.TMScore,
          'perplexity': torchmetrics.text.Perplexity | None
        }
    """
    metrics: Dict[str, Any] = {
        'mae': torchmetrics.MeanAbsoluteError().to(accelerator.device),
        'rmsd': torchmetrics.MeanSquaredError(squared=False).to(accelerator.device),
        'gdtts': GDTTS().to(accelerator.device),
        'tm_score': TMScore().to(accelerator.device),
        'perplexity': None,
    }

    if getattr(configs.train_settings, 'losses', None) and \
            getattr(configs.train_settings.losses, 'next_token_prediction', None) and \
            configs.train_settings.losses.next_token_prediction.enabled:
        from torchmetrics.text import Perplexity
        # cast to Any to satisfy type checkers that try to unify dict value types
        metrics['perplexity'] = cast(Any, Perplexity(ignore_index=-100).to(accelerator.device))

    return metrics


def reset_metrics(metrics: Dict[str, Any]) -> None:
    """Reset all metric states (to start a new epoch or phase).

    Inputs:
    - metrics: Dict returned by init_metrics.
    """
    metrics['mae'].reset()
    metrics['rmsd'].reset()
    metrics['gdtts'].reset()
    metrics['tm_score'].reset()
    if metrics.get('perplexity') is not None:
        metrics['perplexity'].reset()


def update_metrics(metrics: Dict[str, Any],
                   trans_pred_coords: torch.Tensor,
                   trans_true_coords: torch.Tensor,
                   masks: torch.Tensor,
                   output_dict: Dict[str, torch.Tensor],
                   ignore_index: int = -100) -> None:
    """Update metric states with a new batch.

    Inputs:
    - metrics: Dict returned by init_metrics.
    - trans_pred_coords: Tensor of shape (B, L, 3_atoms, 3_xyz) predicted coords (e.g., N-CA-C-O).
    - trans_true_coords: Tensor of shape (B, L, 3_atoms, 3_xyz) target coords.
    - masks: Bool/float tensor of shape (B, L) denoting valid residues.
    - output_dict: Model outputs that may include 'ntp_logits', 'indices', 'valid_mask' for perplexity.
    - ignore_index: Label index to ignore when updating perplexity.

    Behavior:
    - Updates MAE, RMSD, GDTTS using masked per-atom points.
    - Updates TM-Score using C-alpha (atom index 1) and the provided masks.
    - If available, updates Perplexity using next-token prediction logits and indices.
    """
    # Coordinate-based metrics (MAE/RMSD/GDTTS)
    if masks.any():
        masked_outputs = trans_pred_coords[masks].reshape(-1, 3)
        masked_labels = trans_true_coords[masks].reshape(-1, 3)
        if masked_outputs.numel() > 0:
            metrics['mae'].update(masked_outputs.detach(), masked_labels.detach())
            metrics['rmsd'].update(masked_outputs.detach(), masked_labels.detach())
            # GDTTS expects backbone points (we feed masked points)
            metrics['gdtts'].update(masked_outputs.detach(), masked_labels.detach())

    # TM-Score: use C-alpha coordinates and original masks; metric handles masking
    pred_ca_coords = trans_pred_coords[:, :, 1, :].detach()
    true_ca_coords = trans_true_coords[:, :, 1, :].detach()
    metrics['tm_score'].update(pred_ca_coords, true_ca_coords, masks.detach().bool())

    # Optional perplexity from NTP
    if metrics.get('perplexity') is not None:
        update_perplexity_from_ntp(
            metrics['perplexity'],
            output_dict.get('ntp_logits', None),
            output_dict.get('indices', None),
            output_dict.get('ntp_mask', None),
            ignore_index=ignore_index,
        )


def compute_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Compute scalar values from metric states.

    Inputs:
    - metrics: Dict returned by init_metrics.

    Returns:
    - Dict[str, float]: {'mae', 'rmsd', 'gdtts', 'tm_score', 'perplexity'} where
      perplexity is NaN if the metric wasn't enabled.
    """
    out = {
        'mae': metrics['mae'].compute().cpu().item(),
        'rmsd': metrics['rmsd'].compute().cpu().item(),
        'gdtts': metrics['gdtts'].compute().cpu().item(),
        'tm_score': metrics['tm_score'].compute().cpu().item(),
        'perplexity': float('nan')
    }
    if metrics.get('perplexity') is not None:
        out['perplexity'] = metrics['perplexity'].compute().cpu().item()
    return out


def init_accumulator(accum_iter: int) -> Dict[str, Any]:
    """Create a running accumulator dict for losses and codebook activation.

    Inputs:
    - accum_iter: Number of micro-batches per optimizer step (gradient accumulation).

    Returns:
    - Dict[str, Any] with keys tracking per-accumulation values, totals, a step counter,
      and a running set of unique codebook indices.
    """
    return {
        # per-accumulation running values (averaged per micro-step)
        'train_step_loss': 0.0,
        'train_rec_loss': 0.0,
        'train_vq_loss': 0.0,
        'train_ntp_loss': 0.0,
        'train_tik_tok_padding_loss': 0.0,
        # unscaled per-accumulation running values
        'train_unscaled_step_loss': 0.0,
        'train_unscaled_rec_loss': 0.0,
        'train_unscaled_vq_loss': 0.0,
        'train_unscaled_ntp_loss': 0.0,
        'train_unscaled_tik_tok_padding_loss': 0.0,
        # totals across finalized steps
        'total_step_loss': 0.0,
        'total_rec_loss': 0.0,
        'total_vq_loss': 0.0,
        'total_ntp_loss': 0.0,
        'total_tik_tok_padding_loss': 0.0,
        # unscaled totals across finalized steps
        'total_unscaled_step_loss': 0.0,
        'total_unscaled_rec_loss': 0.0,
        'total_unscaled_vq_loss': 0.0,
        'total_unscaled_ntp_loss': 0.0,
        'total_unscaled_tik_tok_padding_loss': 0.0,
        'counter': 0,
        'accum_iter': accum_iter,
        'unique_indices': set(),
    }


def _gather_mean(accelerator, value: torch.Tensor, repeat: Optional[int] = None) -> torch.Tensor:
    """All-gather a tensor across processes and return its mean.

    Inputs:
    - accelerator: accelerate.Accelerator for distributed gather.
    - value: Scalar or 1D batch-sized tensor.
    - repeat: If set, repeat gathered values this many times before averaging (emulates
      legacy averaging behavior that scaled by batch size).

    Returns:
    - torch.Tensor: Scalar tensor of the mean value across processes.
    """
    # value is scalar tensor or batch-sized tensor; we gather then optionally repeat to emulate original averaging pattern
    if value.dim() == 0:
        value = value.unsqueeze(0)
    gathered = accelerator.gather(value.detach())
    if repeat is not None and repeat > 1:
        gathered = gathered.repeat(repeat)
    return gathered.mean()


def accumulate_losses(acc: Dict[str, Any],
                      loss_dict: Dict[str, torch.Tensor],
                      output_dict: Dict[str, torch.Tensor],
                      configs,
                      accelerator,
                      use_output_vq: bool = False) -> None:
    """Accumulate loss values for the current micro-batch.

    Inputs:
    - acc: Accumulator dict from init_accumulator (mutated in-place).
    - loss_dict: Dict with 'step_loss', 'rec_loss', 'vq_loss', and optionally 'ntp_loss' tensors.
    - output_dict: Model outputs, used to source 'vq_loss' when use_output_vq=True (validation pattern).
    - configs: Global config to access train_settings.batch_size.
    - accelerator: accelerate.Accelerator for distributed mean.
    - use_output_vq: If True, use output_dict['vq_loss'] instead of loss_dict['vq_loss'].

    Behavior:
    - Scales each loss contribution by 1/acc['accum_iter'] for gradient accumulation.
    - Averages across processes to get a stable scalar before accumulating.
    """
    bs = configs.train_settings.batch_size
    acc['train_step_loss'] += _gather_mean(accelerator, loss_dict['step_loss'], repeat=bs).item() / acc['accum_iter']
    acc['train_rec_loss'] += _gather_mean(accelerator, loss_dict['rec_loss'], repeat=bs).item() / acc['accum_iter']
    # vq loss source differs between train and valid in original code
    vq_src = output_dict['vq_loss'] if use_output_vq else loss_dict['vq_loss']
    acc['train_vq_loss'] += _gather_mean(accelerator, vq_src, repeat=bs).item() / acc['accum_iter']
    # ntp loss may be missing; default to zero
    if 'ntp_loss' in loss_dict and loss_dict['ntp_loss'] is not None:
        acc['train_ntp_loss'] += _gather_mean(accelerator, loss_dict['ntp_loss'], repeat=bs).item() / acc['accum_iter']
    if 'tik_tok_padding_loss' in loss_dict and loss_dict['tik_tok_padding_loss'] is not None:
        acc['train_tik_tok_padding_loss'] += _gather_mean(
            accelerator,
            loss_dict['tik_tok_padding_loss'],
            repeat=bs,
        ).item() / acc['accum_iter']

    # Unscaled contributions
    if 'unscaled_step_loss' in loss_dict:
        acc['train_unscaled_step_loss'] += _gather_mean(accelerator, loss_dict['unscaled_step_loss'], repeat=bs).item() / acc['accum_iter']
    if 'unscaled_rec_loss' in loss_dict:
        acc['train_unscaled_rec_loss'] += _gather_mean(accelerator, loss_dict['unscaled_rec_loss'], repeat=bs).item() / acc['accum_iter']
    if 'unscaled_vq_loss' in loss_dict:
        # vq unscaled source mirrors scaled behavior when validating (taken from output)
        unscaled_vq_src = output_dict['vq_loss'] if use_output_vq else loss_dict['unscaled_vq_loss']
        acc['train_unscaled_vq_loss'] += _gather_mean(accelerator, unscaled_vq_src, repeat=bs).item() / acc['accum_iter']
    if 'unscaled_ntp_loss' in loss_dict and loss_dict['unscaled_ntp_loss'] is not None:
        acc['train_unscaled_ntp_loss'] += _gather_mean(accelerator, loss_dict['unscaled_ntp_loss'], repeat=bs).item() / acc['accum_iter']
    if 'unscaled_tik_tok_padding_loss' in loss_dict and loss_dict['unscaled_tik_tok_padding_loss'] is not None:
        acc['train_unscaled_tik_tok_padding_loss'] += _gather_mean(
            accelerator,
            loss_dict['unscaled_tik_tok_padding_loss'],
            repeat=bs,
        ).item() / acc['accum_iter']


def finalize_step(acc: Dict[str, Any]) -> None:
    """Move per-accumulation running sums into epoch totals and reset micro-trackers.

    Inputs:
    - acc: Accumulator dict from init_accumulator (mutated in-place).
    """
    acc['total_step_loss'] += acc['train_step_loss']
    acc['total_rec_loss'] += acc['train_rec_loss']
    acc['total_vq_loss'] += acc['train_vq_loss']
    acc['total_ntp_loss'] += acc['train_ntp_loss']
    acc['total_tik_tok_padding_loss'] += acc['train_tik_tok_padding_loss']

    acc['total_unscaled_step_loss'] += acc['train_unscaled_step_loss']
    acc['total_unscaled_rec_loss'] += acc['train_unscaled_rec_loss']
    acc['total_unscaled_vq_loss'] += acc['train_unscaled_vq_loss']
    acc['total_unscaled_ntp_loss'] += acc['train_unscaled_ntp_loss']
    acc['total_unscaled_tik_tok_padding_loss'] += acc['train_unscaled_tik_tok_padding_loss']

    acc['train_step_loss'] = 0.0
    acc['train_rec_loss'] = 0.0
    acc['train_vq_loss'] = 0.0
    acc['train_ntp_loss'] = 0.0
    acc['train_tik_tok_padding_loss'] = 0.0

    acc['train_unscaled_step_loss'] = 0.0
    acc['train_unscaled_rec_loss'] = 0.0
    acc['train_unscaled_vq_loss'] = 0.0
    acc['train_unscaled_ntp_loss'] = 0.0
    acc['train_unscaled_tik_tok_padding_loss'] = 0.0

    acc['counter'] += 1


def average_losses(acc: Dict[str, Any]) -> Dict[str, float]:
    """Compute per-step averages from totals accumulated so far.

    Inputs:
    - acc: Accumulator dict from init_accumulator.

    Returns:
    - Dict[str, float]: {'avg_step_loss', 'avg_rec_loss', 'avg_vq_loss', 'avg_ntp_loss'}
      averaged over acc['counter'] finalized optimizer steps.
    """
    denom = max(1, acc['counter'])
    return {
        'avg_step_loss': acc['total_step_loss'] / denom,
        'avg_rec_loss': acc['total_rec_loss'] / denom,
        'avg_vq_loss': acc['total_vq_loss'] / denom,
        'avg_ntp_loss': acc['total_ntp_loss'] / denom,
        'avg_tik_tok_padding_loss': acc['total_tik_tok_padding_loss'] / denom,
        'avg_unscaled_step_loss': acc['total_unscaled_step_loss'] / denom,
        'avg_unscaled_rec_loss': acc['total_unscaled_rec_loss'] / denom,
        'avg_unscaled_vq_loss': acc['total_unscaled_vq_loss'] / denom,
        'avg_unscaled_ntp_loss': acc['total_unscaled_ntp_loss'] / denom,
        'avg_unscaled_tik_tok_padding_loss': acc['total_unscaled_tik_tok_padding_loss'] / denom,
    }


def update_unique_indices(acc: Dict[str, Any], indices: torch.Tensor, accelerator) -> None:
    """Update the set of unique codebook indices seen so far.

    Inputs:
    - acc: Accumulator dict from init_accumulator (mutated in-place).
    - indices: Tensor of shape (B, L) containing codebook indices for tokens.
    - accelerator: accelerate.Accelerator for distributed gather.
    """
    gathered_indices = accelerator.gather(indices)
    acc['unique_indices'].update(gathered_indices.unique().cpu().tolist())


def compute_activation(acc: Dict[str, Any], codebook_size: int) -> float:
    """Compute codebook activation ratio based on unique indices.

    Inputs:
    - acc: Accumulator dict from init_accumulator.
    - codebook_size: Total number of entries in the codebook.

    Returns:
    - float in [0, 1]: fraction of codebook entries used so far.
    """
    if codebook_size <= 0:
        return 0.0
    return len(acc['unique_indices']) / float(codebook_size)


def progress_postfix(optimizer, loss_dict: Dict[str, torch.Tensor], global_step: int) -> Dict[str, Any]:
    """Build a compact dict for tqdm postfix with current train stats.

    Inputs:
    - optimizer: torch.optim.Optimizer for current learning rate.
    - loss_dict: Dict with current 'step_loss', 'rec_loss', and optionally 'vq_loss'.
    - global_step: Global optimizer step counter (int).

    Returns:
    - Dict[str, Any] with keys {'lr', 'step_loss', 'rec_loss', 'vq_loss', 'global_step'}
      for logging in progress bars.
    """
    return {
        'lr': optimizer.param_groups[0]['lr'],
        'step_loss': float(loss_dict['step_loss'].detach().item()),
        'rec_loss': float(loss_dict['rec_loss'].detach().item()),
        'vq_loss': float(loss_dict['vq_loss'].detach().item()) if 'vq_loss' in loss_dict else float('nan'),
        'global_step': int(global_step),
    }


def log_tensorboard_epoch(writer,
                          avgs: Dict[str, float],
                          metrics_values: Dict[str, float],
                          epoch: int,
                          activation_percent: float,
                          include_ntp: bool = False) -> None:
    """Log epoch-level losses and metrics to TensorBoard.

    Inputs:
    - writer: TensorBoard SummaryWriter or compatible interface.
    - avgs: Dict from average_losses with keys: avg_step_loss, avg_rec_loss, avg_vq_loss, avg_ntp_loss.
    - metrics_values: Dict from compute_metrics with keys: mae, rmsd, gdtts, tm_score, perplexity.
    - epoch: Current epoch index (int) used as global step for TB scalars.
    - activation_percent: Codebook activation percent already scaled 0-100.
    - include_ntp: Whether to also log NTP loss (validation currently does, training previously omitted).

    Behavior:
    - Logs stable set of scalars used across training/validation.
    - Perplexity is only logged if it’s a finite number (metric enabled).
    """
    if writer is None:
        return

    writer.add_scalar('loss/total', avgs['avg_step_loss'], epoch)
    writer.add_scalar('loss/rec_loss', avgs['avg_rec_loss'], epoch)
    writer.add_scalar('loss/vq', avgs['avg_vq_loss'], epoch)
    if 'avg_tik_tok_padding_loss' in avgs:
        writer.add_scalar('loss/tik_tok_padding', avgs['avg_tik_tok_padding_loss'], epoch)
    if include_ntp:
        writer.add_scalar('loss/ntp', avgs['avg_ntp_loss'], epoch)

    # Unscaled epoch logs (if present)
    if 'avg_unscaled_step_loss' in avgs:
        writer.add_scalar('unscaled_loss/total', avgs['avg_unscaled_step_loss'], epoch)
    if 'avg_unscaled_rec_loss' in avgs:
        writer.add_scalar('unscaled_loss/rec_loss', avgs['avg_unscaled_rec_loss'], epoch)
    if 'avg_unscaled_vq_loss' in avgs:
        writer.add_scalar('unscaled_loss/vq', avgs['avg_unscaled_vq_loss'], epoch)
    if 'avg_unscaled_tik_tok_padding_loss' in avgs:
        writer.add_scalar('unscaled_loss/tik_tok_padding', avgs['avg_unscaled_tik_tok_padding_loss'], epoch)
    if include_ntp and 'avg_unscaled_ntp_loss' in avgs:
        writer.add_scalar('unscaled_loss/ntp', avgs['avg_unscaled_ntp_loss'], epoch)

    writer.add_scalar('metric/mae', metrics_values['mae'], epoch)
    writer.add_scalar('metric/rmsd', metrics_values['rmsd'], epoch)
    writer.add_scalar('metric/gdtts', metrics_values['gdtts'], epoch)
    writer.add_scalar('metric/tm_score', metrics_values['tm_score'], epoch)

    writer.add_scalar('codebook_activation', activation_percent, epoch)

    perplexity = metrics_values.get('perplexity', float('nan'))
    if perplexity == perplexity:  # check not NaN
        writer.add_scalar('metric/perplexity', perplexity, epoch)
