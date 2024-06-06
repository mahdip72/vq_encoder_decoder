import torch
import torch.nn as nn
from torchmetrics import Metric


class GDTTS(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_gdt", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, "Predictions and target must have the same shape"

        thresholds = [1.0, 2.0, 4.0, 8.0]
        gdt_scores = torch.zeros(len(thresholds))

        distances = torch.sqrt(torch.sum((preds - target) ** 2, dim=-1))

        for i, threshold in enumerate(thresholds):
            gdt_scores[i] = (distances < threshold).float().mean()

        self.sum_gdt += gdt_scores.mean()
        self.total += 1

    def compute(self):
        return self.sum_gdt / self.total


class LDDT(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_lDDT", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Compute the lDDT score for the predicted and true structures
        lddt_score = self.compute_lddt(preds, target)
        self.sum_lDDT += lddt_score
        self.total += 1

    def compute(self):
        # Compute the final lDDT score based on the state variables
        return self.sum_lDDT / self.total

    @staticmethod
    def compute_lddt(preds: torch.Tensor, target: torch.Tensor, thresholds=(0.5, 1, 2, 4)):
        """
        Compute the local Distance Difference Test (lDDT) score for the predicted and true protein structures.

        Parameters:
        preds (torch.Tensor): The predicted protein structures. Shape: (N, 3) where N is the number of atoms.
        target (torch.Tensor): The true protein structures. Shape: (N, 3) where N is the number of atoms.
        thresholds (tuple, optional): The distance thresholds for the lDDT score. Default is (0.5, 1, 2, 4).

        Returns:
        lddt_score (torch.Tensor): The computed lDDT score. A single floating point number.
        """
        # Ensure the tensors are on the same device and have the same shape
        assert preds.device == target.device
        assert preds.shape == target.shape

        # Compute the pairwise distance matrices for the predicted and actual structures
        preds_dist = torch.cdist(preds, preds)
        target_dist = torch.cdist(target, target)

        # Initialize a tensor to hold the per-atom lDDT scores
        lddt_scores = torch.zeros(preds.shape[0], device=preds.device)

        # For each atom, compute the lDDT score
        for i in range(preds.shape[0]):
            # Get the neighbors within a 15 Ångström radius
            preds_neighbors = preds_dist[i] <= 15
            target_neighbors = target_dist[i] <= 15

            # Compute the lDDT score for each threshold
            for threshold in thresholds:
                # Compute the fraction of distances in the predicted structure that fall within the threshold
                # of the distances in the actual structure
                lddt_scores[i] += ((preds_dist[i, preds_neighbors] - target_dist[i, target_neighbors]).abs() < threshold).float().mean()

            # Average the lDDT scores across all thresholds
            lddt_scores[i] /= len(thresholds)

        # Average the per-atom lDDT scores across all atoms to get the overall lDDT score
        lddt_score = lddt_scores.mean()

        return lddt_score


if __name__ == "__main__":
    # Example usage
    mean_lddt = LDDT()

    # Example true and predicted values (coordinates)
    y_true = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred = torch.tensor([[1.1, 0.2, 9.1], [3.9, 4.9, 5.9]])

    # Update the metric with predictions and true values
    mean_lddt.update(y_pred, y_true)

    # Compute the final GDT-TS score
    mean_lddt_score = mean_lddt.compute()
    print(mean_lddt_score.item())
