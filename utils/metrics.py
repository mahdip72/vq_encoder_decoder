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


if __name__ == "__main__":
    # Example usage
    gdt_ts = GDTTS()

    # Example true and predicted values (coordinates)
    y_true = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred = torch.tensor([[1.1, 2.1, 3.1], [3.9, 4.9, 5.9]])

    # Update the metric with predictions and true values
    gdt_ts.update(y_pred, y_true)

    # Compute the final GDT-TS score
    gdt_ts_score = gdt_ts.compute()
    print(gdt_ts_score.item())
