from __future__ import print_function

import torch
import torch.nn as nn


class FeatLoss(nn.Module):
    """
    Compute the dot between two features as loss.
    """
    def __init__(self, reduce=True, reduction="mean"):
        super().__init__()
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, x, y):
        dot_result = torch.abs(torch.matmul(x, y.T))
        if self.reduce:
            if self.reduction == "mean":
                return dot_result.mean()
            elif self.reduction == "sum":
                return dot_result.sum()
            else:
                raise NotImplementedError
        else:
            return dot_result

if __name__ == "__main__":
    x = torch.randn([4, 128])
    y = torch.randn([4, 128])
    loss = FeatLoss()
    print(loss(x, y))
