from __future__ import absolute_import
import torch
import torch.nn as nn

class IHLoss(nn.Module):
    """
    Loss for hierarchical relationship between multi-label learning.
    """

    def __init__(self, use_sigmoid=True, ignore_thres=0.1):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.ignore_thres = ignore_thres
    
    def forward(self, x):
        """
        Currently only implemented for [N, 2]. The first dimension
        is "basic"
        """
        assert x.shape[1] == 2
        logits = x
        if self.use_sigmoid:
            logits = logits.sigmoid()

        prob_error = (logits[:,1].view(-1) - logits[:,0].view(-1))

        loss = prob_error.clamp(min=0.0, max=self.ignore_thres).mean()

        return loss

if __name__=="__main__":
    p1 = [0.5,0.6,0.6,0.6]
    p2 = [0.8,0.6,0.6,0.4]
    logits = torch.stack([torch.tensor(p1), torch.tensor(p2)]).T
    print(logits)
    loss = IHLoss(False)
    print(loss(logits))
