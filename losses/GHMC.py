import torch
import torch.nn as nn
import torch.nn.functional as F

def _expand_binary_labels(labels, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1

    return bin_labels

class GHMC(nn.Module):
    def __init__(
            self,
            bins=30,
            momentum=0.75,
            use_sigmoid=True,
            loss_weight=1.0,
            logger=None):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.logger = logger
        # self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            # self.acc_sum = [0.0 for _ in range(bins)]
            self.register_buffer("acc_sum", torch.zeros(bins))

        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        """
        # print(self.acc_sum)
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label

        if pred.dim() != target.dim():
            target = _expand_binary_labels(target, pred.size(-1))
        target = target.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        tot = max(float(target.numel()), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            #print(i,num_in_bin)
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights /= n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight