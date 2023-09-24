import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor


class FocalLoss(nn.Module):
    """
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """
    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0,
                 ignore_index: int = -100) -> None:
        super(FocalLoss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError('Reduction must be one of: `mean`, `sum`, `none`.')
        
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)
    
    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)
        
        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]
        log_p = F.log_softmax(x, dim=-1)

        if self.label_smoothing == 0:
            ce = self.nll_loss(log_p, y)
        else:
            ce = (1 - self.label_smoothing) * self.nll_loss(log_p, y) - self.label_smoothing * log_p.mean(dim=-1)
        
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss