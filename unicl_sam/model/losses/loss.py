import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class Lisa_DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self,     
                inputs: torch.Tensor,
                targets: torch.Tensor,
                num_masks: float,
                scale=1000,  # 100000.0,
                eps=1e-6):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1, 2)
        targets = targets.flatten(1, 2)
        numerator = 2 * (inputs / scale * targets).sum(-1)
        denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        loss = loss.sum() / (num_masks + 1e-8)
        return loss

class Lisa_CELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self,     
                inputs: torch.Tensor,
                targets: torch.Tensor,
                num_masks: float):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
        return loss

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        loss = F.mse_loss(inputs, targets, reduction="mean")
        return loss

class BatchIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos.flatten(1), targets) + torch.einsum(
            "nc,mc->nm", neg.flatten(1), (1 - targets)
        )

        return loss / hw

class topk_crossEntrophy(nn.Module):
    def __init__(self, top_k=0.7):
        super(topk_crossEntrophy, self).__init__()
        self.loss = nn.NLLLoss()
        self.top_k = top_k
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, target):
        softmax_result = self.softmax(input)

        loss = torch.autograd.Variable(torch.Tensor(1).zero_()).cuda()
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            gt = torch.unsqueeze(gt, 0)
            cost = self.loss(pred, gt)
            cost = torch.unsqueeze(cost, 0)
            loss = torch.cat((loss, cost), 0)

        loss = loss[1:]
        if self.top_k == 1.0:
            valid_loss = loss

        # import pdb;pdb.set_trace()
        index = torch.topk(loss, int(self.top_k * loss.size()[0]))
        valid_loss = loss[index[1]]

        return torch.mean(valid_loss)

def tversky_loss(pred,
                 target,
                 valid_mask,
                 alpha=0.3,
                 beta=0.7,
                 smooth=1,
                 class_weight=None,
                 ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            tversky_loss = binary_tversky_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                alpha=alpha,
                beta=beta,
                smooth=smooth)
            if class_weight is not None:
                tversky_loss *= class_weight[i]
            total_loss += tversky_loss
    return total_loss / num_classes

def binary_tversky_loss(pred,
                        target,
                        valid_mask,
                        alpha=0.3,
                        beta=0.7,
                        smooth=1):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(pred, target) * valid_mask, dim=1)
    FP = torch.sum(torch.mul(pred, 1 - target) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - pred, target) * valid_mask, dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - tversky

class TverskyLoss(nn.Module):
    """TverskyLoss. This loss is proposed in `Tversky loss function for image
    segmentation using 3D fully convolutional deep networks.

    <https://arxiv.org/abs/1706.05721>`_.
    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        alpha(float, in [0, 1]):
            The coefficient of false positives. Default: 0.3.
        beta (float, in [0, 1]):
            The coefficient of false negatives. Default: 0.7.
            Note: alpha + beta = 1.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_tversky'.
    """

    def __init__(self,
                 smooth=1,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 alpha=0.3,
                 beta=0.7,
                 loss_name='loss_tversky'):
        super().__init__()
        self.smooth = smooth
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        assert (alpha + beta == 1.0), 'Sum of alpha and beta but be 1.0!'
        self.alpha = alpha
        self.beta = beta
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * tversky_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            alpha=self.alpha,
            beta=self.beta,
            smooth=self.smooth,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss
