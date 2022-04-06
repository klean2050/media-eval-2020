import torch
import torch.nn.functional as F
import torch.nn as nn
import sys


"""
An implementation of focal loss for the multi-class, multilabel case.
It is based off of the Tensorflow implementation for SigmoidFocalCrossEntropy, and extended to the multilabel case.
Input is expected to be of shape (num_batches, num_classes). Output will also be of shape (num_batches, num_classes).
Note that alpha can be given as a float, which is applied uniformily to each class logit,
or it can be explicitly set for each class by providing a vector of length equal to the number of classes. 
"""


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:

    # Numerical stability
    input = torch.clamp(input, min=eps, max=-1 * eps + 1.0)

    # Get the cross_entropy for each entry
    bce = F.binary_cross_entropy(input, target, reduction="none")
    p_t = (target * input) + ((1 - target) * (1 - input))

    # If alpha < 0, set the alpha factor (a_t) to be uniformly 1 for all classes
    alpha_factor = 1 if alpha < 0 else target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = torch.pow((1.0 - p_t), gamma)

    return alpha_factor * modulating_factor * bce


class FocalLoss(nn.Module):
    """
    Args:
        alpha: Focal loss weight, as defined in https://arxiv.org/abs/1708.02002 [Float]
        gamma: Focal loss focusing parameter. [Float]
        reduction: How to reduce from element-wise loss to total loss. [String]
    Returns:
        Total loss as a single float value.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = True,
        reduction: str = "none",
    ) -> None:

        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.from_logits: bool = from_logits
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        y = torch.sigmoid(input) if self.from_logits else input
        loss = focal_loss(y, target, self.alpha, self.gamma, eps=self.eps)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            # Default to batch average
            return torch.mean(torch.sum(loss, axis=-1))


"""
This is an implementation of class-balanced loss, from the paper by Cui et al
(https://arxiv.org/abs/1901.05555). 

Note that samples per class is simply a raw count of the number of times each class label 
is present in the input dataset. Here, the order is the same as in the TAGS parameter.
"""


def CB_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    samples_per_cls,
    beta,
    gamma,
    loss_function,
):

    # Calculate the class weight vector
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(weights) * len(samples_per_cls)
    weights = weights.to(target.device)

    if loss_function == "focal_loss":
        loss = focal_loss(input, target, alpha=-1, gamma=gamma)
        return weights.unsqueeze(0) * loss

    elif loss_function == "bce":
        criterion = nn.BCELoss(reduce=False)
        loss = criterion(input, target)
        return weights.unsqueeze(0) * loss

    else:
        print("Invalid loss specified in ClassBalancedLoss.")
        sys.exit()


"""

Args:
input: A torch tensor of class predictions (logits if from_logits is True, else sigmoid outputs). Shape: (batch_size, num_classes)
target: A torch tensor of binary ground truth labels. Shape: (batch_size, num_classes)

class_weights: Vector of integer class counts for the input training dataset. Shape: (1, num_classes)
    Example: class_1 has 5 instances, class_2 has 3 instances, class_3 has 4 instances. class_weights = [5,3,4]
beta: hyperparameter for class-balanced loss, as defined in https://arxiv.org/pdf/1901.05555. Float.
gamma: Focal loss focusing parameter, as defined in https://arxiv.org/abs/1708.02002. Float.

reduction: How to reduce from element-wise loss to total loss. String.
loss_function: Base loss function to use. String.

Returns:
Total loss as a single float value.

"""


class ClassBalancedLoss(nn.Module):
    def __init__(
        self,
        class_weights,
        beta: float,
        gamma: float = 2.0,
        reduction: str = "none",
        from_logits: bool = True,
        loss_function: str = "focal_loss",
    ) -> None:
        super(ClassBalancedLoss, self).__init__()
        self.beta: float = beta
        self.gamma: float = gamma
        self.samples_per_class = torch.as_tensor(class_weights)
        self.no_of_classes = len(class_weights)
        self.from_logits: bool = from_logits
        self.loss_function = loss_function
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # If necessary, apply an activation function to raw input logits
        if self.from_logits:
            y = torch.sigmoid(input)
        else:
            y = input

        loss = CB_loss(
            y, target, self.samples_per_class, self.beta, self.gamma, self.loss_function
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            # Default to batch average
            return torch.mean(torch.sum(loss, axis=-1))


"""
Below is an implementation of Distribution-Balanced Loss first presented by Wu et al (https://arxiv.org/pdf/2007.09654).
Both focal loss and BCE are implemented as possible base loss functions.
Note that kappa here has been explicitly set to 0, to make working directly with sigmoid outputs easier (as opposed to using logits).
"""


def DB_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    samples_per_cls,
    v_i=None,
    alpha=0.25,
    gamma=2.0,
    loss_function="focal_loss",
    r_alpha=0.1,
    r_beta=10.0,
    r_mu=0.2,
    nt_lambda=2.0,
):

    """
    Negative-tolerant regularization: apply the class-specific bias v_i to input logits.
    Additionally scale transformed negative logits by lambda.

    Note that if v_i is not given, explicitely set v_i to zero for all classes.
    """

    if v_i == None:
        num_classes = input.shape[-1]
        v_i = torch.zeros(num_classes)
    else:
        v_i = torch.as_tensor(v_i)
    assert v_i.shape[-1] == input.shape[-1]
    v_i = v_i.to(target.device)

    os_input = input - v_i  # offset input by class-specific bias
    ls_input = (
        (target) + ((1 - target) * (nt_lambda))
    ) * os_input  # Further scale negative logit
    p_input = torch.sigmoid(
        ls_input
    )  # Apply activation function to logits to get probabilities

    # nt_lambda is the regularizer for the negative terms
    nt_lambda = (target) + (
        (1 - target) * (1.0 / nt_lambda)
    )  # scale the negative classes by 1/lambda

    """Rebalanced weighting"""

    # Rebalancing terms (r_k)
    freq_inv = (1.0 / samples_per_cls).to(target.device)
    repeat_rate = torch.sum(target * freq_inv, dim=1, keepdim=True)
    pos_weight = freq_inv.clone().detach().unsqueeze(0) / repeat_rate
    # pos and neg are equally treated
    r_k = torch.sigmoid(r_beta * (pos_weight - r_mu)) + r_alpha

    if loss_function == "focal_loss":
        loss = focal_loss(p_input, target, alpha=alpha, gamma=gamma)

    elif loss_function == "bce":
        criterion = nn.BCELoss(reduce=False)
        loss = criterion(p_input, target)

    else:
        print("Invalid loss function specified in DistributionBalancedLoss.")
        sys.exit()

    # Compute distribution-balanced loss
    loss = r_k * nt_lambda * loss

    return loss


"""

Args:
input: A torch tensor of class predictions (logits). Shape: (batch_size, num_classes)
target: A torch tensor of binary ground truth labels. Shape: (batch_size, num_classes)

class_weights: vector of integer class counts for the input training dataset. Shape: (1, num_classes)
    Example: class_1 has 5 instances, class_2 has 3 instances, class_3 has 4 instances. class_weights = [5,3,4]
alpha: Focal loss weight, as defined in https://arxiv.org/abs/1708.02002. Float.
gamma: Focal loss focusing parameter. Float.

rebalance_alpha: rebalancing alpha for the rebalancing weight (r_hat_k), as defined in https://arxiv.org/pdf/2007.0965. Float.
reblance_beta: rebalancing beta for the rebalancing weight. Float.
rebalancing_mu: rebalancing mu for the rebalancing weight. Float.

nt_lambda: Lambda for negative-tolerant regularization. Float.

reduction: How to reduce from element-wise loss to total loss. String.
loss_function: Base loss function to use. String.

Returns:
Total loss as a single float value.

"""


class DistributionBalancedLoss(nn.Module):
    def __init__(
        self,
        class_weights,
        v_i=None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        rebalance_alpha: float = 0.1,
        rebalance_beta: float = 10.0,
        rebalance_mu: float = 0.2,
        nt_lambda: float = 2.0,
        reduction: str = "none",
        loss_function: str = "focal_loss",
    ) -> None:
        super(DistributionBalancedLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.samples_per_class = class_weights

        self.r_alpha = rebalance_alpha
        self.r_beta = rebalance_beta
        self.r_mu = rebalance_mu

        self.v_i = v_i
        self.nt_lambda = nt_lambda

        self.loss_function = loss_function
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        loss = DB_loss(
            input,
            target,
            self.samples_per_class,
            self.v_i,
            self.alpha,
            self.gamma,
            self.loss_function,
            self.r_alpha,
            self.r_beta,
            self.r_mu,
            self.nt_lambda,
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            # Default to batch average
            return torch.mean(torch.sum(loss, axis=-1))
