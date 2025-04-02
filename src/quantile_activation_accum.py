import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def weighted_quantile_vectorized(values, quantiles, interpolation="linear"):
    """Compute the weighted quantile for each row in a 2D tensor vectorized.
    Args:
        values (Tensor): 2D Tensor of shape (N_c, N_samples) containing data points.
        quantiles (float or Tensor): Which quantile to compute, range between 0 and 1.
        interpolation (str): Interpolation method, 'linear' by default.

    Returns:
        Tensor: The computed quantile values for each row.
    """
    N_c, N_samples = values.shape

    C = 100
    minval = -C * torch.ones_like(values[:, 0]).unsqueeze(1)
    maxval = C * torch.ones_like(values[:, 0]).unsqueeze(1)
    values_context = torch.concatenate([values, minval, maxval], dim=1)

    # Create masks for positive and negative values
    positive_mask = values_context >= 0
    negative_mask = values_context < 0

    # Sum of positive and negative masks
    sum_positives = positive_mask.sum(dim=1, keepdim=True)
    sum_negatives = negative_mask.sum(dim=1, keepdim=True)

    # Weights for positives and negatives
    positive_weights = positive_mask.float() / torch.where(
        sum_positives == 0, torch.ones_like(sum_positives), sum_positives
    )
    negative_weights = negative_mask.float() / torch.where(
        sum_negatives == 0, torch.ones_like(sum_negatives), sum_negatives
    )

    # Equalize total weight of positives and negatives if both are present
    weights = torch.where(positive_mask, positive_weights, negative_weights)
    if (sum_positives > 0).logical_and(sum_negatives > 0).all():
        weights *= N_samples

    # Sorting values and corresponding weights for each row
    sorted_indices = torch.argsort(values_context, dim=1)
    sorted_values = torch.gather(values_context, 1, sorted_indices)
    sorted_weights = torch.gather(weights, 1, sorted_indices)

    # Cumulative weights calculation
    cum_weights = torch.cumsum(sorted_weights, dim=1)
    total_weight = cum_weights[:, -1].unsqueeze(1)  # Total weights per row

    # Interpolation: find the closest ranks for the given quantiles
    target_weights = quantiles.unsqueeze(0) * total_weight
    above = torch.searchsorted(cum_weights, target_weights, right=True)
    below = above - 1

    # Clamp indices to be within bounds
    below = torch.clamp(below, 0, N_samples - 1)
    above = torch.clamp(above, 0, N_samples - 1)

    # Gather the values for linear interpolation
    value_below = torch.gather(sorted_values, 1, below)
    value_above = torch.gather(sorted_values, 1, above)
    weight_below = torch.gather(cum_weights, 1, below)
    weight_above = torch.gather(cum_weights, 1, above)

    # Linear interpolation
    interp_fraction = (target_weights - weight_below) / (weight_above - weight_below + 1e-6)
    quantile_values = value_below + (value_above - value_below) * interp_fraction

    out = quantile_values.clone()

    return out


def weighted_sample(data, weights, num_samples):
    """
    Samples `num_samples` for each row of `data` using the provided `weights`.

    Parameters:
    data (torch.Tensor): The data tensor with shape (N_c, N_samples).
    weights (torch.Tensor): The weights tensor with shape (N_c, N_samples).
    num_samples (int): Number of samples to draw from each row.

    Returns:
    torch.Tensor: The sampled data with shape (N_c, num_samples).
    """
    # Ensure the weights sum to 1 along the last dimension
    weights_normalized = weights / weights.sum(dim=1, keepdim=True)

    # Sample indices based on weights
    sampled_indices = torch.multinomial(weights_normalized, num_samples, replacement=True)

    # Gather the samples using the sampled indices
    sampled_data = torch.gather(data, 1, sampled_indices)

    return sampled_data


def weighted_vectorized_kde(values, eval_points, bandwidth):
    """
    Perform vectorized Kernel Density Estimation using a Gaussian kernel for sorted data.

    Parameters:
    sorted_data (Tensor): Sorted data points from which to estimate the density of shape (n_samples,).
    eval_points (Tensor): Points at which to estimate the density of shape (n_eval,).
    bandwidth (float): The bandwidth of the kernel.

    Returns:
    Tensor: Density estimates at each point in eval_points.
    """
    # Compute the entire kernel matrix in a vectorized form
    C = 100
    minval = -C * torch.ones_like(values[:, 0]).unsqueeze(1)
    maxval = C * torch.ones_like(values[:, 0]).unsqueeze(1)
    values_context = torch.concatenate([values, minval, maxval], dim=1)

    # Create masks for positive and negative values
    positive_mask = values_context >= 0
    negative_mask = values_context < 0

    # Sum of positive and negative masks
    sum_positives = positive_mask.sum(dim=1, keepdim=True)
    sum_negatives = negative_mask.sum(dim=1, keepdim=True)

    # Weights for positives and negatives
    positive_weights = positive_mask.float() / torch.where(
        sum_positives == 0, torch.ones_like(sum_positives), sum_positives
    )
    negative_weights = negative_mask.float() / torch.where(
        sum_negatives == 0, torch.ones_like(sum_negatives), sum_negatives
    )

    # Equalize total weight of positives and negatives if both are present
    weights = torch.where(positive_mask, positive_weights, negative_weights)
    # weights is of shape (N_c, n_samples)

    values_context = weighted_sample(values_context, weights, 1000)

    kernel_matrix = eval_points.unsqueeze(1) - values_context.unsqueeze(
        2
    )  # Shape (N_c, n_samples, n_eval)
    kernel_matrix = torch.exp(-0.5 * (kernel_matrix**2) / (bandwidth**2))
    densities = kernel_matrix.sum(dim=1) / values_context.shape[1]
    h = bandwidth
    torch_pi = torch.acos(torch.zeros(1)).item() * 2
    c = torch.sqrt(torch.tensor(2 * torch_pi))
    densities = densities / (h * c)

    out = densities.clone()
    # del values_context, kernel_matrix, densities, positive_mask, negative_mask, sum_positives, sum_negatives, positive_weights, negative_weights, weights
    # torch.cuda.empty_cache()
    # gc.collect()
    return out


class CustomQuantFunction1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, quant_list, out_context):
        """
        - inp : tensor of shape (B, N_c)
        """
        with torch.no_grad():
            inp_context = inp.clone().transpose(0, 1)
            quantiles = weighted_quantile_vectorized(out_context, quant_list)
            densities = weighted_vectorized_kde(out_context, quantiles, 0.5)
            out = quant_list[
                torch.searchsorted(quantiles, inp_context, side="right").clamp_(
                    0, quantiles.shape[1] - 1
                )
            ]
            inv_derivative_out = torch.gather(
                densities,
                1,
                torch.searchsorted(quant_list, out, side="right").clamp_(0, quantiles.shape[1] - 1),
            )

            # shaping the tensors
            output = out.clone().transpose(0, 1)
            inv_derivative_out = inv_derivative_out.clone().transpose(0, 1)

        ctx.save_for_backward(inv_derivative_out)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inv_derivative_out = ctx.saved_tensors[0]

        grad_input = grad_output * inv_derivative_out

        return grad_input, None, None


class quantile_activation_1d(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(n_features, affine=True, track_running_stats=True)
        self.bn_out = nn.BatchNorm1d(n_features, affine=True, track_running_stats=True)
        self.quantile_list = nn.Parameter(torch.linspace(0.01, 0.99, 102), requires_grad=False)
        self.num_samples = 1000
        self.context_distribution = nn.Parameter(
            torch.randn(n_features, self.num_samples), requires_grad=False
        )
        self.flat_init_context = False

    def forward(self, x):
        out = self.bn_in(x)
        if self.training:
            self.update_context(out)
        out = CustomQuantFunction1D.apply(out, self.quantile_list, self.context_distribution)
        out = self.bn_out(out)
        return out

    @torch.no_grad()
    def init_context(self, x):
        """
        x : tensor of shape (B, N_c)
        self.context_distribution : tensor of shape (N_c, 1000)
        """
        weights = torch.ones_like(x.transpose(0, 1))
        ind_sample = torch.multinomial(weights, self.num_samples, replacement=True)
        tmp = torch.gather(x.transpose(0, 1), 1, ind_sample)
        self.context_distribution.data.copy_(tmp)
        self.flat_init_context = True

    @torch.no_grad()
    def update_context(self, x):
        """
        x : tensor of shape (B, N_c)
        """
        if self.flat_init_context:
            weights = torch.ones_like(x.transpose(0, 1))
            ind_sample = torch.multinomial(weights, self.num_samples, replacement=True)
            tmp1 = torch.gather(x.transpose(0, 1), 1, ind_sample)
            tmp2 = self.context_distribution.data
            tmp = torch.cat([tmp1, tmp2], dim=1)
            weights = torch.ones_like(tmp)
            ind_sample = torch.multinomial(weights, self.num_samples, replacement=True)
            tmp = torch.gather(tmp, 1, ind_sample)
            self.context_distribution.data.copy_(tmp)
        else:
            self.init_context(x)


class CustomQuantFunction2D(torch.autograd.Function):
    def forward(ctx, inp, quant_list, out_context):
        """
        - inp : tensor of shape (B, N_c)
        """
        with torch.no_grad():
            shape = (inp.transpose(0, 1)).shape
            inp_context = inp.clone().transpose(0, 1).flatten(1)
            quantiles = weighted_quantile_vectorized(out_context, quant_list)
            densities = weighted_vectorized_kde(out_context, quantiles, 0.5)
            out = quant_list[
                torch.searchsorted(quantiles, inp_context, side="right").clamp_(
                    0, quantiles.shape[1] - 1
                )
            ]
            inv_derivative_out = torch.gather(
                densities,
                1,
                torch.searchsorted(quant_list, out, side="right").clamp_(0, quantiles.shape[1] - 1),
            )

            # shaping the tensors
            output = out.clone().reshape(shape).transpose(0, 1)
            inv_derivative_out = inv_derivative_out.clone().reshape(shape).transpose(0, 1)

        ctx.save_for_backward(inv_derivative_out)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inv_derivative_out = ctx.saved_tensors[0]

        # grad_input = grad_output
        grad_input = grad_output * inv_derivative_out

        return grad_input, None, None


class quantile_activation_2d(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        num_quantiles = 100
        self.bn = nn.BatchNorm2d(n_features, affine=True, track_running_stats=True)
        self.quant_list = nn.Parameter(
            torch.linspace(0, 1, num_quantiles + 2)[1:-1], requires_grad=False
        )
        self.bn_out = nn.BatchNorm2d(n_features, affine=True, track_running_stats=True)
        self.num_samples = 1000
        self.context_distribution = nn.Parameter(
            torch.randn(n_features, self.num_samples), requires_grad=False
        )
        self.flat_init_context = False

    def forward(self, x):
        # Batch normalization
        out = self.bn(x)
        if self.training:
            self.update_context(out)
        out = CustomQuantFunction2D.apply(out, self.quant_list, self.context_distribution)
        out = self.bn_out(out)
        return out

    @torch.no_grad()
    def init_context(self, x):
        """
        x : tensor of shape (B, N_c, H, W)
        """
        weights = torch.ones_like(x.transpose(0, 1).flatten(1))
        ind_sample = torch.multinomial(weights, self.num_samples, replacement=True)
        tmp = torch.gather(x.transpose(0, 1).flatten(1), 1, ind_sample)
        self.context_distribution.data.copy_(tmp)
        self.flat_init_context = True

    @torch.no_grad()
    def update_context(self, x):
        """
        x : tensor of shape (B, N_c, H, W)
        """
        if self.flat_init_context:
            weights = torch.ones_like(x.transpose(0, 1).flatten(1))
            ind_sample = torch.multinomial(weights, self.num_samples, replacement=True)
            tmp1 = torch.gather(x.transpose(0, 1).flatten(1), 1, ind_sample)
            tmp2 = self.context_distribution.data
            tmp = torch.cat([tmp1, tmp2], dim=1)
            weights = torch.ones_like(tmp)
            ind_sample = torch.multinomial(weights, self.num_samples, replacement=True)
            tmp = torch.gather(tmp, 1, ind_sample)
            self.context_distribution.data.copy_(tmp)
        else:
            self.init_context(x)
