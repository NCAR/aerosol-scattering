"""
functions for calculating a parametric PDF using a Bernstein basis
"""


import torch
import numpy as np


def binomial_coefficient_pytorch(n, k):
    """
    Computes the binomial coefficient C(n, k) using PyTorch's lgamma function.

    Args:
        n (torch.Tensor or int): The total number of items.
        k (torch.Tensor or int): The number of items to choose.

    Returns:
        torch.Tensor: The binomial coefficient C(n, k).
    """
    if isinstance(n, int):
        n = torch.tensor(float(n))
    if isinstance(k, int):
        k = torch.tensor(float(k))

    # Handle cases where k is out of range (k < 0 or k > n) where C(n,k) = 0
    if k < 0 or k > n:
        return torch.tensor(0.0)

    # Calculate using lgamma
    log_binom = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return torch.exp(log_binom)

def chebyshev_poly(order:int,device=None,dtype=None)->torch.tensor:
    """
    Generates the coefficient matrix for chebyshev polynomials
    where the polynomial is generated using matrix multiplication
    [1, x, x^2, ... , x^N] @ coef_mat
    """

    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float

    if order == 0:
        return torch.tensor([1],dtype=dtype,device=device)
    elif order == 1:
        return torch.tensor([1,0],[0,1],dtype=dtype,device=device)
    else:
        coef_mat = torch.zeros((order+1,order+1),dtype=dtype,device=device)
        coef_mat[0,0] = 1
        coef_mat[1,1] = 1
        for idx in range(2,order+1):
            coef_mat[1:,idx] = 2*coef_mat[:-1,idx-1]
            coef_mat[:,idx] -= coef_mat[:,idx-2]
    
    return coef_mat

def chebyshev_poly_int(order:int)->torch.tensor:
    """
    Generates the coefficient matrix for integral of chebyshev polynomials
    where the integral of the polynomial is generated using matrix multiplication
    [x, x^2, ... , x^(N+1)] @ coef_mat
    """
    if order == 0:
        return torch.tensor([1])
    elif order == 1:
        return torch.tensor([1,0],[0,0.5])
    else:
        # rescale by the integral polynomial power
        coef_mat = chebyshev_poly(order)/(torch.arange(order+1)+1)[:,None]
        
    return coef_mat


def eval_2var_chebyshev_poly(fit_coef_mat: torch.tensor, 
                             cheb_coef_mat1: torch.tensor, x1: torch.tensor,
                             cheb_coef_mat2: torch.tensor, x2: torch.tensor):
    """
    evaluate a two variable chebyshev polynomial for matrices defined using
    chebyshev_poly.  This function is not optimized for speed because it creates
    the power coefficients in it, but it makes evaluation easy and documents the
    flow of how the calculation can be done.

    """
    x1p = x1.reshape(-1,1)**torch.arange(0,cheb_coef_mat1.shape[0],device=cheb_coef_mat1.device,dtype=cheb_coef_mat1.dtype)
    x2p = x2.reshape(-1,1)**torch.arange(0,cheb_coef_mat2.shape[0],device=cheb_coef_mat2.device,dtype=cheb_coef_mat2.dtype)

    x1c = x1p @ cheb_coef_mat1
    x2c = x2p @ cheb_coef_mat2

    xc = torch.einsum('bi,bj->bij', x1c, x2c).view(x1.shape[0], -1)
    return torch.sum(fit_coef_mat*xc,dim=1)
    
