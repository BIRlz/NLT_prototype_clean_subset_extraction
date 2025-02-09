
import sys

import random
import time
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
# from scipy.spatial import KDTree
# from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader

import ot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`

    """

    def __init__(self, eps, max_iter, dis='cos', reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.dis = dis

    def forward(self, x, y, q):
        if self.dis == 'cos':
            C = self._cost_matrix(x, y, 'cos')
        elif self.dis == 'euc':
            C = self._cost_matrix(x, y, 'euc')
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).to(device).squeeze()
        
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).to(device).squeeze()
        
        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)

        actual_nits = 0
        thresh = 1e-1
        
        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v, q), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v, q).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits = actual_nits + 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V, q))
        cost = torch.sum(pi * C, dim=(-2, -1))

        return cost, pi

    def M(self, C, u, v, q):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2) - torch.log(q)) / self.eps

    def _cost_matrix(self,x, y, dis, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if dis == 'cos':
            C = 1 - d_cosine(x_col, y_lin)
        elif dis == 'euc':
            C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)

        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
    
def log_ratio_sinkhorn(a, b, q, epsilon=0.1, max_iter=200):
    # A: cost matrix, a: source distribution, b: target distribution
    # q: reference distribution, epsilon: regularization parameter, max_iter: maximum number of iterations
    # returns: P: optimal transport matrix, u: row scaling vector, v: column scaling vector
    
    # initialize u and v

    x_col = a.unsqueeze(-2)
    y_lin = b.unsqueeze(-3)
    
    A = 1 - d_cosine(x_col, y_lin)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    # compute the kernel matrix
    K = torch.exp(-A / epsilon) * q
    
    # iterate until convergence or max_iter
    for i in range(max_iter):
        u0 = u
        u = a / torch.matmul(K, v) # update u
        v = b / torch.matmul(K.t(), u) # update v

        if (u - u0).abs().mean() < 1e-1: # if error is small enough, stop
            break
    
    # Convert results back to NumPy arrays
    
    P = torch.matmul(u, v.T) * K    
    # torch.diag(u) @ K @ torch.diag(v) # compute the transport matrix

    return torch.sum(P * A, dim=(-2, -1)), P


class SinkhornDistance_uniform(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`

    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance_uniform, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, mu=None, nu=None):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0] 
        
        if mu is None:
            # both marginals are fixed with equal weights
            mu = torch.empty(batch_size, x_points, dtype=torch.float,
                            requires_grad=False).fill_(1.0 / x_points).to(device).squeeze()
            
        if nu is None:
            nu = torch.empty(batch_size, y_points, dtype=torch.float,
                            requires_grad=False).fill_(1.0 / y_points).to(device).squeeze()

        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        for i in range(self.max_iter):

            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            # err_iter.append(err.item())
            actual_nits += 1

            U, V = u, v
            pi = torch.exp(self.M(C, U, V))
            # pi_iter.append(pi.detach().numpy())

            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))

        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi
     #   return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C =  1-d_cosine(x_col , y_lin)
        # C= torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

if __name__ == "__main__":
    
    # generate some random data
    np.random.seed(0) # set the random seed for reproducibility
    n = 10 # dimension of the distributions

    A = torch.rand(128, 10) # cost matrix

    a = torch.rand(128, 512) # source distribution
    b = torch.rand(10, 512) # target distribution

    a = a / a.sum() # normalize a
    b = b / b.sum() # normalize b
    
    epsilon = 0.1 # regularization parameter
    max_iter = 1000 # maximum number of iterations

    # q = torch.ones((128, 10))
    
    q = torch.rand((128, 10))
    # q = q / q.sum()

    # compute the optimal transport matrix, row scaling vector, and column scaling vector
    loss1, P1 = log_ratio_sinkhorn(a, b, q)
    _p_label_1 = torch.argmax(P1.log().softmax(-1), -1)

    ot_criterion_2 = SinkhornDistance_uniform(epsilon, max_iter)
    loss2, P2 = ot_criterion_2(a, b)
    _p_label_2 = torch.argmax(P2.log().softmax(-1), -1)

    ot_criterion_3 = SinkhornDistance(epsilon, max_iter)
    loss3, P3 = ot_criterion_3(a, b, q)
    _p_label_3 = torch.argmax(P3.log().softmax(-1), -1)

    print(_p_label_1 == _p_label_2)
    print(_p_label_2 == _p_label_3)
    print(_p_label_1 == _p_label_3)


    import pdb;pdb.set_trace()