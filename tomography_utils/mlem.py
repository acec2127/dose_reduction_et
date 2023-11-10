"""
Implmentation of a batched version of the Maximum Likelihood Estimator Method. 
"""

from torch import Tensor 
import torch

def mlem_precomputed(sinograms: Tensor, 
                        proj_matrix: Tensor, 
                        reltol_l2 = 1e-10,
                        max_iter = 100, 
                        device = torch.device('cuda')
                        ):
    
    """
    MLEM algorithm without regularization with precomputed matrices

    Parameters
    sinograms     : batch flattened sinograms, e.g., with np.ravel (original shape is (nproj,nshifts))
                    shape : N x P (batch x lines of response) = (batch, nproj*nshifts)
    proj_matrix   : projection matrix, shape (P, D) = (lines of responses, pixels)
    reltol_l2 : relative l2 error for when to stop iterations
    max_iter  : maximum number of iterations

    """
    # N x P
    rec = torch.ones((sinograms.shape[0], proj_matrix.shape[1])).to(device)
    reltol_l2_curr = torch.tensor(float("Inf")).to(device)
    iter = 0

    while (iter < max_iter and reltol_l2 < torch.max(reltol_l2_curr).item()):
        # N x P @ P x D -> N x P
        sinograms_current = torch.matmul(rec, proj_matrix.T)
        # N x D @ D x P -> N x P
        rec_next = rec * torch.matmul(sinograms/sinograms_current, proj_matrix)
        rec_norm = torch.linalg.vector_norm(rec, dim=-1)
        error_norm = torch.linalg.vector_norm(rec_next - rec, dim=-1)
        reltol_l2_curr = torch.where(rec_norm > 0, error_norm / rec_norm, torch.full(rec_norm.shape, float("inf")).to(device))
        rec = rec_next
        iter += 1

    return rec
