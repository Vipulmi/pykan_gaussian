# import torch


# def B_batch(x, grid, k=0, extend=True, device='cpu'):
#     '''
#     evaludate x on B-spline bases
    
#     Args:
#     -----
#         x : 2D torch.tensor
#             inputs, shape (number of splines, number of samples)
#         grid : 2D torch.tensor
#             grids, shape (number of splines, number of grid points)
#         k : int
#             the piecewise polynomial order of splines.
#         extend : bool
#             If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
#         device : str
#             devicde
    
#     Returns:
#     --------
#         spline values : 3D torch.tensor
#             shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
#     Example
#     -------
#     >>> from kan.spline import B_batch
#     >>> x = torch.rand(100,2)
#     >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
#     >>> B_batch(x, grid, k=3).shape
#     '''
    
#     x = x.unsqueeze(dim=2)
#     grid = grid.unsqueeze(dim=0)
    
#     if k == 0:
#         value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
#     else:
#         B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
#         value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
#                     grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
#     # in case grid is degenerate
#     value = torch.nan_to_num(value)
#     return value



# def coef2curve(x_eval, grid, coef, k, device="cpu"):
#     '''
#     converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
#     Args:
#     -----
#         x_eval : 2D torch.tensor
#             shape (batch, in_dim)
#         grid : 2D torch.tensor
#             shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
#         coef : 3D torch.tensor
#             shape (in_dim, out_dim, G+k)
#         k : int
#             the piecewise polynomial order of splines.
#         device : str
#             devicde
        
#     Returns:
#     --------
#         y_eval : 3D torch.tensor
#             shape (batch, in_dim, out_dim)
        
#     '''
    
#     b_splines = B_batch(x_eval, grid, k=k)
#     y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    
#     return y_eval


# def curve2coef(x_eval, y_eval, grid, k):
#     '''
#     converting B-spline curves to B-spline coefficients using least squares.
    
#     Args:
#     -----
#         x_eval : 2D torch.tensor
#             shape (batch, in_dim)
#         y_eval : 3D torch.tensor
#             shape (batch, in_dim, out_dim)
#         grid : 2D torch.tensor
#             shape (in_dim, grid+2*k)
#         k : int
#             spline order
#         lamb : float
#             regularized least square lambda
            
#     Returns:
#     --------
#         coef : 3D torch.tensor
#             shape (in_dim, out_dim, G+k)
#     '''
#     #print('haha', x_eval.shape, y_eval.shape, grid.shape)
#     batch = x_eval.shape[0]
#     in_dim = x_eval.shape[1]
#     out_dim = y_eval.shape[2]
#     n_coef = grid.shape[1] - k - 1
#     mat = B_batch(x_eval, grid, k)
#     mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
#     #print('mat', mat.shape)
#     y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
#     #print('y_eval', y_eval.shape)
#     device = mat.device
    
#     #coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
#     try:
#         coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
#     except:
#         print('lstsq failed')
    
#     # manual psuedo-inverse
#     '''lamb=1e-8
#     XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
#     Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
#     n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
#     identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
#     A = XtX + lamb * identity
#     B = Xty
#     coef = (A.pinverse() @ B)[:,:,:,0]'''
    
#     return coef
import torch

# 1. Basis Function Evaluation
def gaussian_basis(x, centers, variances):
    """
    Evaluate Gaussian basis functions for inputs x.

    Args:
    -----
    x : torch.tensor
        Input data, shape (batch, in_dim).
    centers : torch.tensor
        Gaussian centers, shape (num_basis, in_dim).
    variances : torch.tensor
        Variances of Gaussian functions, shape (num_basis, in_dim).

    Returns:
    --------
    basis_values : torch.tensor
        Evaluated Gaussian basis values, shape (batch, num_basis).
    """
    # Reshape for broadcasting
    x = x.unsqueeze(1)  # Shape: (batch, 1, in_dim)
    centers = centers.unsqueeze(0)  # Shape: (1, num_basis, in_dim)
    variances = variances.unsqueeze(0)  # Shape: (1, num_basis, in_dim)

    # Gaussian kernel: exp(- ||x - center||^2 / (2 * variance^2))
    diff = x - centers
    squared_dist = torch.sum(diff**2 / variances, dim=2)
    basis_values = torch.exp(-0.5 * squared_dist)

    return basis_values


# 2. Coefficients to Function Evaluation
def coef2gaussian(x_eval, centers, variances, coef):
    """
    Convert Gaussian basis coefficients to function values.

    Args:
    -----
    x_eval : torch.tensor
        Input points, shape (batch, in_dim).
    centers : torch.tensor
        Gaussian centers, shape (num_basis, in_dim).
    variances : torch.tensor
        Gaussian variances, shape (num_basis, in_dim).
    coef : torch.tensor
        Coefficients for Gaussian basis, shape (num_basis, out_dim).

    Returns:
    --------
    y_eval : torch.tensor
        Evaluated function values, shape (batch, out_dim).
    """
    # Evaluate Gaussian basis functions
    basis_values = gaussian_basis(x_eval, centers, variances)
    
    # Linear combination of basis values with coefficients
    y_eval = torch.matmul(basis_values, coef)

    return y_eval


# 3. Function to Coefficients (Least Squares)
def gaussian2coef(x_eval, y_eval, centers, variances):
    """
    Compute Gaussian basis coefficients from function values using least squares.

    Args:
    -----
    x_eval : torch.tensor
        Input points, shape (batch, in_dim).
    y_eval : torch.tensor
        Function values, shape (batch, out_dim).
    centers : torch.tensor
        Gaussian centers, shape (num_basis, in_dim).
    variances : torch.tensor
        Gaussian variances, shape (num_basis, in_dim).

    Returns:
    --------
    coef : torch.tensor
        Coefficients for Gaussian basis, shape (num_basis, out_dim).
    """
    # Evaluate Gaussian basis functions
    basis_values = gaussian_basis(x_eval, centers, variances)
    
    # Solve least squares: basis_values @ coef ≈ y_eval
    coef = torch.linalg.lstsq(basis_values, y_eval).solution

    return coef


# 4. Utility for Center and Variance Initialization
def init_gaussian_basis(x_range, num_basis, in_dim, scale=1.0):
    """
    Initialize Gaussian centers and variances.

    Args:
    -----
    x_range : tuple
        Min and max values of input range (x_min, x_max).
    num_basis : int
        Number of Gaussian basis functions.
    in_dim : int
        Input dimensionality.
    scale : float
        Scaling factor for variances.

    Returns:
    --------
    centers : torch.tensor
        Gaussian centers, shape (num_basis, in_dim).
    variances : torch.tensor
        Gaussian variances, shape (num_basis, in_dim).
    """
    centers = torch.linspace(x_range[0], x_range[1], num_basis).repeat(in_dim, 1).T
    variances = torch.ones(num_basis, in_dim) * scale

    return centers, variances

# def extend_grid(grid, k_extend=0):
#     '''
#     extend grid
#     '''
#     h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

#     for i in range(k_extend):
#         grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
#         grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

#     return grid
