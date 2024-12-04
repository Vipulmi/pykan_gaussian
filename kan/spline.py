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

def Gaussian_batch(x, grid, k = None, device='cpu'):
    """
    Evaluate Gaussian basis functions using the provided grid.
    
    Args:
    -----
        x : 2D torch.tensor
            Inputs, shape (batch, in_dim)
        grid : 2D torch.tensor
            Gaussian centers, shape (in_dim, num_centers)
        width : float
            Standard deviation (same for all Gaussians)
        device : str
            Device ('cpu' or 'cuda')
            
    Returns:
    --------
        Gaussian values : 3D torch.tensor
            Shape (batch, in_dim, num_centers)
    """
    # Derive centers and width from grid
    centers = grid
    width = grid[1] - grid[0]  # Assuming uniform spacing for width

    # Ensure proper tensor dimensions for broadcasting
    x = x.unsqueeze(2)  # Add dimension for broadcasting
    centers = centers.unsqueeze(0)  # Add batch dimension for broadcasting
    
    # Compute the Gaussian function
    gaussians = torch.exp(-((x - centers) ** 2) / (2 * width ** 2))
    
    return gaussians


def coef2curve(x_eval, grid, coef, k = None, device="cpu"):
    """
    Calculate the curve based on coefficients and a Gaussian basis, where centers and width are derived from the grid.
    
    Parameters:
        x_eval: Tensor, input evaluation points.
        grid: Tensor, points defining the Gaussian grid.
        coef: Tensor, coefficients for the basis functions.
        k: int, controls Gaussian properties (e.g., sharpness).
        device: str, computation device ("cpu" or "cuda").
    
    Returns:
        Tensor representing the evaluated curve.
    """
    # Derive centers and width from the grid
    centers = grid
    width = grid[1] - grid[0]  # Assuming uniform spacing
    
    # Perform Gaussian basis evaluation
    diff = x_eval[:, None, :] - centers[None, :, None]  # Shape: (batch, num_centers, input_dim)
    gaussians = torch.exp(-k * (diff ** 2) / (2 * width ** 2))  # Shape: (batch, num_centers, input_dim)
    
    # Combine with coefficients
    result = torch.sum(gaussians * coef[None, :, :], dim=1)  # Shape: (batch, output_dim)
    
    return result


def curve2coef(x_eval, y_eval, grid, k = None):
    """
    Convert Gaussian curves to Gaussian coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        grid : 1D torch.tensor
            shape (num_centers,)
        k : float
            Controls Gaussian properties (e.g., sharpness).
        
    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, num_centers)
    """
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    num_centers = grid.shape[0]

    # Derive centers and width from the grid
    centers = grid
    width = grid[1] - grid[0]  # Assuming uniform spacing

    # Compute Gaussian basis matrix
    mat = Gaussian_batch(x_eval, centers, width)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, num_centers)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)

    try:
        # Solve least squares to find coefficients
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    except Exception as e:
        print(f"LSTSQ failed: {e}")
        coef = None

    return coef


def extend_grid(grid, k_extend=0):
    """
    Extend grid for Gaussian centers.
    
    Args:
    -----
        grid : 2D torch.tensor
            shape (in_dim, num_points)
        k_extend : int
            Number of points to extend on each side.
        
    Returns:
    --------
        extended_grid : 2D torch.tensor
            Extended grid.
    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid


# def extend_grid(grid, k_extend=0):
#     '''
#     extend grid
#     '''
#     h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

#     for i in range(k_extend):
#         grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
#         grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

#     return grid
