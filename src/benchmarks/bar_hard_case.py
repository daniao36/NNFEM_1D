import torch
import numpy as np

# ------------------------------------------------------------------
# 物理問題定義: 1D Bar (Hard Case with high gradients)
# ------------------------------------------------------------------

def body_force_b(x):
    """計算體力 b(x)"""
    pi = np.pi
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    term1 = -(4*pi**2 * (x - 2.5)**2 - 2*pi)
    term2 = torch.exp(pi * (x - 2.5)**2)
    term3 = -(8*pi**2 * (x - 7.5)**2 - 4*pi)
    term4 = torch.exp(pi * (x - 7.5)**2)
    return term1 / term2 + term3 / term4

def analytical_u(x, E, A):
    """計算位移的解析解 u(x)"""
    pi = np.pi
    AE = E * A
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        
    term1 = (1/AE) * (np.exp(-pi*(x-2.5)**2) - np.exp(-6.25*pi))
    term2 = (2/AE) * (np.exp(-pi*(x-7.5)**2) - np.exp(-56.25*pi))
    term3 = (np.exp(-6.25*pi) - np.exp(-56.25*pi)) / (10*AE) * x
    return term1 + term2 - term3

def analytical_du_dx(x, E, A):
    """計算應變的解析解 du/dx"""
    pi = np.pi
    AE = E * A
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    term1 = (2/AE) * (- pi * (x - 2.5)) * np.exp(-pi * (x - 2.5)**2)
    term2 = (4/AE) * (- pi * (x - 7.5)) * np.exp(-pi * (x - 7.5)**2)
    term3 = (np.exp(-6.25*pi) - np.exp(-56.25*pi)) / (10*AE)
    return term1 + term2 - term3

def analytical_d2u_dx2(x, E, A):
    """計算應變梯度的解析解 d2u/dx2"""
    pi = np.pi
    AE = E * A
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    term1 = (2/AE) * np.exp(-pi*(x-2.5)**2) * (2*pi**2*(x-2.5)**2 - pi)
    term2 = (4/AE) * np.exp(-pi*(x-7.5)**2) * (2*pi**2*(x-7.5)**2 - pi)
    return term1 + term2

def compute_pure_internal_energy(u_predicted, x, A, E):
    """計算純內能 (輔助函數)"""
    du_dx = torch.autograd.grad(
        outputs=u_predicted, 
        inputs=x, 
        grad_outputs=torch.ones_like(u_predicted), 
        create_graph=True
    )[0]
    
    W_e = 0.5 * A * E * du_dx*du_dx
    total_W = torch.sum(W_e)
    return W_e, total_W