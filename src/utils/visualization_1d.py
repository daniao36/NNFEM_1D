import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_1d_field_derivatives(solver, x_plot_torch):
    """
    使用 Autograd 計算 u, du/dx, d2u/dx2
    
    Args:
        solver: 已訓練的 HiDeNNSolver1D 實例
        x_plot_torch (Tensor): [n_points, 1] - 繪圖用的 x 座標 (需要 requires_grad=True)

    Returns:
        (np.array, np.array, np.array): (u, du/dx, d2u/dx2) 的 NumPy 陣列
    """
    print(" ... 正在計算 u, u', u'' (使用 autograd)...")
    
    # 確保 x_plot 需要梯度
    if not x_plot_torch.requires_grad:
        x_plot_torch.requires_grad = True
    
    # 1. 計算 u
    # solver.get_displacement 返回 shape [n_points]
    u_pred = solver.get_displacement(x_plot_torch)
    
    # 2. 計算 du/dx
    du_dx = torch.autograd.grad(
        outputs=u_pred.sum(),
        inputs=x_plot_torch,
        create_graph=True  # 保持圖結構以計算二階
    )[0]
    
    # 3. 計算 d2u/dx2
    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx.sum(),
        inputs=x_plot_torch,
        create_graph=False
    )[0]
    
    # 返回 NumPy 陣列
    return (
        u_pred.detach().cpu().numpy().flatten(),
        du_dx.detach().cpu().numpy().flatten(),
        d2u_dx2.detach().cpu().numpy().flatten()
    )

def compute_l2_error(pred_np, exact_np):
    """計算 L2 相對誤差"""
    # 確保分母不為零
    norm_exact = np.linalg.norm(exact_np)
    if norm_exact < 1e-10:
        return np.linalg.norm(pred_np - exact_np)
        
    norm_diff = np.linalg.norm(pred_np - exact_np)
    return norm_diff / norm_exact

def plot_1d_result(
    x_np, 
    pred_np, 
    exact_np, 
    nodes_x, 
    nodes_y, 
    initial_nodes_x=None, 
    title="Result", 
    ylabel="Value",
    save_path=None
):
    """
    通用的 1D 繪圖函數
    """
    plt.figure(figsize=(10, 6))
    
    # 畫解析解 (紅色虛線)
    plt.plot(x_np, exact_np, 'r--', lw=2, label='Analytical Solution')
    # 畫 HiDeNN 預測解 (綠色實線)
    plt.plot(x_np, pred_np, 'g-', lw=3, alpha=0.8, label='HiDeNN Solution')
    
    # 畫最終節點位置 (藍色)
    plt.scatter(nodes_x, nodes_y, c='blue', s=50, zorder=5, label='Final Nodes')
    
    # 如果有傳入初始節點位置 (r-adaptivity 開啟時)，畫出來 (粉色)
    if initial_nodes_x is not None:
        y_initial = np.zeros_like(initial_nodes_x) if nodes_y.ndim == 1 and np.allclose(nodes_y, 0) else nodes_y
        plt.scatter(initial_nodes_x, y_initial, 
                    c='pink', s=50, zorder=4, alpha=0.8, label='Initial Nodes')

    plt.title(title)
    plt.xlabel("Coordinate x")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖表已儲存至: {save_path}")
    
    plt.show()

def analyze_and_plot_1d(solver, analytical_funcs, bar_length, plot_resolution=400):
    """
    一鍵分析函數：計算誤差並畫出 u, du/dx, d2u/dx2
    
    Args:
        solver: 已訓練的 HiDeNNSolver1D 實例
        analytical_funcs (tuple): 包含 (func_u, func_du, func_d2u) 的元組
        bar_length (float): 桿長
        plot_resolution (int): 繪圖點的密度
    """
    print("\n--- 正在進行後處理與視覺化 ---")
    
    # 1. 準備繪圖網格
    device = solver.internal_displacements.device
    x_plot_torch = torch.linspace(0, bar_length, plot_resolution, dtype=solver.dtype, device=device).view(-1, 1)
    x_np = x_plot_torch.cpu().numpy().flatten()
    
    # 2. 計算 HiDeNN 預測值 (含導數)
    u_h, du_h, d2u_h = compute_1d_field_derivatives(solver, x_plot_torch)
    
    # 3. 計算解析解
    E, A = solver.E, solver.A
    ana_u, ana_du, ana_d2u = analytical_funcs
    
    u_true = ana_u(x_np, E, A)
    du_true = ana_du(x_np, E, A)
    d2u_true = ana_d2u(x_np, E, A)
    
    # 4. 計算誤差
    err_u = compute_l2_error(u_h, u_true)
    err_du = compute_l2_error(du_h, du_true)
    err_d2u = compute_l2_error(d2u_h, d2u_true)
    print(f"相對 L2 誤差 - 位移 u:     {err_u:.4e}")
    print(f"相對 L2 誤差 - 應變 du/dx: {err_du:.4e}")
    print(f"相對 L2 誤差 - 應變梯度 d2u/dx2: {err_d2u:.4e}")

    # 5. 獲取節點資訊 (用於散點圖)
    with torch.no_grad():
        final_d, final_x = solver.get_full_vectors()
    final_d_np = final_d.cpu().numpy()
    final_x_np = final_x.cpu().numpy()
    
    # 判斷是否需要畫初始網格
    init_x_np = None
    if solver.internal_coordinates.requires_grad:
        init_x_np = solver.initial_coords_np

    # --- 6. 繪圖 ---
    
    # (A) 位移 u
    plot_1d_result(x_np, u_h, u_true, final_x_np, final_d_np, init_x_np,
                   title=f"Displacement u(x) ({solver.element_type}, {solver.integration})", 
                   ylabel="Displacement u")
    
    # (B) 應變 du/dx
    plot_1d_result(x_np, du_h, du_true, final_x_np, np.zeros_like(final_x_np), init_x_np,
                   title=f"Strain du/dx ({solver.element_type}, {solver.integration})", 
                   ylabel="Strain ε")
                   
    # (C) 應變梯度 d2u/dx2
    plot_1d_result(x_np, d2u_h, d2u_true, final_x_np, np.zeros_like(final_x_np), init_x_np,
                   title=f"Strain Gradient d²u/dx² ({solver.element_type}, {solver.integration})", 
                   ylabel="Strain Gradient")