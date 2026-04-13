import torch
import numpy as np
import sys
import os
import time

# ------------------------------------------------------------------
# 1. 路徑設定 (確保能 import src)
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
if project_root not in sys.path:
    sys.path.append(project_root)

# ------------------------------------------------------------------
# 2. 模組導入
# ------------------------------------------------------------------
from src.benchmarks.bar_hard_case import (
    body_force_b, 
    analytical_u, 
    analytical_du_dx, 
    analytical_d2u_dx2
)
from src.solvers.hidenn_1d import HiDeNNSolver1D
from src.utils.visualization_1d import analyze_and_plot_1d

# ==================================================================
# 3. 實驗設定 (STUDY CONFIGURATION)
# ==================================================================

# *** 在這裡切換您想跑的實驗 ***
STUDY_CONFIG = {
    # 1D(1) L2 Global: 'L2', 'Global'
    # 1D(2) L3 Global: 'L3', 'Global'
    # 1D(3) L2 Gauss:  'L2', 'Gauss'
    # 1D(4) L3 Gauss:  'L3', 'Gauss'
    "element_type": 'L2',
    "integration_method": 'Global',
    
    # --- 訓練控制 ---
    "optimizer": 'Adam',     # 'Adam' (推薦用於 r-adaptivity) 或 'LBFGS'
    "freeze_mesh": True,    # True: 固定網格, False: r-adaptivity
    "num_epochs": 500,      # 若開啟 r-adaptivity，建議增加 epoch
    "learning_rate": 1e-3,   # 主要學習率 (位移 u)
    "lr_coord_scale": 0.1,  # 座標 x 的學習率縮放比例 (x_lr = lr * scale)
    
    # --- 物理參數 ---
    "E": 175.0,
    "A": 1.0,
    "bar_length": 10.0,
    
    # --- 網格參數 ---
    "n_nodes": 23, 
    
    # --- 求解器精度 ---
    "gauss_order": 5,           
    "global_points": 2000,      
    "plot_resolution": 400      
}
# ==================================================================


def main():
    
    # --- 4. 準備網格與參數 ---
    cfg = STUDY_CONFIG
    n_nodes = cfg['n_nodes']
    
    # 檢查 L3 節點是否為奇數
    if cfg['element_type'] == 'L3' and (n_nodes - 1) % 2 != 0:
        print(f"錯誤: L3 元素要求 n_nodes 為奇數 (例如 3, 5, ...)，但收到 {n_nodes}")
        return

    # 初始網格 (均勻分佈)
    initial_coords_np = np.linspace(0, cfg['bar_length'], n_nodes)

    # --- 5. 初始化求解器 ---
    # 這裡會呼叫 src/solvers/hidenn_1d.py 中的 HiDeNNSolver1D
    solver = HiDeNNSolver1D(
        n_nodes=n_nodes, 
        initial_coords_np=initial_coords_np, 
        E=cfg['E'], 
        A=cfg['A'], 
        body_force_fn=body_force_b,  
        element_type=cfg['element_type'],
        integration_method=cfg['integration_method'],
        gauss_order=cfg['gauss_order'],
        global_points=cfg['global_points']
    )

    # --- 6. 設定優化器 (關鍵修正: 參數分組) ---
    if cfg['freeze_mesh']:
        solver.freeze_mesh()
        # 凍結模式：只有一組參數 (Displacements)
        params = solver.parameters()
        
        if cfg['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(params, lr=cfg['learning_rate'])
        else:
            optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
            
    else:
        solver.unfreeze_mesh()
        # 解凍模式 (r-adaptivity)：
        # [關鍵]: 必須把座標 (coordinates) 的學習率設得很小，防止網格翻轉
        lr_u = cfg['learning_rate']
        lr_x = lr_u * cfg['lr_coord_scale'] # 例如 1e-3 * 0.01 = 1e-5

        print(f"--- r-adaptivity 參數分組: LR_u={lr_u:.1e}, LR_x={lr_x:.1e} ---")

        # 建立參數分組
        # solver.internal_displacements -> 使用較大的 lr_u
        # solver.internal_coordinates   -> 使用極小的 lr_x
        param_groups = [
            {'params': [solver.internal_displacements], 'lr': lr_u},
            {'params': [solver.internal_coordinates],   'lr': lr_x}
        ]

        if cfg['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(param_groups)
        else:
            # LBFGS 對多組 LR 支援較差，這裡統一使用較小的 LR 或者建議使用 Adam
            print("警告: LBFGS 在 r-adaptivity 模式下建議使用統一的小 LR，或改用 Adam。")
            optimizer = torch.optim.LBFGS(solver.parameters(), lr=0.1, max_iter=20, line_search_fn="strong_wolfe")

    # --- 7. 訓練迴圈 ---
    print(f"\n--- 開始訓練 (Optimizer: {cfg['optimizer']}, Method: {cfg['integration_method']}) ---")
    start_time = time.time()
    
    def closure():
        optimizer.zero_grad()
        loss = solver() 
        if torch.isnan(loss):
             # 如果還是出現 NaN，這裡會捕捉到，避免程式崩潰但提示錯誤
             raise ValueError("Loss 變為 NaN (網格可能翻轉)，請嘗試降低 learning_rate 或 lr_coord_scale")
        loss.backward()
        return loss

    for epoch in range(cfg['num_epochs']):
        try:
            if isinstance(optimizer, torch.optim.LBFGS):
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.8f}")
        
        except ValueError as e:
            print(f"訓練中止於 Epoch {epoch}: {e}")
            break
        except Exception as e:
            print(f"發生未預期的錯誤: {e}")
            break

    end_time = time.time()
    print(f"\n--- 訓練完成 (耗時: {end_time - start_time:.2f} 秒) ---")

    # --- 8. 後處理與視覺化 ---
    analytical_funcs = (analytical_u, analytical_du_dx, analytical_d2u_dx2)

    analyze_and_plot_1d(
        solver=solver,
        analytical_funcs=analytical_funcs,
        bar_length=cfg['bar_length'],
        plot_resolution=cfg['plot_resolution']
    )

if __name__ == "__main__":
    main()