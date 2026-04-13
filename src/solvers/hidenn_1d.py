import torch
import torch.nn as nn
import numpy as np

from src.nn_modules.shape_1d import (
    ShapeNet1D_L2_Global, ShapeNet1D_L2_Parent, DerivativeNet1D_L2_Parent,
    ShapeNet1D_L3_Global, ShapeNet1D_L3_Parent, DerivativeNet1D_L3_Parent
)

class HiDeNNSolver1D(nn.Module):
    """
    通用 1D HiDeNN 求解器 (增強版：數值穩定性修復)
    """
    def __init__(self, n_nodes, initial_coords_np, E, A, body_force_fn,
                 element_type='L2', 
                 integration_method='Global', 
                 gauss_order=5, 
                 global_points=2000):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.E = E
        self.A = A
        self.body_force_fn = body_force_fn
        self.element_type = element_type
        self.integration = integration_method
        self.initial_coords_np = initial_coords_np
        # 使用 float64 以獲得更好的穩定性 (特別是 r-adaptivity)
        self.dtype = torch.float64 
        
        print(f"--- HiDeNN 1D Solver: {element_type} + {integration_method} (Nodes: {n_nodes}) ---")

        # --- 2. 建立元素連接 (Connectivity) ---
        if element_type == 'L2':
            self.connectivity = self._create_l2_connectivity(n_nodes)
            self.nodes_per_elem = 2
        elif element_type == 'L3':
            self.connectivity = self._create_l3_connectivity(n_nodes)
            self.nodes_per_elem = 3
        else:
            raise ValueError("element_type 必須是 'L2' 或 'L3'")
            
        self.n_elements = self.connectivity.shape[0]

        # --- 3. Parameters (初始化為 float64) ---
        self.internal_displacements = nn.Parameter(torch.zeros(n_nodes - 2, dtype=self.dtype))
        # [重要] 這裡使用 clone().detach() 確保斷開梯度並複製數據
        self.internal_coordinates = nn.Parameter(
            torch.tensor(initial_coords_np[1:-1], dtype=self.dtype).clone().detach()
        )
        
        # --- 4. Boundary Buffers ---
        self.register_buffer('u_boundary', torch.tensor([0.0, 0.0], dtype=self.dtype))
        self.register_buffer('x_boundary', torch.tensor([initial_coords_np[0], initial_coords_np[-1]], dtype=self.dtype))
        
        # --- 5. Networks & Quadrature ---
        if self.integration == 'Global':
            # 全域積分 (Trapezoidal)
            bar_length = initial_coords_np[-1]
            trapz_coords = torch.linspace(0, bar_length, global_points, dtype=self.dtype).view(-1, 1)
            trapz_coords.requires_grad = True
            self.register_buffer('TrialCoordinates', trapz_coords)
            
            if element_type == 'L2':
                self.global_shape_net = ShapeNet1D_L2_Global(self.n_elements)
            else:
                self.global_shape_net = ShapeNet1D_L3_Global(self.n_nodes, self.connectivity)
            
        elif self.integration == 'Gauss':
            # 高斯積分
            xi_g, w_g = np.polynomial.legendre.leggauss(gauss_order)
            self.register_buffer('gauss_points', torch.tensor(xi_g, dtype=self.dtype).view(-1, 1))
            self.register_buffer('gauss_weights', torch.tensor(w_g, dtype=self.dtype).view(-1, 1))
            
            if element_type == 'L2':
                self.parent_shape_net = ShapeNet1D_L2_Parent()
                self.parent_deriv_net = DerivativeNet1D_L2_Parent()
                self.global_shape_net = ShapeNet1D_L2_Global(self.n_elements) # 僅用於視覺化
            else:
                self.parent_shape_net = ShapeNet1D_L3_Parent()
                self.parent_deriv_net = DerivativeNet1D_L3_Parent()
                self.global_shape_net = ShapeNet1D_L3_Global(self.n_nodes, self.connectivity) # 僅用於視覺化
        
        # 將整個模型轉為 float64 (Double) 以確保穩定
        self.to(torch.float64)

    def _create_l2_connectivity(self, n_nodes):
        conn = [[i, i + 1] for i in range(n_nodes - 1)]
        return torch.tensor(conn, dtype=torch.long)

    def _create_l3_connectivity(self, n_nodes):
        if (n_nodes - 1) % 2 != 0:
            raise ValueError(f"L3 元素要求 n_nodes 為奇數，收到 {n_nodes}")
        n_elem = (n_nodes - 1) // 2
        conn = [[i*2, i*2 + 1, i*2 + 2] for i in range(n_elem)]
        return torch.tensor(conn, dtype=torch.long)

    def get_full_vectors(self):
        # 組合邊界與內部參數
        full_displacements = torch.cat([self.u_boundary[0:1], self.internal_displacements, self.u_boundary[1:2]])
        full_coordinates = torch.cat([self.x_boundary[0:1], self.internal_coordinates, self.x_boundary[1:2]])
        return full_displacements, full_coordinates

    def get_displacement(self, x_eval):
        # 用於視覺化 (使用全域網路)
        d_full, x_full = self.get_full_vectors()
        x_eval = x_eval.to(device=x_full.device, dtype=self.dtype)
        N_values = self.global_shape_net(x_eval, x_full)
        u_pred = N_values @ d_full
        return u_pred
        
    def forward(self):
        if self.integration == 'Global':
            return self.forward_global()
        else:
            return self.forward_gauss()

    def forward_global(self):
        # 梯形積分方法
        x_eval = self.TrialCoordinates
        u_predicted = self.get_displacement(x_eval).view(-1, 1)

        du_dx = torch.autograd.grad(
            outputs=u_predicted, 
            inputs=x_eval, 
            grad_outputs=torch.ones_like(u_predicted), 
            create_graph=True
        )[0]
        
        b = self.body_force_fn(x_eval)
        integrand = 0.5 * self.A * self.E * du_dx**2 - u_predicted * b
        
        potential_energy = torch.trapezoid(integrand.squeeze(), x_eval.squeeze())
        return potential_energy

    def forward_gauss(self):
        d_full, x_full = self.get_full_vectors()
        
        # 1. Shape Functions & Derivatives (in Parent domain)
        N_k = self.parent_shape_net(self.gauss_points)
        dN_dxi_k = self.parent_deriv_net(self.gauss_points)

        # 2. Element Data
        X_e = x_full[self.connectivity].unsqueeze(-1) # [n_el, nodes, 1]
        d_e = d_full[self.connectivity].unsqueeze(-1) # [n_el, nodes, 1]
        
        # 3. Jacobian Mapping
        # J_k = dN/dxi * X_e
        # [n_gauss, nodes] @ [nodes, 1] -> [n_gauss, 1] per element
        J_k = torch.einsum('gk,ekd->egd', dN_dxi_k, X_e)
        
        # [關鍵修正 1]: 防止 J 為 0 或負數導致 NaN
        # 我們取絕對值進行積分 (體積元素)
        detJ = J_k # 1D Jacobian
        abs_detJ = torch.abs(detJ)
        
        # [關鍵修正 2]: 計算導數時的分母保護
        # dN/dx = (dN/dxi) / J
        # 如果 J 接近 0，導數會爆炸 -> NaN
        # 我們保持 J 的符號，但加入一個極小的 epsilon (1e-12) 防止除以零
        J_denom = J_k + 1e-12 * torch.sign(J_k)
        # 如果 sign 是 0，再加一個常數
        J_denom = torch.where(torch.abs(J_denom) < 1e-12, torch.tensor(1e-12, dtype=self.dtype, device=J_denom.device), J_denom)
        
        x_k = torch.einsum('gk,ekd->egd', N_k, X_e)
        u_k = torch.einsum('gk,ekd->egd', N_k, d_e)
        
        dN_dx_k = dN_dxi_k.unsqueeze(0) / J_denom
        du_dx_k = torch.einsum('egk,ekd->egd', dN_dx_k, d_e)
        
        # 4. Integration
        b_k = self.body_force_fn(x_k)
        integrand_k = 0.5 * self.A * self.E * du_dx_k**2 - u_k * b_k
        
        # 使用 abs_detJ 確保積分權重為正
        w_g = self.gauss_weights.unsqueeze(0)
        weighted_integrand = w_g * integrand_k * abs_detJ
        
        total_potential_energy = torch.sum(weighted_integrand)
        return total_potential_energy

    def freeze_mesh(self):
        self.internal_coordinates.requires_grad = False

    def unfreeze_mesh(self):
        self.internal_coordinates.requires_grad = True