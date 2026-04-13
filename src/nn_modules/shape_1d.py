import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Section 1: 基礎構建模塊 (Building Blocks)
# ------------------------------------------------------------------

class IdentityActivation(nn.Module):
    """ 實現 A1(x) = x 激活函數 (恆等映射) """
    def forward(self, x):
        return x

class QuadraticActivation(nn.Module):
    """ 實現 A2(x) = x^2 激活函數 """
    def forward(self, x):
        return x * x

class MultiplicationBlock(nn.Module):
    """ 
    實現 M 節點 (乘法節點) M(F1, F2) = F1 * F2
    使用 A2(x) = x^2 (Quarter-Square Trick)
    """
    def __init__(self):
        super().__init__()
        self.a2_quadratic = QuadraticActivation()

    def forward(self, f1, f2):
        """ 
        實現 F1*F2 = 0.5 * [ (F1+F2)^2 - F1^2 - F2^2 ]
        """
        o1 = self.a2_quadratic(f1)
        o2 = self.a2_quadratic(f1 + f2)
        o3 = self.a2_quadratic(f2)
        
        M_out = -0.5 * o1 + 0.5 * o2 - 0.5 * o3
        return M_out

# ------------------------------------------------------------------
# Section 2: 1D 二節點 (L2) 網路
# ------------------------------------------------------------------

class ShapeNet1D_L2_Global(nn.Module):
    """
    1D (1) 二節點全域 (Global Coordinate) 形狀函數
    (來自 nn_1D.py)
    
    直接在真實座標 'x' 上計算 hat functions。
    輸入:
        x_eval (Tensor): [n_points, 1] - 查詢點
        x_full (Tensor): [n_nodes] - 節點座標
    輸出:
        N_values (Tensor): [n_points, n_nodes] - 全域形狀函數
    """
    def __init__(self, n_elements):
        super().__init__()
        self.n_elements = n_elements

    def forward(self, x_eval, x_full):
        all_N0_tilde = []
        all_N1_tilde = []

        for i in range(self.n_elements):
            xa = x_full[i]
            xb = x_full[i+1]
            
            # --- 動態計算權重 (w) 與偏置 (b) ---
            h = xb-xa
            
            # Layer 1
            w1 = -1.0
            b1 = xb
            
            # Layer 2
            w2 = -1.0 / (h + 1e-9)
            b2 = 1.0
            
            # Layer 3 (N0: 下降斜坡 1->0)
            w3_n0 = -1.0
            b3_n0 = 1.0 
            
            # Layer 3 (N1: 上升斜坡 0->1)
            w3_n1 = 1.0
            b3_n1 = 0.0

            # --- 前向傳播 ---
            layer1_out = torch.relu(w1 * x_eval + b1)
            layer2_out = torch.relu(w2 * layer1_out + b2)
            
            n0_i = w3_n0 * layer2_out + b3_n0
            n1_i = w3_n1 * layer2_out + b3_n1

            all_N0_tilde.append(n0_i)
            all_N1_tilde.append(n1_i)

        # --- 組裝層 ---
        N0_tilde = torch.cat(all_N0_tilde, dim=1)
        N1_tilde = torch.cat(all_N1_tilde, dim=1)
        
        N0 = N0_tilde[:, 0:1]
        N_last = N1_tilde[:, -1:]
        
        n_nodes = len(x_full)
        if n_nodes > 2:
            N_internal = N0_tilde[:, 1:] + N1_tilde[:, :-1] - 1.0
            N_values = torch.cat([N0, N_internal, N_last], dim=1)
        else:
            N_values = torch.cat([N0, N_last], dim=1)

        return N_values

class ShapeNet1D_L2_Parent(nn.Module):
    """
    1D (3) 二節點父座標 (Parent Coordinate) 形狀函數
    (來自 nn_1D_guass.py)
    
    在父座標 'xi' (範圍 -1 到 1) 上計算。
    N1(xi) = (1-xi)/2
    N2(xi) = (1+xi)/2
    
    輸入:
        xi (Tensor): [n_points, 1] - 父座標
    輸出:
        N_values (Tensor): [n_points, 2] - 局部形狀函數
    """
    def __init__(self):
        super().__init__()
        self.A1 = IdentityActivation()
        # 權重/偏置 (來自論文 Fig. 5b)
        self.w11_12 = -1.0; self.b1 = 1.0
        self.w12_12 = 1.0; self.b2 = 1.0
        self.w11_23 = 0.5
        self.w22_23 = 0.5

    def forward(self, xi):
        a1_out = self.A1(self.w11_12 * xi + self.b1) # 1 - xi
        N1 = self.w11_23 * a1_out                  # 0.5 * (1 - xi)
        a2_out = self.A1(self.w12_12 * xi + self.b2) # 1 + xi
        N2 = self.w22_23 * a2_out                  # 0.5 * (1 + xi)
        return torch.cat([N1, N2], dim=1)

class DerivativeNet1D_L2_Parent(nn.Module):
    """
    1D (3) 二節點父座標 (Parent Coordinate) *導數*
    
    dN1/dxi = -0.5
    dN2/dxi = +0.5
    
    輸入:
        xi (Tensor): [n_points, 1] - (僅用於確定批次大小)
    輸出:
        DN_values (Tensor): [n_points, 2] - 局部導數
    """
    def __init__(self):
        super().__init__()
        # 導數是常數
        self.register_buffer('DN_DXI', torch.tensor([[-0.5, 0.5]]))

    def forward(self, xi):
        # 擴展以匹配輸入 xi 的批次大小
        return self.DN_DXI.expand(xi.shape[0], -1)

# ------------------------------------------------------------------
# Section 3: 1D 三節點 (L3) 網路
# ------------------------------------------------------------------

class ShapeNet1D_L3_Global(nn.Module):
    """
    1D (2) 三節點全域 (Global Coordinate) 形狀函數
    (來自 nn_1D_2P.py)
    
    使用 Lagrange 多項式在真實座標 'x' 上計算。
    *需要 MultiplicationBlock*
    
    輸入:
        x_eval (Tensor): [n_points, 1] - 查詢點
        x_full (Tensor): [n_nodes] - 節點座標
        connectivity (Tensor): [n_elements, 3] - 元素連接
    輸出:
        N_values (Tensor): [n_points, n_nodes] - 全域形狀函數
    """
    def __init__(self, n_nodes, connectivity):
        super().__init__()
        self.n_nodes = n_nodes
        self.connectivity = connectivity
        self.n_quad_elements = connectivity.shape[0]
        
        # 激活函數
        self.a1_relu = nn.ReLU()
        self.M = MultiplicationBlock()
        self.eps = 1e-9

    def _linear_block(self, x_eval, xA, xB, yA, yB):
        """ (b) Linear building block L(x; xA, xB, yA, yB) """
        h = xB - xA + self.eps
        w1 = -1.0
        b1 = xB
        l1_out = self.a1_relu(w1 * x_eval + b1)
        w2 = -1.0 / h
        b2 = 1.0
        l2_out = self.a1_relu(w2 * l1_out + b2)
        w3 = yB - yA
        b3 = yA
        L_out = w3 * l2_out + b3
        return L_out

    def forward(self, x_eval, x_full):
        all_NI_tilde = []
        all_NM_tilde = []
        all_NJ_tilde = []

        # --- 1. 元素迴圈 (Element Loop) ---
        for i in range(self.n_quad_elements):
            element_nodes = self.connectivity[i]
            idx_i, idx_m, idx_j = element_nodes[0], element_nodes[1], element_nodes[2]
            
            xi = x_full[idx_i]
            xm = x_full[idx_m]
            xj = x_full[idx_j]
            
            # --- 計算 N_I(x) ---
            den_i = (xi - xm) * (xi - xj) + self.eps
            L_A_i = self._linear_block(x_eval, xi, xj, (xi-xm), (xj-xm))
            L_B_i = self._linear_block(x_eval, xi, xj, (xi-xj), (xj-xj))
            Ni = self.M(L_A_i, L_B_i) / den_i

            # --- 計算 N_m(x) ---
            den_m = (xm - xi) * (xm - xj) + self.eps
            L_A_m = self._linear_block(x_eval, xi, xj, (xi-xi), (xj-xi))
            L_B_m = self._linear_block(x_eval, xi, xj, (xi-xj), (xj-xj))
            Nm = self.M(L_A_m, L_B_m) / den_m

            # --- 計算 N_J(x) ---
            den_j = (xj - xi) * (xj - xm) + self.eps
            L_A_j = self._linear_block(x_eval, xi, xj, (xi-xi), (xj-xi))
            L_B_j = self._linear_block(x_eval, xi, xj, (xi-xm), (xj-xm))
            Nj = self.M(L_A_j, L_B_j) / den_j

            # --- 遮罩 (Masking) ---
            mask = (x_eval >= xi) & (x_eval <= xj)
            
            all_NI_tilde.append(Ni * mask)
            all_NM_tilde.append(Nm * mask)
            all_NJ_tilde.append(Nj * mask)

        # --- 2. 組裝層 (Assembly Layer) ---
        NI_tilde = torch.cat(all_NI_tilde, dim=1)
        NM_tilde = torch.cat(all_NM_tilde, dim=1)
        NJ_tilde = torch.cat(all_NJ_tilde, dim=1)
        
        N_values = torch.zeros((x_eval.shape[0], self.n_nodes), 
                               dtype=x_eval.dtype, device=x_eval.device)

        for i in range(self.n_quad_elements):
            idx_i, idx_m, idx_j = self.connectivity[i]
            
            N_values[:, idx_i] += NI_tilde[:, i]
            N_values[:, idx_m] += NM_tilde[:, i]
            N_values[:, idx_j] += NJ_tilde[:, i]

        return N_values

class ShapeNet1D_L3_Parent(nn.Module):
    """
    1D (4) 三節點父座標 (Parent Coordinate) 形狀函數
    [*** 新增 ***]
    
    在父座標 'xi' (範圍 -1 到 1) 上計算，節點位於 (-1, 0, 1)。
    N1(xi) = 0.5 * xi * (xi - 1)
    N2(xi) = (1 - xi^2)
    N3(xi) = 0.5 * xi * (xi + 1)
    
    輸入:
        xi (Tensor): [n_points, 1] - 父座標
    輸出:
        N_values (Tensor): [n_points, 3] - 局部形狀函數
    """
    def __init__(self):
        super().__init__()
        self.M = MultiplicationBlock()
        # 為了清晰，定義一些常數
        self.register_buffer('one', torch.tensor(1.0))
        self.register_buffer('c_half', torch.tensor(0.5))
        self.register_buffer('neg_one', torch.tensor(-1.0))

    def forward(self, xi):
        xi_plus_1 = xi + self.one
        xi_minus_1 = xi + self.neg_one
        
        # N1 = 0.5 * xi * (xi - 1)
        N1 = self.M(self.c_half * xi, xi_minus_1)
        
        # N2 = (1 - xi) * (1 + xi)
        one_minus_xi = self.one - xi
        N2 = self.M(one_minus_xi, xi_plus_1)
        
        # N3 = 0.5 * xi * (xi + 1)
        N3 = self.M(self.c_half * xi, xi_plus_1)
        
        return torch.cat([N1, N2, N3], dim=1)

class DerivativeNet1D_L3_Parent(nn.Module):
    """
    1D (4) 三節點父座標 (Parent Coordinate) *導數*
    [*** 新增 ***]
    
    dN1/dxi = xi - 0.5
    dN2/dxi = -2*xi
    dN3/dxi = xi + 0.5
    
    輸入:
        xi (Tensor): [n_points, 1]
    輸出:
        DN_values (Tensor): [n_points, 3] - 局部導數
    """
    def __init__(self):
        super().__init__()

    def forward(self, xi):
        dN1 = xi - 0.5
        dN2 = -2.0 * xi
        dN3 = xi + 0.5
        return torch.cat([dN1, dN2, dN3], dim=1)