import torch
import torch.nn as nn

class ShapeFunctionNetwork1D(nn.Module):
    """
    1D HiDeNN 形狀函數神經網路。
    動態生成 w=f(x*), b=f(x*)。
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

            # Layer 1
            w1 = -1.0
            b1 = xb
            h = xb - xa
            
            # Layer 2
            w2 = -1.0 / (h + 1e-9)
            b2 = 1.0
            
            # Layer 3
            w3_n0 = -1.0
            b3_n0 = 1.0
            w3_n1 = 1.0
            b3_n1 = 0.0

            # Forward
            layer1_out = torch.relu(w1 * x_eval + b1)
            layer2_out = torch.relu(w2 * layer1_out + b2)
            
            n0_i = w3_n0 * layer2_out + b3_n0
            n1_i = w3_n1 * layer2_out + b3_n1

            all_N0_tilde.append(n0_i)
            all_N1_tilde.append(n1_i)

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