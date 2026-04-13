# 1D Neural Network Finite Element Method (HiDeNN)

本專案實作基於 Hierarchical Deep Neural Networks (HiDeNN) 架構的一維有限元素法 (FEM) 求解器。藉由將 FEM 形狀函數表示為神經網路，並利用 PyTorch 進行總位能的自動微分與優化，實現高梯度的固體力學問題求解。

本求解器不僅支援傳統的固定網格 (Fixed Mesh) 分析，亦支援 **r-adaptivity**，允許節點座標作為可訓練參數，在優化過程中自動調整網格密度以降低局部誤差。

## 核心技術特點

1. **神經網路形狀函數**: 支援 2 節點 (L2) 與 3 節點 (L3) 一維元素。
2. **數值積分方案**:
   * **Gauss Quadrature**: 於父座標 (Parent domain) 計算後映射，支援精確 Jacobian 轉換。
   * **Global Integration**: 使用梯形法則 (Trapezoidal rule) 進行全域數值積分。
3. **無矩陣求解 (Matrix-free)**: 不需組裝總體剛度矩陣 $[K]$，直接最小化系統總位能 $\Pi(u) = \int (\frac{1}{2} E A (\frac{du}{dx})^2 - ub) dx$。
4. **r-Adaptivity (網格自適應)**: 可同時優化節點位移 (Displacements) 與節點座標 (Coordinates)。

