# 1D Neural Network Finite Element Method (HiDeNN)

本專案實作了一種基於 **HiDeNN (Hierarchical Deep-learning Neural Networks)** 架構的一維有限元素分析工具。本方法將有限元素法（FEM）的形狀函數（Shape Functions）映射為神經網路的層級結構，並利用 PyTorch 的自動微分（Autograd）直接最小化系統總位能，取代傳統 FEM 的剛度矩陣組裝與求解流程。

## 1. 理論背景 (Theoretical Basis)

本程式碼的核心算法與架構開發嚴格遵循以下學術論文提出的理論：

> **Zhang, L., Cheng, L., Li, H., Gao, J., Yu, C., Domel, R., Yang, Y., Tang, S., & Liu, W. K. (2020).**
> *Hierarchical deep-learning neural networks: finite elements and beyond.*
> **Computational Mechanics**, 67(1), 207-230. [DOI: 10.1007/s00466-020-01928-9](https://doi.org/10.1007/s00466-020-01928-9)

### 核心特性：
* **形狀函數神經網路化**：將傳統 L2（線性）或 L3（二次）形狀函數建構為多層神經網路。例如，利用 ReLU 激活函數的組合特性來精確重現 Hat functions。
* **變分原理 (Variational Principle)**：求解過程不需組裝剛度矩陣 $[K]$，而是直接優化位能泛函 $\Pi(u) = \int (\frac{1}{2} E A (\frac{du}{dx})^2 - ub) dx$。
* **r-Adaptivity (網格自適應)**：支援將節點座標 $x$ 視為可訓練參數。透過優化器（如 Adam），網格會自動向解梯度較大（如應力集中）的區域聚集，以極小化數值誤差。

---

## 2. 專案特點

* **支援元素類型**：
    * **L2 (2-node)**：基於 Hat function 的線性網路。
    * **L3 (3-node)**：使用乘法塊（Multiplication Block）實作的二次 Lagrange 多項式網路。
* **數值積分方案**：
    * **Gauss Quadrature**：於父座標計算並映射，支援精確 Jacobian 轉換與體積積分。
    * **Global Integration**：使用梯形法則（Trapezoidal rule）進行全域點積分。
* **穩定性機制**：針對 r-adaptivity 加入了 Jacobian 保護與極小 epsilon 修正，防止網格翻轉導致梯度爆炸 (NaN)。

---

## 3. 安裝與快速上手

### 環境需求
請確保您的環境中已安裝以下 Python 套件（建議 Python 3.8 以上）：
```bash
pip install torch numpy matplotlib

使用方法
主要實驗控制位於 run_1d_study.py。您可以透過修改檔案內的 STUDY_CONFIG 字典來調整超參數與物理設定：

Python
# 實驗設定範例 (位於 run_1d_study.py)
STUDY_CONFIG = {
    "element_type": 'L2',        # 選擇 L2 或 L3 元素
    "integration_method": 'Gauss', # 選擇 Global 或 Gauss 積分
    "freeze_mesh": False,        # 設為 False 以啟動 r-adaptivity (動態網格)
    "optimizer": 'Adam',         # r-adaptivity 建議使用 Adam
    "learning_rate": 1e-3,       # 位移 u 的主學習率
    "lr_coord_scale": 0.1,       # 座標 x 的學習率縮放比例 (防止網格翻轉)
}
執行測試：

Bash
python run_1d_study.py
4. 專案結構
Plaintext
.
├── run_1d_study.py          # 實驗執行與參數設定主腳本
└── src/
    ├── solvers/
    │   └── hidenn_1d.py     # HiDeNN 核心求解器與總位能計算邏輯
    ├── nn_modules/
    │   └── shape_1d.py      # L2/L3 形狀函數的神經網路實作
    ├── benchmarks/
    │   └── bar_hard_case.py # 物理問題定義（邊界條件、體積力、解析解）
    └── utils/
        └── visualization_1d.py # 後處理、L2 誤差計算與繪圖工具
5. 結果視覺化 (Example Results)
本專案針對具有高梯度特性的 1D Bar 進行測試（Hard Case）。若啟動 r-adaptivity，可以觀察到節點（Final Nodes）會自動向應變梯度較大的區域聚集。