# K-String 划分 + D_I_all 存储方案设计 (2026-07-11)

## 1. 当前 K-String 划分方案

### 1.1 K 点网格

ABACUS 用 Monkhorst-Pack 网格，k 点按 `nmp[0] × nmp[1] × nmp[2]` 均匀分布。
绝对 k-index: `ik = ix + iy·mp_x + iz·mp_x·mp_y`

### 1.2 沿极化方向的 K-String 划分

```cpp
// setup_kstring(): gdir=3 (z方向)
int mp_dir = nmp[2];                            // 沿z的K点密度
int num_string = nmp[0] * nmp[1];               // 垂直于z的(Kx,Ky)组合数
int nppstr_ = mp_dir + 1;                       // 每串K点数 (含闭合复制)

// 示例: nmp=[4,4,4], gdir=3
//   16 条串, 每串 5 个 K 点 (含 k_{N}=k_{0} 闭合)
//   每条串对应固定 (ix, iy), 沿 iz=0,1,2,3 排列
```

**K-Index 映射** (`k_index_[string][pos]`):

```
串 0  (ix=0,iy=0): k_index_[0]  = [0,  16, 32, 48, 0]    ← k₀闭合
串 1  (ix=1,iy=0): k_index_[1]  = [1,  17, 33, 49, 1]
串 2  (ix=2,iy=0): k_index_[2]  = [2,  18, 34, 50, 2]
...
串 15 (ix=3,iy=3): k_index_[15] = [15, 31, 47, 63, 15]
```

每条串贡献 gamma 的 1/16。Wilson loop: `W = Π (C†·S·C)`, gamma = Σ w_I · arg(eigenvalue)

### 1.3 当前 compute_hk_correction 仅修正串 0

```cpp
// 只遍历 k_index_[0][j]
for (int j = 0; j < nppstr_-1; ++j) {
    int ik_L = k_index_[0][j];     // 仅 0, 16, 32, 48 (4个K点)
    int ik_R = k_index_[0][j+1];
    // 用 kstring_data_[j].D_I 计算 HK correction
}
```

**问题**: 只修正了 4/64 (6%) 个 K 点, gamma 响应强度被稀释 ~16×.

## 2. D_I 数据结构

### 2.1 当前存储 (kstring_data_)

```
kstring_data_[j]      // j = 0..4 (nppstr_个位置)
  .S_k[iat][lm][mu]   // SMO 重叠矩阵元
  .dS_k[iat][...]     // SMO 重叠梯度
  .D_I[iat][lm][n]    // SMO 投影 <α_iat_lm | ψ_n>
```

- `iat`: 原子索引 (0..nat-1)
- `lm`: 投影子索引 (0..nproj_per_atom_[iat]-1)
- `n`: 能带索引 (0..nbands-1)
- `mu`: 基函数局部索引

**D_I 的计算** (compute_D_I):

```
D_I[iat][lm][n] = Σ_μ conj(S_k[iat][lm][μ]) · ψ_{n}(μ)
```

S_k 由 `compute_S_k(j)` 按位置 j 计算, ψ 从 `psi->get_pointer()` 读取当前 K 点。

### 2.2 问题: D_I 被不同串反复覆盖

`compute_wannier_polarization` 遍历所有 16 条串, 每条串都调用 `compute_D_I(j, psi, ...)` 覆盖 `kstring_data_[j].D_I`.

函数结束时, `kstring_data_[j]` 保存的是**最后一条串** (istring=15, kx=3,ky=3) 的 D_I.

但 `compute_hk_correction` 用 `k_index_[0][j]` 访问 K 点 — 这是串 0 的 K 点, D_I 却是串 15 的!

### 2.3 D_I_all 存储设计

**新增成员** (`deltap.h`):

```cpp
// D_I_all[ik]: SMO projections at absolute k-point index
// D_I_all[ik][iat][lm][n] = <α_iat_lm | ψ_{n,k}>
// Iat loop: 0..nat-1, lm loop: 0..nproj[Iat]-1, n loop: 0..nbands-1
std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> D_I_all_;
```

**各维度大小**:

| 维度 | 符号 | 典型值(H2O) | 大系统 | 说明 |
|------|------|-----------|--------|------|
| K 点 | nks | 64 | 1000 | 随网格密度变化 |
| 原子 | nat | 3 | 10 | 元素种类无关 |
| 投影子 | Σ nproj_i | 17 | 200 | 基组相关 |
| 能带 | nbands | 7 | 200 | 电子数相关 |
| 总元素 | nks × Σ nproj × nbands | 64×17×7=7616 | 1000×200×200=40M | complex<double> |
| 内存 | | 122 KB | 640 MB | 大系统需注意 |

**存储时机**: 在 `compute_wannier_polarization` 中, 每串处理完后保存:

```cpp
// 在第435行 (compute_D_I 完成后):
for (int j = 0; j < nppstr_; ++j) {
    int ik = k_index_[istring][j];
    if (ik < nks && ik < (int)D_I_all_.size())
        D_I_all_[ik] = kstring_data_[j].D_I;
}
```

**使用**: compute_hk_correction 遍历所有串时取出:

```cpp
for (int istring = 0; istring < total_string_; ++istring) {
    for (int j = 0; j < nppstr_ - 1; ++j) {
        int ik_L = k_index_[istring][j];
        const auto& D_at_L = D_I_all_[ik_L];  // 该K点的正确投影
        // 用 D_at_L 计算 w_eff, 建立 H_sym
        hk_correction[ik_L] = H_sym;
    }
}
```

### 2.4 D_I_all 时效性

D_I 依赖 ψ (波函数)。内循环每步都调用 `compute_gamma_scf` → `compute_wannier_polarization`, 期间 ψ 不变化（内循环内 HSolver 操作后才变）。所以 D_I_all 在内循环步内始终有效。

下一内循环步 HSolver 更新 ψ 后, 下一次 `compute_gamma_scf` 会重新填充 D_I_all。

## 3. 实现步骤

1. 在 `deltap.h` 添加 `D_I_all_` 成员
2. 在 `deltap.cpp` init 中分配 `D_I_all_.resize(nks)`
3. 在 `compute_wannier_polarization` 中, MPI_Allreduce 完成后保存 D_I_all
4. 在 `compute_hk_correction` 中, 外层遍历所有串, 内层遍历每串的 link
5. 测试: 验证 gamma 响应增强 ~16×, lambda 可大幅降低
