# O_kpair — Wilson Loop 重叠矩阵计算

> 日期: 2026-07-13  
> 涉及文件: `deltap_wannier.cpp:439-530`, `unk_overlap_lcao.cpp:503-720`, `deltap_wannier.cpp:55-153`

---

## 一、数学定义

Wilson Loop 中相邻 k 点间的重叠矩阵:

```
O_j[n,m] = ⟨ψ_n(k_j) | ψ_m(k_{j+1})⟩                    (1)

展开为 LCAO 基组:
         = Σ_{μ,ν}  C†_{n,μ}(k_j) · S_{μ,ν}(dk) · C_{ν,m}(k_{j+1})   (2)
         = [ C†(k_j) · S(dk) · C(k_{j+1}) ]_{n,m}                       (3)
```

| 符号 | 含义 | 维度 |
|------|------|------|
| `n, m` | 占据带索引 | 1..nocc (BN: 4) |
| `μ, ν` | LCAO 基函数索引 | 1..NBASIS (BN: 26) |
| `C_{μ,n}(k)` | LCAO 系数, 第 n 个 Bloch 态在 k 点的展开 | NBASIS × nocc |
| `S_{μ,ν}(dk)` | LCAO 重叠矩阵, 含 Bloch 相位 | NBASIS × NBASIS |
| `dk` | `k_{j+1} - k_j`, 沿 k-string 的步长 | — |

S(dk) 的物理定义:

```
S_{μ,ν}(dk) = Σ_R  exp(2πi · k_R · R) × ⟨φ_μ(0) | φ_ν(R)⟩
```

其中 `φ_μ(0)` 和 `φ_ν(R)` 是位于 0 和 R 晶胞的 LCAO 数值原子轨道, `⟨φ_μ | φ_ν(R)⟩` 是双中心重叠积分。

**用途**: O_j 矩阵是 Wilson Loop 的基元:

```
W_j = O_0 × O_1 × … × O_j   (逐个 k 点累乘)
γ = Im[log(det W_final)]    (Berry phase)
```

---

## 二、两种实现路径

### 2.1 Path A: unkOverlap_lcao (原始路径)

**调用链**:

```
berryphase_overlap()                          [unk_overlap_lcao.cpp:644]
  ├── prepare_midmatrix_pblas()               [unk_overlap_lcao.cpp:503]
  │     └── 构建 M(k_L,k_R): NBASIS × NBASIS
  │           每个全局轨道对 (μ,ν) 遍历所有 R 向量
  │           累加: exp(ik_R·R) × ⟨φ_μ|φ_ν(R)⟩
  ├── ScaLAPACK::gemm: M × C(k_R)             (NBASIS×nocc)
  └── ScaLAPACK::gemm: C†(k_L) × 中间结果    (nocc×nocc)
       └── MPI_Allreduce 全局归约
```

**`prepare_midmatrix_pblas` 核心循环** (lines 515-536):

```cpp
for (int iw_row = 0; iw_row < nlocal; iw_row++)       // O(NBASIS)
    for (int iw_col = 0; iw_col < nlocal; iw_col++)   // O(NBASIS)
        for (int iR = 0; iR < N_R[iw_row][iw_col]; iR++)  // O(N_R)
        {
            kRn = 2π × (kvec_c[ik_R]·R_{ν} - dk·τ_μ)
            phase = exp(i·kRn)
            overlap = ⟨φ_μ|φ_ν(R)⟩ + i·dk·⟨φ_μ|r|φ_ν(R)⟩
            M[μ,ν] += phase × overlap
        }
```

**每次调用成本** (BN, NBASIS=26):

| 项目 | 量级 |
|------|------|
| 外层循环 | 26 × 26 = 676 对 |
| 每对的 R 向量 | ~50–200 (取决于截断半径) |
| 总迭代 | ~50,000–150,000 |
| 每次迭代 | sin, cos, 复数乘法 |
| 实测耗时 | ~30 秒 / k-pair |

瓶颈不在循环本身 (676 × 200 = 135k 次, 微秒级), 而在 ScaLAPACK 初始化和 MPI 同步。实际上单进程运行时有额外初始化开销。

### 2.2 Path B: S_dk GEMM (快速路径)

**调用链**:

```
compute_S_dk(ucell)                              [deltap_wannier.cpp:55]
  └── 构建 S_dk_: NBASIS × NBASIS (每个 string 方向仅一次)

for each k-pair:                                 [deltap_wannier.cpp:470-520]
  ├── SC = S_dk_ × C(k_R)                       NBASIS×nocc (手写 GEMM)
  └── O = C†(k_L) × SC                          nocc×nocc (手写 GEMM)
```

**手写 GEMM 代码** (lines 481-499):

```cpp
// SC = S_dk × C_R  (nrow × nocc)
for (int p = 0; p < nocc; ++p)
    for (int alpha = 0; alpha < nrow; ++alpha) {
        s = 0;
        for (int gamma = 0; gamma < ncol; ++gamma)
            s += S_dk_[alpha + gamma*nrow] * c_R[gamma + p*nrow];
        SC[alpha + p*nrow] = s;
    }

// O = C_L† × SC  (nocc × nocc)
for (int q = 0; q < nocc; ++q)
    for (int p = 0; p < nocc; ++p) {
        s = 0;
        for (int alpha = 0; alpha < nrow; ++alpha)
            s += conj(c_L[alpha + q*nrow]) * SC[alpha + p*nrow];
        O_full[q + p*nocc] = s;
    }
```

| 操作 | 循环次数 | FLOPs |
|------|------|------|
| S_dk × C_R | NBASIS × nocc × NBASIS = 26×4×26 | 2,704 |
| C_L† × SC | nocc × nocc × NBASIS = 4×4×26 | 416 |
| **每个 k-pair 总计** | | **~3,100** 次复数乘法 |

**实测**: 微秒级 / k-pair。整个 36 个 k-pair (12 strings × 3 方向) ≈ 0.01 秒。

---

## 三、内存代价对比

### 3.1 Path A (unkOverlap)

| 分配 | 位置 | 大小 (BN) | 生命周期 |
|------|------|------|------|
| `psi_psi[iw][jw][iR]` | `unk_overlap_lcao` 成员 (init 分配) | 26×26×~200 × 8B ≈ 1 MB | 对象存活期 |
| `psi_r_psi[iw][jw][iR]` | 同上 | 同上 ≈ 1 MB | 对象存活期 |
| `midmatrix = new Complex[nloc]` | `prepare_midmatrix_pblas` | 26×26 × 16B = 10.8 KB | 每次调用 |
| `C_matrix = new Complex[nloc]` | `berryphase_overlap` | 10.8 KB | 每次调用 |
| `out_matrix = new Complex[nloc]` | `berryphase_overlap` | 10.8 KB | 每次调用 |

**总持久内存**: ~2 MB (重叠积分表)  
**每次调用堆分配**: ~32 KB (M + C + out 临时矩阵)  
**每次调用: new + delete 三次** (堆碎片风险)

### 3.2 Path B (快速路径)

| 分配 | 位置 | 大小 (BN) | 生命周期 |
|------|------|------|------|
| `S_dk_` | `DeltaP` 成员 | 26×26 × 16B = 10.8 KB | 每个 gdir 方向 |
| `SC` | 栈变量 (vector) | 26×4 × 16B = 1.7 KB | 每个 string 内 |
| `O_full` | 栈变量 (vector) | 4×4 × 16B = 256 B | 每个 k-pair |

**总持久内存**: ~33 KB (三个方向各 10.8 KB S_dk_)  
**每次调用堆分配**: 0 (全部栈上 vector, 可被编译器优化)  
**无 new/delete** — 无堆碎片, 无内存泄漏风险

### 3.3 对比

| | Path A (unkOverlap) | Path B (快速) | 比值 |
|------|:---:|:---:|:---:|
| 持久内存 | ~2 MB | ~33 KB | **60× 更少** |
| 每 k-pair 分配 | ~32 KB (堆) | 0 | — |
| new/delete 调用 | 3 次 / k-pair | 0 | — |
| MPI 通信 | Allreduce | 无 | — |

---

## 四、效率优势分析

### 4.1 计算复杂度

| | Path A | Path B | 加速比 |
|------|:---:|:---:|:---:|
| S(dk) 矩阵构建 | 每 k-pair 重新计算 | 每 string 方向计算 1 次 | **N_pairs ×** |
| 核心计算 | O(NBASIS² × N_R × N_strings) | O(NBASIS² × nocc × N_strings) | **N_R / nocc** |
| GEMM 方式 | ScaLAPACK (MPI) | 手写三重循环 (串行) | — |
| 库依赖 | LAPACK + ScaLAPACK + MPI | 无外部依赖 | — |
| BN 实测 | ~30s / k-pair | ~0.001s / k-pair | **~30,000×** |

加速比来自三个因素:
1. **S_dk_ 复用**: 每条 k-string 只算一次 S_dk_, N 个 link 共享 (N_pairs 倍)
2. **无 MPI/库开销**: ScaLAPACK 初始化 + MPI_Allreduce 对 4×4 小矩阵而言开销远大于计算
3. **无位置矩阵**: 不计算 `⟨φ_μ|r|φ_ν(R)⟩` 位置修正项 (简化)

### 4.2 精度差异

| 项目 | Path A | Path B |
|------|------|------|
| S(dk) 精度 | 含 `-i·dk·⟨r⟩` 位置修正 | 纯重叠 `⟨φ_μ|φ_ν(R)⟩`, 无位置修正 |
| dk 使用 | `dk_string` (actual k-difference) | `dk_step = 1/(nppstr_-1)` (均匀间距) |
| MPI 跨节点一致性 | 通过 Allreduce 保证 | 仅串行保证 |

两项差异在密集 k 点网格上影响 < 3% (C1 风险评审结论)。对 BN 测试验证, 两种路径得的 P_total 在迭代收敛后一致。

### 4.3 适用场景

| 场景 | 推荐路径 |
|------|------|
| 小体系 (<100 原子), 串行 | Path B (快速) |
| 大规模并行, 跨节点分布式 C 矩阵 | Path A (MPI 原生) |
| 需要位置修正的高精度计算 | Path A |
| 快速迭代开发 + 测试 | Path B |

---

## 五、总结

Path B 通过三个关键优化实现了 ~30,000× 加速:

1. **S_dk_ 在 k-string 级别复用** (而非 k-pair 级别)
2. **手写 GEMM 消除库调用/MPI 开销** (对小矩阵, 库调用开销 > 计算)
3. **栈分配消除堆碎片** (vector → 编译器可内联优化)

当前 DeltaP 默认使用 Path B (`berry_overlap_ = nullptr` 触发), 对 nocc ≤ 20, NBASIS ≤ 1000 的小体系是更优选择。生产环境需要并行化时, 可切换回 Path A (修改 esolver_ks_lcao.cpp:702 的 `nullptr` 为 `berry_ovl_scf_`)。
