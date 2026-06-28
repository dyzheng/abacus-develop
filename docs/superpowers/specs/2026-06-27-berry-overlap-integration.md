# berryphase_overlap 集成与 zeta 对比：执行记录

> **日期**: 2026-06-27
> **修改**: 添加 `berryphase_overlap` 函数 + 集成到 DeltaP

---

## 1. 代码修改

### 1.1 新增 `berryphase_overlap` 函数

在 `unk_overlap_lcao.h/cpp` 中添加 `berryphase_overlap`:
- 与 `det_berryphase` 使用相同的 `prepare_midmatrix_pblas` + pzgemm
- 但返回完整的 O_j 矩阵 (nocc×nocc), 而非仅 det(O_j)
- O_j 在所有 MPI rank 上 replicated

### 1.2 DeltaP 集成

- `deltap.h`: 添加 `unkOverlap_lcao* berry_overlap_` 成员
- `deltap.cpp`: init 接受 `unkOverlap_lcao*` 参数
- `deltap_wannier.cpp`: 当 `berry_overlap_` 非空时, 调用 `berryphase_overlap` 获取 O_j
- `ctrl_scf_lcao.cpp`: 创建 `unkOverlap_lcao` 对象, 初始化后传给 DeltaP

### 1.3 初始化

```cpp
unkOverlap_lcao berry_overlap;
berry_overlap.init(ucell, kv.get_nkstot(), orb);
berry_overlap.cal_R_number(ucell, gd);
berry_overlap.cal_orb_overlap(ucell);  // 同时计算 psi_psi 和 psi_r_psi
```

与 berry_phase 的 `lcao_init` 完全一致。

---

## 2. 测试 1 结果: zeta_string 对比

### 2.1 结果

**arg(zeta) 仍然不一致**:
- berry arg[0] = -1.124, DeltaP arg[0] = +2.019 (diff ≈ π)
- mean |diff| = 1.265, max = 3.140

**|zeta| 不一致**:
- berry |z| ≈ 1.06, DeltaP |z| ≈ 0.29

**按 arg 值匹配**: 无法找到精确匹配 (最近 diff ≈ 0.02-0.05)

### 2.2 分析

尽管 `berryphase_overlap` 使用与 `det_berryphase` 相同的积分表和计算路径, zeta 仍不一致。

**可能原因**:

1. **k-string 排序不同**: berry_phase 的 `set_kpoints` 和 DeltaP 的 `setup_kstring` 可能以不同顺序遍历 k-strings。但按 arg 匹配也无法找到精确对应。

2. **归一化影响**: DeltaP 的 zeta = det(W_normalized), W 每步除以 max|element|。虽然 arg 应不变, 但数值误差可能累积。

3. **O_j 矩阵 vs det(O_j)**: DeltaP 做矩阵乘法 W = ∏O_j 再 det(W), berry 做标量乘积 ∏det(O_j)。矩阵乘法的归一化可能引入 arg 偏差。

4. **dk 差异**: berry_phase 在 `stringPhase` 中计算 dk = kvec_c[k_index[1]] - kvec_c[k_index[0]] (per string), DeltaP 用 dk_string (相同公式)。但 `prepare_midmatrix_pblas` 内部用 dk 计算 `dk * tau1`, 如果 dk 不同, midmatrix 不同。

### 2.3 下一步验证

**关键验证**: 对同一个 link j, 直接对比 O_j 矩阵 (而非 zeta)。
- 在 `berryphase_overlap` 中输出 O_j[0,0]
- 在 `det_berryphase` 中输出 O_j 的 det
- 如果 O_j 一致, det(O_j) 应该相同

如果 O_j 不一致, 问题在 `prepare_midmatrix_pblas` 的输入 (ik_L, ik_R, dk)。

