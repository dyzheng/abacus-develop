# 2026-07-29 DeltaP Phase 2 验证报告

> 状态：代码修改完成，主二进制编译通过。计算验证待集群运行。

---

## 1. 已完成的代码修改（Phase 1 + Phase 2）

共 **12 个 DeltaP 专用 commits**（不含文档和构建修复）：

| Phase | Commits | 修复项 |
|-------|---------|--------|
| P1 (6) | C-01/03/04/06/08/09/10/12/13/14/15/19 | dp_escon MPI、PW λ分离、nrow守卫、rank文件I/O、hR重建、默认wannier等 |
| P2 (5) | C-05/11/02step1/07 | gdir重建、分支晶格统一、力τ_α/应力/×2、resta_z MPI |
| Build (1) | spin_constrain API + __LCAO | 预先存在构建错误 |
| Docs (2) | 评估报告 + 第二阶段审查+TODO | 文档 |

## 2. 待 HPC 集群运行的验证

### 2.1 TODO-3b: ×2 因子 + 力一致性 FD 验证

**测试位置**：`tests/deltap_fd_force/`

**运行方式**：
```bash
cd tests/deltap_fd_force
bash run_fd.sh h2o 0.005 2   # 使用 2 MPI 进程
```

**验收检查**：
- [ ] H2O（4 分子）所有原子 |F_FD − F_analytic| / |F_FD| < 5%
- [ ] BN 2×2×2 同上
- [ ] 如果 FD 偏差恰好 ~2× → 恢复 `force = force * 2.0`
- [ ] 如果 FD 偏差与原子分数坐标成正比 → ∂τ/∂R 项是关键缺失
- [ ] 如果 FD 偏差普遍 > 5% → 需要实现 H_HK 力贡献

### 2.2 C-11 分支晶格回归测试

**测试位置**：`tests/deltap_bn_sampling/`

**运行方式**：
```bash
# 使用现有 BN 9-point PES 采样脚本
cd tests/deltap_bn_sampling
bash run_all.sh   # 如存在
```

**验收检查**：
- [ ] 9/9 点约束收敛（|γ−t| < 0.05 rad）
- [ ] diag_minus 点不再出现 γ 分支跳变
- [ ] per-atom gamma 序列在相邻 target 点之间平滑

### 2.3 C-05 gdir≠3 方向测试

**验收检查**：
- [ ] BN gdir=1,2,3 各跑一次约束 SCF，γ 收敛到 target
- [ ] 三个方向的最终 λ 量级相当（极化率张量元素差 ~2× 以内）
- [ ] MPI np=2 不 crash

### 2.4 现有回归测试（CI 集成后）

**已有测试（在 tests/17_DS_DFTU/ 下）**：
| 编号 | 测试 | 系统 | 预期结果 |
|------|------|------|----------|
| 66 | BN nscf berry | BN (8×8×8) | γ 输出 = reference |
| 68 | H2O nscf berry | 4 H2O (4×4×4) | γ 输出 = reference |
| 19 | Si nscf | Si (10×10×10) | γ 输出 = reference |
| 20 | BTO Born | BaTiO3 | γ 输出 = reference |

**注意**：这些测试的输出文件（deltap_results.dat 等）在本次代码修改后**会发生变化**（C-11 分支平移量改变、C-02 力修正等），reference 文件需要重新生成。

## 3. 编译状态

```
abacus_basic_para: BUILD SUCCESS
2 test targets fail (pre-existing DeltaSpin linking issues, unrelated)
```

## 4. 未修复的已知问题（维护清单）

| ID | 简要 | 优先级 | 下次处理条件 |
|----|------|--------|-------------|
| C-02 ∂τ/∂R | 力缺 Hellmann-Feynman 项 | P1 | FD 验证确认该贡献显著时 |
| C-02 H_HK 力 | Berry 联络算符无力修正 | P2 | 需要 relax/MD 精确力时 |
| S-06 nspin 防护 | 不支持 nspin=2/4 | P2 | 有用户需求时 |
| S-07 非正交晶胞 | 极化换算假定正交 | P2 | 非立方体系用户报告问题时 |
| S-10 金属防护 | Wilson loop 无定义 | P3 | 文档声明限制 |
| Q-07 K=5 组合爆炸 | nocc>4 时 O(11^nocc) | P2 | 大体系用户报告 hang 时 |

## 5. 文档状态

| 文件 | 内容 |
|------|------|
| `2026-07-29-deltap-risk-assessment-review.md` | 61 项风险评估 |
| `2026-07-29-deltap-phase2-repair-plan.md` | 第二阶段修复计划 |
| `2026-07-29-deltap-phase2-review-todo.md` | Review + 具体 TODO |
| `deltap-development-log.md` | 开发日志（持续更新） |
| `tests/deltap_fd_force/` | FD 力验证脚本 + 输入 |

---

## 下一步

1. **在 HPC 集群上运行**：TODO-3b FD 验证 + C-11 回归 + 现有测试回归
2. **根据 FD 结果决定**：×2 恢复 or 保持移除；∂τ/∂R 是否需要实现
3. **更新 reference 文件**：CI 测试需要重新生成 deltap_results.dat baseline
4. **合并到 develop**：所有验证通过后
