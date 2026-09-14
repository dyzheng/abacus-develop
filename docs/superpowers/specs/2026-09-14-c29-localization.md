# 2026-09-14 C-29 定位：收尾堆破坏 = 重启电荷文件的平面波基组不匹配

- 前置：Active Bug C-29（`deltap-development-log.md`）；用户批复的后续优先级第 1 项
  「C-29 定位（最高优先）」，且新事实指出"崩溃发生在 `!FINAL_ETOT_IS` 打印之后，
  物理数据在崩溃前完整"，据此把范围收窄到"能量决算后的清理路径"。
- 结论摘要：**该新事实只提供了"检测点"，不是"破坏点"**。真正的非法写发生在
  **run 开头读重启电荷文件**时（`before_all_runners` → `Charge::init_rho` →
  `ModuleIO::read_rhog`），破坏 malloc chunk 头；glibc 直到收尾释放相邻数组时才
  报错，于是表现为"决算后清理路径崩溃"。C-29 与约束框架**无关**（关掉 constraint
  同样复现）。
- 证据目录：`tests/deltap_c29/`（README + results/）。

## 1. 测试计划

| # | 问题 | 判据 |
|---|---|---|
| Q1 | 崩溃的**检测点**在哪个调用栈/哪次 free？ | 非 ASAN 调试构建 + gdb 捕获 abort 栈 |
| Q2 | 首次**非法写**在哪一行、写到哪、越界多少？ | ASAN（`ENABLE_ASAN=ON`）首次非法写即报 |
| Q3 | 是否为约束框架/双迭代引入？ | 关掉 `constraint` 复跑同一输入；父提交二进制已复现过 |
| Q4 | 触发条件到底是什么？ | 保持体系/参数不变，只切换热启动源（无 / 同基组 / 更大基组） |
| Q5 | 最小修复能否消除症状？ | 加 `ig < 0` 守卫后 ASAN 复跑原崩溃输入 + 单元回归测试 |

## 2. 测试环境

- 体系：MgO rocksalt 8 原子胞，LCAO（Mg/O ONCV-PBE + `*_gga_*` 轨道），
  `symmetry 0`，2×2×2 MP，`ks_solver genelpa`，`scf_thr 1e-8`，Broyden 0.7；
  约束：charge，`constraint_weight_type becke`，`constraint_thr 1e-4`，靶点 δ=+0.5 e
  （个别对照用 δ=+0.1 e）。
- 运行：`np=4`，`OMP_NUM_THREADS=1`（与既有对照口径一致，见用户手册/开发者文档 §3.5）。
- 二进制：
  - 调试构建 `build/abacus_basic_para`（`CMAKE_BUILD_TYPE=debug`，含当前 HEAD）；
  - ASAN 构建 `build_asan/abacus_basic_para`（`-DENABLE_ASAN=ON`，Debug）；
  - 热启动源 `/tmp/mgo_scan/S3_O_+0p3/OUT.autotest`（**9-11 旧扫描，160/640**）与
    `/tmp/c29/quick_constrained_0p1/OUT.autotest`（本次 60/240，同基组对照）。
- gdb/valgrind 可用；`ENABLE_ASAN` 由 `CMakeLists.txt` 提供。

## 3. 结果

### R1（Q1）检测点：`Charge::destroy` 收尾 free（非 ASAN + gdb）

`np=4`、δ=+0.5、读取 160/640 的重启文件；SCF 收敛（`final status: CONVERGED`）之后
四个 rank 同时 abort，栈为：

```
#6  malloc_printerr  "free(): invalid next size (normal)"   (或 "corrupted size vs. prev_size")
#7  _int_free_merge_chunk / unlink_chunk
#10 Charge::destroy            charge.cpp:80   (delete[] _space_rho)
#12 Charge::destroy            charge.cpp:79   (delete[] rhog_core)
#14 ESolver_FP::~ESolver_FP    esolver_fp.cpp:38
#16 ESolver_KS_LCAO<...>::~ESolver_KS_LCAO
#16 Driver::driver_run         driver_run.cpp:104
```

即：**检测点在收尾清理路径**（与分析"物理数据完整"一致），但只能说明"某块
chunk 头早已被写坏"。

### R2（Q2）破坏点：`read_rhog` 越界写 16 字节（ASAN）

同一输入在 ASAN 构建下 **2.6 秒**即报（`results/asan_mismatch_constraint_on.report`）：

```
ERROR: AddressSanitizer: heap-buffer-overflow ... WRITE of size 16
  #0 ModuleIO::read_rhog(...)            rhog_io.cpp:166
  #1 Charge::init_rho(...)               charge_init.cpp:51
  #2 ESolver_KS::before_all_runners(...) esolver_ks.cpp:73
  #4 Driver::driver_run()                driver_run.cpp:67
0x5340000a07f0 is located 16 bytes before 125904-byte region
allocated by:  Charge::allocate(int const&, bool)  charge.cpp:126  (new std::complex<double>[nspin*ngmc])
```

四个 rank 各报一次，地址同构。`WRITE of size 16` = 一个 `std::complex<double>`；
"16 bytes before" = `rhog[is][-1]`。

### R3（Q3）与约束框架无关

- 同一输入把 `constraint*` 全部关掉：ASAN 报**完全相同**的溢出
  （`results/asan_mismatch_constraint_off.report`）；
- 与双迭代调度无关（OUTER/INNER 都中，父提交二进制也中——此前已记录）。

### R4（Q4）触发条件 = 重启文件的基组**更大**

只改 `read_file_dir`（其余全同，δ=+0.1 e）：

| 热启动 | 基组关系 | 结果 |
|---|---|---|
| 无 | — | `rc=0`，`TOTAL Time 177`，破坏行 0 |
| `S3_O_+0p3/OUT.autotest`（160/640） | 文件**更大** | SCF 收敛、CONVERGED，随后 4 rank 堆破坏 abort（`results/repro_quick_delta0p1.txt`） |
| `quick_constrained_0p1/OUT.autotest`（60/240） | 一致 | ASAN 干净、正常结束（`results/asan_control_matching_basis.txt`） |

`rhog_io.cpp` 头部对 `npwtot_in > pw_rhod->npwtot` 只发 `WARNING "some planewaves
in file are not used"`（不返回错误），因此该路径静默继续。

### R5（Q5）最小修复 + 回归测试

- 修复：映射出 `ig` 后加 `if (ig < 0) { continue; }`（基组外平面波用不上）；
- 修复后 ASAN 复跑**原崩溃输入**：0 条 ASAN 报告、0 条堆破坏，
  `START 08:32:45 / FINISH 08:34:08 / TOTAL Time 83`，
  `settle check PASSED` + `final status: CONVERGED`（`results/asan_after_fix.txt`）；
- 再用**普通调试二进制**跑**原崩溃算例本身**（非 ASAN，`np=4`）：
  `rc=0`（不再 abort）、0 条堆破坏、`final status: CONVERGED`、`TOTAL Time 93`
  （`results/e2e_after_fix.txt`）；
- 新单测 `ReadRhogTest.LargerBasisInFileDoesNotWriteBeforeBuffer`：把 `rhog[0]`
  指向"缓冲区前留一个哨兵槽"的 vector，读入 160/640 写的 `charge-density.dat`
  （`npwtot_in = 1471`）到一个 30 Ry 的小球 → 断言哨兵未被改写；
  - 有守卫：**PASS**（`ctest -R read_rhog` 1/1）；
  - 拆掉守卫（`git stash` 后重建）：**FAIL**，实测哨兵 imag 被写成 `-2.6e-14`
    ——正是文件中 `ig == -1` 那个系数，直接证明越界写的就是它。

## 4. 分析

**根因**：`read_rhog` 用 `fftixyz2ig[fftixyz]` 把"文件里的 Miller 下标"翻译成"当前
基组的本地平面波下标"，但该表对**当前基组外**的项保持 `-1`（表的用途是反向映射），
代码没有判 `-1` 就直接 `rhog[is][ig] = rhog_in[i]`。文件基组更大时（例如跨 `ecutwfc`
热启动），文件里存在大量"在 FFT 盒内、但在当前球外"的平面波，每次这类平面波都会
把 16 字节写到 `rhog[is] - 16`，即 malloc 的 chunk 头：

- 覆盖 `prev_size`/`size` 字段 → glibc 在**后续**对该 chunk 或其邻居做
  `free`/合并时才校验 → 报 `free(): invalid next size` / `corrupted size vs. prev_size`；
- 破坏的是元数据，**不影响密度数值**，所以 μ*/`E_tot`/审计行仍然正确。

**为什么此前被误判为"约束/MgO 专属"**：全部历史复现都在**跨基组热启动**的 run 上
（9-14 的 MgO 长跑把 9-11 的 160/640 输出当 60/240 的 `read_file_dir`）；而 9-11
那批 160/640 扫描内部互相热启动（基组一致）从不崩，于是被读成"MgO 体系/约束特性"。
关掉约束即复现，直接否掉了约束假设。

**与"决算后清理路径"的关系**：用户的收窄方向正确地描述了**观测**（abort 在
`!FINAL_ETOT_IS` 之后、`Charge::destroy` 里），但那是**检测**位置；写入位置在
run 开头。这也是"物理数据完整"的原因——被破坏的只有堆元数据。

## 5. 下一步

1. 修复落地（当前在**工作区、未提交**）：`rhog_io.cpp` 守卫 + 新的 `read_rhog` 单测；
   建议作为第 3 个 commit（`fix(io)`）入库，文档随行（本 spec + 证据目录 +
   开发者文档补 C-29 条目）。
2. 是否要把"文件基组更大"从 WARNING 升级为可配置的严格模式（`read_rhog` 返回
   false 走 cube 回退）——待决策；当前保持最小行为改动。
3. 同名/同族排查：`read_rhog` 之外还有 `write_rhog` 的 Miller 表使用；以及
   `Charge::init_rho` 的 cube 回退路径（`read_vdata_palgrid`）是否也存在跨网格尺寸
   假设，值得一并审计。
4. C-29 关闭后，MgO 类长跑不再需要"只信已落盘输出"的被动纪律（保留 `timeout`
   仍属好习惯）；`inner_thr` 标定（用户优先级 2）可解锁。
