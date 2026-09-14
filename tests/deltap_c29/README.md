# C-29 定位：约束 LCAO「收尾阶段堆破坏」的真身是**重启电荷文件基组不匹配**

Spec：`docs/superpowers/specs/2026-09-14-c29-localization.md`。
历史状态：`docs/superpowers/specs/deltap-development-log.md` 的 Active Bug C-29。

## 一句话结论

C-29 **不在约束框架里，也不是"能量决算后的清理路径"**。它是
`ModuleIO::read_rhog()`（`source/source_io/module_chgpot/rhog_io.cpp`）在读
`-CHARGE-DENSITY.restart` 时**缺少一个 `ig < 0` 守卫**：

- `fftixyz2ig` 只对**当前基组内**的平面波填了合法下标，其余保持 `-1`；
- 当文件是用**更大的平面波基组**（更大的 `ecutwfc`）写的，文件里那些"在 FFT 盒内、
  但在当前基组外"的 Miller 下标就会映射到 `ig == -1`；
- `rhog[is][ig] = ...` 于是落在 **`rhog[-1]`，即缓冲区前 16 字节**——正好是 malloc
  chunk 头。ASAN 报的正是 `WRITE of size 16 at 16 bytes before` 那块 125904 字节区域。

**为什么表现为"收尾崩溃"**：往 chunk 头里写 16 字节不会立刻被 glibc 发现；直到
`Charge::~Charge` 释放相邻数组时才校验 chunk 头 → `free(): invalid next size (normal)`
/ `corrupted size vs. prev_size` → abort。所以

- 崩溃点（检测点）= 收尾清理路径（`Charge::destroy`，`charge.cpp:79-80`）；
- 破坏点（写入点）= **run 开头读重启文件那一瞬间**（`before_all_runners`）。

这与"物理数据在崩溃前完整"完全自洽：被破坏的只是 malloc 元数据，密度本身照常读入。

## 为什么看起来像 MgO/约束专属

只是因为 **2026-09-14 那批 MgO 长跑的热启动源是 9-11 的旧扫描目录**：
`/tmp/mgo_scan/S3_O_+0p3`（`ecutwfc 160 / ecutrho 640`）被当成了
`60 / 240` 运行的 `read_file_dir`。而 9-11 的 S3 扫描自身在 160/640 内互相热启动
（基组一致）→ 从不崩。同样地，9-14 的 δ=+0.1 快速复现（无热启动）也干净。

**与约束模块无关**：把 `constraint` 全部关掉，同一输入在 ASAN 下报同一个溢出
（见 `results/asan_mismatch_constraint_off.report`）。

## 复现（最小）

同一 MgO 输入、同一约束靶点，只改 `read_file_dir`：

| 热启动源 | 基组 | 结果 |
|---|---|---|
| 无（原子叠加初猜） | — | `rc=0`，`TOTAL Time 177`，0 条破坏（`results/repro_quick_delta0p1.txt` §1） |
| 160/640 的旧扫描 `OUT.autotest` | **更大** | SCF 收敛、`final status: CONVERGED`，随后 4 rank 全部堆破坏 abort（同文件 §2） |
| 60/240 的正常重启文件 | 一致 | ASAN 干净、run 正常结束（`results/asan_control_matching_basis.txt`） |

ASAN 版本（`-DENABLE_ASAN=ON`）在 2.6 秒内即报第一次非法写，栈顶就是
`ModuleIO::read_rhog`。

## 修复

`source/source_io/module_chgpot/rhog_io.cpp`：映射后加 `if (ig < 0) continue;`
（基组外的平面波本来就用不上，rank 0 早已警告 "some planewaves in file are not used"）。
回归测试：`source/source_io/test/read_rhog_test.cpp` 新增
`ReadRhogTest.LargerBasisInFileDoesNotWriteBeforeBuffer`（缓冲区前放哨兵槽，
不依赖 ASAN 即可判红/判绿；拆掉守卫时该测试**必红**，实测哨兵被写成
`-2.6e-14`——正是文件里 `ig == -1` 那个系数）。

修复后用 ASAN 跑原崩溃输入：**0 条 ASAN 报告、0 条堆破坏**，`TOTAL Time 83`
（`results/asan_after_fix.txt`）；再用**普通调试二进制**跑**原崩溃算例本身**：
`rc=0`、0 条堆破坏、`final status: CONVERGED`、`TOTAL Time 93`
（`results/e2e_after_fix.txt`）。

## 影响与善后

- 受影响场景：`init_chg file` / `init_chg auto` 读**更大基组**写的
  `-CHARGE-DENSITY.restart`（改过 `ecutwfc`、或跨不同截断的扫描热启动）；
- 未修前的缓解：重启源与当前运行的 `ecutwfc/ecutrho` 保持一致；
- 历史结论仍然有效：崩溃点之前已落盘的 μ*、`E_tot`、`CONVERGED`/审计行可信。
