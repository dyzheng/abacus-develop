#!/usr/bin/env python3
"""
细粒度 Subagent Orchestrator: migrate zdy-tmp top 7 commits to dftu-pw-port

设计原则:
1. 每个子任务 10-20 分钟
2. 每个子任务有独立的 worktree
3. 实时 PROGRESS.md 轮询
4. 严格的验收检查清单
"""

import time
import subprocess
from datetime import datetime

# =============================================================================
# 配置区
# =============================================================================
TARGET_REPO = "/root/abacus-dftu-pw-port"
REF_REPO = "/root/abacus-zdy-tmp"
UPSTREAM_REPO = "/root/abacus-develop"
WORKTREE_BASE = "/tmp/subagent-work/zdy-migrate-" + datetime.now().strftime("%Y%m%d_%H%M%S")

# commit hash -> 修改文件列表 (zdy-tmp 路径)
COMMITS = {
    "19ade1859": ["source/module_hsolver/kernels/rocm/dngvd_op.hip.cu"],
    "e9e91d7fe": [
        "source/module_esolver/esolver_ks_pw.cpp",
        "source/module_hamilt_lcao/module_dftu/dftu_occup.cpp",
        "source/module_hamilt_lcao/module_dftu/dftu_pw.cpp",
    ],
    "a9d881c95": [
        "source/module_elecstate/module_charge/charge_mixing.h",
        "source/module_esolver/esolver_ks_pw.cpp",
    ],
    "bce760541": ["source/module_hsolver/kernels/rocm/dngvd_op.hip.cu"],
    "1a6871dca": [
        "source/module_hamilt_lcao/module_dftu/dftu.cpp",
        "source/module_hamilt_lcao/module_dftu/dftu_pw.cpp",
    ],
    "b9ce68339": ["source/module_hsolver/kernels/rocm/dngvd_op.hip.cu"],
    "34f564ef1": [
        "source/module_hamilt_pw/hamilt_pwdft/kernels/cuda/force_op.cu",
        "source/module_hamilt_pw/hamilt_pwdft/kernels/force_op.cpp",
    ],
}

# 路径映射: zdy-tmp -> dftu-pw-port
PATH_MAP = {
    "source/module_hamilt_lcao/module_dftu/": "source/source_lcao/module_dftu/",
    "source/module_hamilt_pw/hamilt_pwdft/": "source/source_pw/module_pwdft/",
    "source/module_esolver/": "source/source_esolver/",
    "source/module_elecstate/module_charge/": "source/source_estate/module_charge/",
    # dngvd 文件映射待定 (目标仓库可能不存在)
    "source/module_hsolver/kernels/rocm/": "source/source_hsolver/kernels/rocm/",
}


def map_path(zdy_path: str) -> str:
    """将 zdy-tmp 路径映射到 dftu-pw-port 路径"""
    for old, new in PATH_MAP.items():
        if zdy_path.startswith(old):
            return zdy_path.replace(old, new, 1)
    return zdy_path


def create_baseline_worktree(worktree: str):
    """创建隔离工作区并初始化 git"""
    subprocess.run(["mkdir", "-p", worktree], check=True)
    subprocess.run(["cp", "-r", f"{TARGET_REPO}/source", worktree + "/"], check=True)
    subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=worktree, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=worktree, check=True, capture_output=True)
    # 初始化 PROGRESS.md
    with open(f"{worktree}/PROGRESS.md", "w") as f:
        f.write(f"# Task Progress\\n\\n")


# =============================================================================
# 契约模板
# =============================================================================

def build_commit_migration_prompt(commit_hash: str, files: list, worktree: str) -> str:
    """构建单个 commit 迁移的 subagent 契约"""
    file_tasks = []
    for f in files:
        target_f = map_path(f)
        exists = os.path.exists(f"{TARGET_REPO}/{target_f}")
        file_tasks.append(f"""
- **zdy-tmp 文件**: `{f}`
- **目标文件**: `{target_f}`
- **目标文件存在**: {'是' if exists else '否 (需要调查映射关系)'}
""")

    return f"""## 任务：将 zdy-tmp commit `{commit_hash}` 迁移到 dftu-pw-port

### 输入
- 参考仓库：{REF_REPO}
- 目标工作区：{worktree}/source
- 上游 develop：{UPSTREAM_REPO}

### 需要处理的文件
{chr(10).join(file_tasks)}

### 执行步骤（必须严格按顺序）
1. **差异分析**：在参考仓库中执行 `git show {commit_hash}`，逐行理解修改意图
2. **状态调查**：检查目标工作区中对应文件的当前内容，判断该修改是否已经部分存在
3. **API 适配**：按照以下映射规则将修改适配到 dftu-pw-port 的代码风格：
   - `ModuleDFTU::DFTU` → `Plus_U`
   - `GlobalV::NSPIN` → `PARAM.inp.nspin`
   - `GlobalV::KPAR` → `PARAM.inp.kpar`
   - `GlobalC::ucell` → 参数传递 `const UnitCell& cell`
   - `psi_p->npol` → `psi_p->get_npol()`
   - `GlobalV::NPROC_IN_POOL` → `PARAM.globalv.nproc_in_pool`
4. **应用修改**：使用 `patch` 工具修改目标工作区的文件。如果直接 patch 不适用，则手动编辑
5. **编译检查**：在修改完成后，对涉及文件做最小编译验证（如 `g++ -std=c++11 -c <file> -o /dev/null`，或尝试完整 cmake build）
6. **写报告**：在 `{worktree}/MIGRATION_REPORT.md` 中记录：
   - 每处修改的对应关系
   - 遇到的 API 差异及处理
   - 编译结果

### 里程碑日志（必须实时更新）
每完成一个阶段，在 `{worktree}/PROGRESS.md` 追加一行（带时间戳）：
- `[HH:MM] 开始分析 commit {commit_hash}`
- `[HH:MM] 差异分析完成`
- `[HH:MM] 正在适配文件 X`
- `[HH:MM] 编译检查通过/失败`
- `[HH:MM] 任务完成`

### 特殊规则
- 如果目标文件不存在（如 `dngvd_op.hip.cu`），不要创建新文件。立即停止，在 `{worktree}/BLOCKERS.md` 中记录：
  `BLOCKER: 目标文件 {target_f} 不存在，需要调查 develop 中的对应物`
- 如果 20 分钟后仍未编译通过，停止工作并写 `BLOCKERS.md`
- 禁止修改任务范围外的文件
"""


def build_compile_fix_prompt(worktree: str, error_log: str) -> str:
    """编译修复子任务契约"""
    return f"""## 任务：修复编译错误

### 输入
- 工作区：{worktree}
- 编译错误日志：{error_log}

### 输出要求
1. 修复所有编译错误，使 `cmake --build .` 在该工作区成功（如果 build 目录存在）
2. 或者至少保证你修改的文件能独立编译通过
3. 每处修复必须在代码上方加 `// FIX:` 注释
4. 在 `{worktree}/PROGRESS.md` 记录修复列表

### 禁止事项
- 不要改变算法语义
- 如果错误根因是缺失文件，写 `{worktree}/BLOCKERS.md`
"""


def build_review_prompt(worktree: str) -> str:
    """代码审查子任务契约"""
    return f"""## 任务：审查迁移质量

### 输入
- 工作区：{worktree}
- 对比基线：{worktree} 的 baseline commit

### 审查维度
1. **逻辑一致性**：迁移后的代码是否与 zdy-tmp commit 的意图一致
2. **API 适配**：是否正确使用了 Plus_U、PARAM.inp.nspin 等 develop 风格 API
3. **编译安全**：是否有语法错误、类型不匹配、未声明变量
4. **范围控制**：是否只修改了目标文件

### 输出要求
1. 在 `{worktree}/REVIEW_REPORT.md` 中写入：
   - 总体结论：`PASS` / `NEEDS_FIX`
   - 逐条问题（位置 + 严重级别 + 建议）
2. 如果 `NEEDS_FIX`，给出具体的修改建议或代码片段

### 禁止事项
- 不要直接修改代码文件
"""


# =============================================================================
# 主 Orchestrator 逻辑
# =============================================================================

def run_orchestrator():
    """主控制流程"""
    print(f"[Orchestrator] 创建 baseline worktree: {WORKTREE_BASE}")
    create_baseline_worktree(WORKTREE_BASE)

    # 按 commit 串行处理（因为它们可能修改同一文件，串行更安全）
    for commit_hash, files in COMMITS.items():
        worktree = f"{WORKTREE_BASE}/{commit_hash}"
        print(f"\\n{'='*60}")
        print(f"[Orchestrator] 开始处理 commit {commit_hash}")
        print(f"[Orchestrator] worktree: {worktree}")

        # Step 1: 为当前 commit 创建隔离目录
        subprocess.run(["mkdir", "-p", worktree], check=True)
        subprocess.run(["cp", "-r", f"{WORKTREE_BASE}/source", worktree + "/"], check=True)
        subprocess.run(["git", "init"], cwd=worktree, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=worktree, capture_output=True)
        subprocess.run(["git", "commit", "-m", "baseline"], cwd=worktree, capture_output=True)

        # Step 2: 分发迁移子任务
        # 注意：这里是提供给主 agent 的 prompt，实际执行时需要调用 delegate_task
        prompt = build_commit_migration_prompt(commit_hash, files, worktree)

        print(f"[Orchestrator] 请将以下任务分发给 subagent：")
        print(f"--- PROMPT START ---")
        print(prompt)
        print(f"--- PROMPT END ---")

        # Step 3: 轮询 PROGRESS.md
        print("[Orchestrator] 进入轮询模式（每 30 秒检查一次，最多 20 分钟）...")
        for i in range(40):
            time.sleep(30)
            progress_path = f"{worktree}/PROGRESS.md"
            blocker_path = f"{worktree}/BLOCKERS.md"

            if os.path.exists(blocker_path) and os.path.getsize(blocker_path) > 0:
                with open(blocker_path) as f:
                    print(f"[Orchestrator] ⚠️ BLOCKER 检测到: {f.read().strip()}")
                print(f"[Orchestrator] 暂停处理 commit {commit_hash}，请人工介入或调整任务后继续。")
                break

            if os.path.exists(progress_path):
                with open(progress_path) as f:
                    lines = f.read().strip().split("\\n")
                    if lines and lines[-1].strip():
                        print(f"  [{i*30}s] {lines[-1]}")
                    if any("任务完成" in l for l in lines):
                        print(f"[Orchestrator] ✅ commit {commit_hash} 子任务报告完成")
                        break
        else:
            print(f"[Orchestrator] ⏰ commit {commit_hash} 超时")
            continue

        # Step 4: 验收检查
        report_path = f"{worktree}/MIGRATION_REPORT.md"
        if not os.path.exists(report_path):
            print(f"[Orchestrator] ❌ 缺少 MIGRATION_REPORT.md，验收失败")
            continue

        with open(report_path) as f:
            report = f.read()
        if "编译检查通过" not in report and "编译失败" not in report:
            print(f"[Orchestrator] ⚠️ 编译状态不明确，派发编译修复任务")
            # 这里会输出编译修复 prompt
            print(build_compile_fix_prompt(worktree, "/tmp/build.log"))
            continue

        if "编译失败" in report:
            print(f"[Orchestrator] ❌ 编译未通过，派发编译修复任务")
            print(build_compile_fix_prompt(worktree, "/tmp/build.log"))
            continue

        # Step 5: 代码审查
        print(f"[Orchestrator] 编译通过，派发代码审查任务...")
        review_prompt = build_review_prompt(worktree)
        print(review_prompt)

        # 审查完成后，如果 PASS，主 agent 可以执行 git diff 并合并
        print(f"[Orchestrator] 审查完成后，请手动验证并执行合并。")

    print(f"\\n{'='*60}")
    print("[Orchestrator] 所有 commit 处理完毕")


if __name__ == "__main__":
    run_orchestrator()
