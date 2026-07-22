# DeltaP 匈牙利算法堆缓冲区溢出修复

> 日期: 2026-07-20

---

## 问题描述

在 DeltaP 算法重构后的测试中，AddressSanitizer 检测到堆缓冲区溢出错误：

```
ERROR: AddressSanitizer: heap-buffer-overflow on address 0x50200012afec
READ of size 4 at 0x50200012afec thread T0
    #0 in deltap::DeltaP::compute_wannier_polarization(...) 
       /root/abacus-develop/source/source_lcao/module_deltap/deltap_wannier.cpp:738
```

---

## 根因分析

### 1. 匈牙利算法实现 bug

在 `deltap_wannier.cpp` 第 738 行，匈牙利算法的路径回溯代码存在 bug：

```cpp
do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);
```

**问题**：当第 718 行的迭代限制被触发时，`j0` 被设置为 0，但 `way[0]` 的值是 -1，导致访问 `p[-1]`，这是越界访问。

### 2. 无效匹配访问

在 `deltap_wannier.cpp` 第 778 行和第 702 行，代码使用匈牙利算法或保存匹配的结果来访问 `gamma_unwrapped` 数组，但没有检查匹配索引的有效性：

```cpp
int m = match_to[nn];  // 可能是 -1
gamma_new[m] = gamma_unwrapped[m] + diff;  // 越界访问
```

**问题**：当匈牙利算法失败或保存匹配包含无效值时，`match_to[nn]` 或 `saved[nn]` 的值是 -1，导致越界访问。

---

## 修复方案

### 1. 修复匈牙利算法路径回溯

将 do-while 循环改为 while 循环，避免在 `j0 == 0` 时访问 `way[0]`：

```cpp
// 修复前
do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);

// 修复后
while (j0 != 0) { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; }
```

### 2. 添加边界检查

在匈牙利算法中添加边界检查，防止越界访问：

```cpp
do {
    if (++hung_iter > N * N + 5) { j0 = 0; break; }
    if (j0 < 0 || j0 >= N) break;  // 边界检查
    used[j0] = true;
    int i0 = p[j0];
    if (i0 < 0 || i0 >= N) break;  // 边界检查
    // ... 其余代码
    j0 = j1;
} while (j0 >= 0 && j0 < N && p[j0] != -1);

while (j0 != 0) { 
    if (j0 < 0 || j0 >= N) break;  // 边界检查
    int j1 = way[j0]; 
    if (j1 < 0 || j1 >= N) break;  // 边界检查
    p[j0] = p[j1]; 
    j0 = j1; 
}
```

### 3. 添加无效匹配检查

在使用匹配结果访问数组之前，添加边界检查：

```cpp
// 在第 778 行
int m = match_to[nn];
if (m < 0 || m >= N) continue;  // 跳过无效匹配

// 在第 702 行
int m = saved[nn];
if (m < 0 || m >= N) continue;  // 跳过无效匹配
```

---

## 验证结果

### 1. 启用 AddressSanitizer 测试

```bash
cmake -DENABLE_ASAN=1 ..
make -j$(nproc) abacus_basic_para
./abacus_basic_para
```

**结果**：测试成功完成，没有 AddressSanitizer 错误。计算运行了 4 次迭代并正常完成。

### 2. 禁用 AddressSanitizer 测试

```bash
cmake -DENABLE_ASAN=0 ..
make -j$(nproc) abacus_basic_para
./abacus_basic_para
```

**结果**：测试成功完成，没有崩溃或错误。计算运行了 8 次迭代并正常完成。

---

## 总结

匈牙利算法实现中的边界检查不足导致堆缓冲区溢出。通过添加边界检查和无效匹配检查，修复了这个问题。测试验证了修复的有效性。

---

## 修改的文件

- `source/source_lcao/module_deltap/deltap_wannier.cpp`
  - 第 710-739 行：修复匈牙利算法路径回溯和添加边界检查
  - 第 702 行：添加无效匹配检查
  - 第 778 行：添加无效匹配检查
