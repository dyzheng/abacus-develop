# AGENTS.md — ABACUS Development Rules

## Code Quality

### Branch Comments
Every `if-else` branch and non-trivial `if` block MUST be preceded by a comment explaining what the branch does and why. Example:

```cpp
// Branch A: inner-loop mode (deltap_nscf > 0)
// Lambda is optimized by BFGS with frozen charge density.
// The inner loop runs inside hamilt2rho_single like deltaspin.
if (dp->inner_loop_active() && iter > 1)
{
    ...
}
else
{
    // Branch B: synchronous mode (deltap_nscf == 0)
    // Lambda is updated via simple gradient descent in iter_finish.
    // No action needed here — the normal HSolver runs below.
}
```

### Dangerous Syntax
- **NO `goto` statements** — use structured control flow (if-else, break, continue)
- **NO `setjmp`/`longjmp`**
- When function grows >300 lines, refactor into smaller helpers

### Memory Safety
- `new`/`delete` should only be used when RAII (`std::vector`, `std::unique_ptr`) is impractical
- `std::vector::data()` is preferred over raw C arrays and pointers

## Documentation Rules

### Per-Round Development Logs
**EVERY round of modification and testing** MUST produce a dated document in `docs/superpowers/specs/` with the format `YYYY-MM-DD-<topic>.md`. Each document must contain:

1. **Test plan** — what you intend to verify
2. **Test setup** — system, parameters, input files used
3. **Results** — raw output data, pass/fail status
4. **Analysis** — interpretation of results, root cause if failed
5. **Next steps** — what to do in the next round

**No modification is considered complete until documented.**

### DeltaP Long-Context Document
Maintain `docs/superpowers/specs/deltap-development-log.md` as a running log. After each round:
1. Append a dated section summarizing what was done
2. Update the file list (what was modified)
3. Update the bug/fix list
4. Update the next-steps list

This document serves as compressed context for long-term development across sessions.
