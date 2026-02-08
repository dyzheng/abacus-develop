"""
Compare NEB results from ESolver mode and Driver mode.

Reads the .npz and .json output files produced by neb_esolver.py and
neb_driver.py, then prints a side-by-side comparison of energies,
barriers, and wall-clock times.

Usage:
    # Run both NEB scripts first, then:
    python compare_results.py
"""

import json
import sys
import numpy as np
from pathlib import Path


def load_results(prefix):
    """Load result .npz and timing .json for a given mode prefix."""
    npz_path = f"neb_{prefix}_result.npz"
    json_path = f"neb_{prefix}_timing.json"

    data = {}
    if Path(npz_path).exists():
        with np.load(npz_path, allow_pickle=True) as f:
            for k in f.files:
                data[k] = f[k]
    else:
        print(f"WARNING: {npz_path} not found.")

    timing = {}
    if Path(json_path).exists():
        with open(json_path) as f:
            timing = json.load(f)
    else:
        print(f"WARNING: {json_path} not found.")

    return data, timing


def fmt(val, unit=""):
    """Format a numeric value for display."""
    if isinstance(val, (np.ndarray, list)):
        return str(val)
    if isinstance(val, float):
        return f"{val:.6f} {unit}".strip()
    return str(val)


def compare():
    """Print side-by-side comparison."""
    es_data, es_time = load_results("esolver")
    dr_data, dr_time = load_results("driver")

    if not es_data and not dr_data:
        print("No result files found. Run neb_esolver.py and neb_driver.py first.")
        sys.exit(1)

    sep = "-" * 70
    print()
    print("=" * 70)
    print("  NEB Results Comparison: ESolver mode vs Driver mode")
    print("=" * 70)
    print()

    # --- Energies ---
    print(sep)
    print(f"{'Property':<30} {'ESolver':>18} {'Driver':>18}")
    print(sep)

    for key, label, _unit in [
        ("energy_IS",        "IS energy",         "eV"),
        ("energy_FS",        "FS energy",         "eV"),
        ("barrier_forward",  "Forward barrier",   "eV"),
        ("barrier_reverse",  "Reverse barrier",   "eV"),
    ]:
        es_val = float(es_data[key]) if key in es_data else None
        dr_val = float(dr_data[key]) if key in dr_data else None
        es_str = f"{es_val:.6f}" if es_val is not None else "N/A"
        dr_str = f"{dr_val:.6f}" if dr_val is not None else "N/A"
        print(f"  {label:<28} {es_str:>18} {dr_str:>18}")

    # --- Energy difference ---
    if "barrier_forward" in es_data and "barrier_forward" in dr_data:
        diff = float(es_data["barrier_forward"]) - float(dr_data["barrier_forward"])
        print(f"  {'Barrier diff (ES-DR)':<28} {diff:>18.6f}")
    print()

    # --- Image energies ---
    if "image_energies" in es_data or "image_energies" in dr_data:
        print(sep)
        print(f"{'Image':<10} {'ESolver (eV)':>18} {'Driver (eV)':>18} {'Diff (eV)':>18}")
        print(sep)

        es_imgs = es_data.get("image_energies", [])
        dr_imgs = dr_data.get("image_energies", [])
        n = max(len(es_imgs), len(dr_imgs))

        for i in range(n):
            es_e = float(es_imgs[i]) if i < len(es_imgs) else None
            dr_e = float(dr_imgs[i]) if i < len(dr_imgs) else None
            es_str = f"{es_e:.6f}" if es_e is not None else "N/A"
            dr_str = f"{dr_e:.6f}" if dr_e is not None else "N/A"
            diff_str = ""
            if es_e is not None and dr_e is not None:
                diff_str = f"{es_e - dr_e:.6f}"
            print(f"  {i:<8} {es_str:>18} {dr_str:>18} {diff_str:>18}")
        print()

    # --- Timing ---
    print(sep)
    print(f"{'Timing':<30} {'ESolver (s)':>18} {'Driver (s)':>18}")
    print(sep)

    all_keys = sorted(set(list(es_time.keys()) + list(dr_time.keys())))
    for key in all_keys:
        es_t = es_time.get(key)
        dr_t = dr_time.get(key)
        es_str = f"{es_t:.1f}" if es_t is not None else "N/A"
        dr_str = f"{dr_t:.1f}" if dr_t is not None else "N/A"
        print(f"  {key:<28} {es_str:>18} {dr_str:>18}")

    # Speedup
    if "total" in es_time and "total" in dr_time and es_time["total"] > 0:
        speedup = dr_time["total"] / es_time["total"]
        print()
        print(f"  Speedup (Driver/ESolver): {speedup:.2f}x")

    print()
    print("=" * 70)


if __name__ == "__main__":
    compare()
