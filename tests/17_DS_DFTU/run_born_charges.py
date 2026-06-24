#!/usr/bin/env python3
"""
Born effective charge calculation via finite differences.
Displaces each atom along z, runs SCF+NSCF, extracts P_z from berry_phase and DeltaP.
Z*_{I,zz} = Omega * [P_z(+dtau) - P_z(-dtau)] / (2*dtau)
"""
import os, sys, shutil, subprocess, re

# Configuration
BINARY = "/root/abacus-develop/build/abacus_basic_para"
BASE_DIR = "/root/abacus-develop/tests/17_DS_DFTU/20_LCAO_BTO_BORN"
PP_DIR = "/root/pporb/apns-pseudopotentials-v1"
ORB_DIR = "/root/pporb/apns-orbitals-precision-v1"
DTAU_BOHR = 0.05  # displacement in Bohr
LAT0 = 1.8897261254578284  # Bohr (LATTICE_CONSTANT in STRU)
C_VEC = 4.2  # z lattice vector component
DTAU_DIRECT = DTAU_BOHR / (LAT0 * C_VEC)  # in direct coordinates
CELL_VOLUME = (LAT0 * 4.0) * (LAT0 * 4.0) * (LAT0 * C_VEC)  # Bohr^3

# BaTiO3 atoms: (element, type_index, x, y, z)
ATOMS = [
    ("Ba", 0, 0.00, 0.00, 0.00),
    ("Ti", 1, 0.50, 0.50, 0.52),
    ("O",  2, 0.50, 0.50, 0.00),
    ("O",  2, 0.50, 0.00, 0.50),
    ("O",  2, 0.00, 0.50, 0.50),
]

STRU_TEMPLATE = """ATOMIC_SPECIES
Ba 137.328 Ba_ONCV_PBE-1.0.upf
Ti 47.867 Ti_ONCV_PBE-1.2.upf
O  15.999 O.upf

NUMERICAL_ORBITAL
Ba_gga_10au_100Ry_6s3p3d2f.orb
Ti_gga_10au_100Ry_6s3p3d2f.orb
O_gga_10au_100Ry_3s3p2d1f.orb

LATTICE_CONSTANT
{lat0}

LATTICE_VECTORS
4.00    0.00    0.00
0.00    4.00    0.00
0.00    0.00    4.20

ATOMIC_POSITIONS
Direct

Ba
0.0
1
{ba_x}   {ba_y}   {ba_z}

Ti
0.0
1
{ti_x}   {ti_y}   {ti_z}

O
0.0
1
{o1_x}   {o1_y}   {o1_z}

O
0.0
2
{o2_x}   {o2_y}   {o2_z}
{o3_x}   {o3_y}   {o3_z}
"""

INPUT_TEMPLATE = """INPUT_PARAMETERS
suffix    autotest
calculation    scf
basis_type    lcao
ecutwfc    100
gamma_only    0

nspin    1
scf_thr    1.0e-7
scf_nmax    100
out_chg    1
smearing_method    gauss
smearing_sigma    0.002
mixing_type    broyden
mixing_beta    0.4
ks_solver    genelpa
symmetry    0

pseudo_dir    {pp_dir}
orbital_dir    {orb_dir}
"""

INPUT_NSCF = """INPUT_PARAMETERS
suffix    autotest
calculation    nscf
basis_type    lcao
ecutwfc    100
gamma_only    0

nspin    1
scf_thr    1.0e-7
scf_nmax    50
out_chg    0
smearing_method    gauss
smearing_sigma    0.002
ks_solver    genelpa
symmetry    0
init_chg    file
read_file_dir    ./OUT.autotest

berry_phase    1
gdir    3

deltap_switch    1
deltap_rm    3.0
deltap_gdir    3
deltap_gauge_mode    smo_anchored
deltap_method    berry_connection

pseudo_dir    {pp_dir}
orbital_dir    {orb_dir}
"""

KPT = """K_POINTS
0
Monkhorst-Pack
10 10 10 0 0 0
"""

def write_stru(path, atoms_mod):
    """Write STRU file with modified atom positions."""
    fmt = {
        "lat0": LAT0,
        "ba_x": atoms_mod[0][2], "ba_y": atoms_mod[0][3], "ba_z": atoms_mod[0][4],
        "ti_x": atoms_mod[1][2], "ti_y": atoms_mod[1][3], "ti_z": atoms_mod[1][4],
        "o1_x": atoms_mod[2][2], "o1_y": atoms_mod[2][3], "o1_z": atoms_mod[2][4],
        "o2_x": atoms_mod[3][2], "o2_y": atoms_mod[3][3], "o2_z": atoms_mod[3][4],
        "o3_x": atoms_mod[4][2], "o3_y": atoms_mod[4][3], "o3_z": atoms_mod[4][4],
    }
    with open(path, "w") as f:
        f.write(STRU_TEMPLATE.format(**fmt))

def extract_berry_pz(log_path):
    """Extract P_z from berry_phase output."""
    try:
        with open(log_path) as f:
            content = f.read()
        # Look for: P = 0.0005102 (mod ...) (0, 0, 0.0005102) e/bohr^2
        match = re.search(r'P =\s+[\d.]+\s*\(mod\s+[\d.]+\)\s*\(\s*[\d.]+,\s*[\d.]+,\s*([\d.eE+-]+)\)\s*e/bohr', content)
        if match:
            return float(match.group(1))
    except:
        pass
    return None

def extract_deltap_pz(dat_path):
    """Extract P_total z from DeltaP output."""
    try:
        with open(dat_path) as f:
            for line in f:
                if line.startswith("# Total"):
                    parts = line.split()
                    return float(parts[-1])  # last number is Pz
    except:
        pass
    return None

def extract_deltap_per_atom(dat_path):
    """Extract per-atom Pz from DeltaP output."""
    results = []
    try:
        with open(dat_path) as f:
            for line in f:
                if not line.startswith("#") and not line.startswith(" ") and line.strip():
                    parts = line.split()
                    if len(parts) >= 6:
                        results.append(float(parts[-1]))  # Pz
    except:
        pass
    return results

def run_case(case_name, atoms_mod):
    """Run SCF + NSCF for a given atom configuration."""
    case_dir = os.path.join(BASE_DIR, case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    # Write files
    with open(os.path.join(case_dir, "INPUT"), "w") as f:
        f.write(INPUT_TEMPLATE.format(pp_dir=PP_DIR, orb_dir=ORB_DIR))
    with open(os.path.join(case_dir, "KPT"), "w") as f:
        f.write(KPT)
    write_stru(os.path.join(case_dir, "STRU"), atoms_mod)
    
    # Run SCF
    print(f"  Running SCF for {case_name}...")
    subprocess.run(f"cd {case_dir} && OMP_NUM_THREADS=1 mpirun -np 4 {BINARY} > scf.log 2>&1",
                   shell=True, check=True, timeout=600)
    
    # Write NSCF INPUT and run
    with open(os.path.join(case_dir, "INPUT"), "w") as f:
        f.write(INPUT_NSCF.format(pp_dir=PP_DIR, orb_dir=ORB_DIR))
    
    print(f"  Running NSCF for {case_name}...")
    subprocess.run(f"cd {case_dir} && OMP_NUM_THREADS=1 mpirun -np 4 {BINARY} > nscf.log 2>&1",
                   shell=True, check=True, timeout=600)
    
    # Extract results
    berry_pz = extract_berry_pz(os.path.join(case_dir, "OUT.autotest", "running_nscf.log"))
    deltap_pz = extract_deltap_pz(os.path.join(case_dir, "OUT.autotest", "deltap_results.dat"))
    deltap_per = extract_deltap_per_atom(os.path.join(case_dir, "OUT.autotest", "deltap_results.dat"))
    
    return berry_pz, deltap_pz, deltap_per

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Run equilibrium
    print("=== Equilibrium structure ===")
    berry_eq, deltap_eq, peratom_eq = run_case("equilibrium", ATOMS)
    print(f"  berry Pz = {berry_eq:.6e} e/bohr^2")
    print(f"  deltap Pz = {deltap_eq:.6e} e/bohr^2")
    
    # Run displaced structures
    results = {}
    for iat in range(5):
        elem = ATOMS[iat][0]
        for sign, label in [(+1, "plus"), (-1, "minus")]:
            case_name = f"atom{iat}_{elem}_{label}"
            atoms_mod = [list(a) for a in ATOMS]
            atoms_mod[iat][4] += sign * DTAU_DIRECT  # displace z
            # Wrap to [0,1)
            atoms_mod[iat][4] = atoms_mod[iat][4] % 1.0
            
            print(f"\n=== Displace {elem} (atom {iat}) z by {sign*DTAU_BOHR} Bohr ===")
            berry, deltap, peratom = run_case(case_name, atoms_mod)
            results[(iat, sign)] = (berry, deltap, peratom)
            print(f"  berry Pz = {berry:.6e}, deltap Pz = {deltap:.6e}")
    
    # Compute Born charges
    print("\n" + "=" * 70)
    print("BORN EFFECTIVE CHARGES Z*_{I,zz}")
    print(f"  dtau = {DTAU_BOHR} Bohr, Omega = {CELL_VOLUME:.2f} Bohr^3")
    print(f"  Z* = Omega * [P(+dtau) - P(-dtau)] / (2*dtau)")
    print("=" * 70)
    
    print(f"\n{'Atom':>6} {'Z*_berry':>12} {'Z*_deltap':>12} {'Ratio':>8} {'Literature':>12}")
    print("-" * 55)
    
    # Literature values
    lit = {"Ba": 2.74, "Ti": 7.18, "O": None}  # approximate, varies by O site
    
    for iat in range(5):
        elem = ATOMS[iat][0]
        berry_plus = results[(iat, +1)][0]
        berry_minus = results[(iat, -1)][0]
        deltap_plus = results[(iat, +1)][1]
        deltap_minus = results[(iat, -1)][1]
        
        z_berry = CELL_VOLUME * (berry_plus - berry_minus) / (2 * DTAU_BOHR)
        z_deltap = CELL_VOLUME * (deltap_plus - deltap_minus) / (2 * DTAU_BOHR)
        ratio = z_deltap / z_berry if abs(z_berry) > 1e-10 else float('inf')
        
        lit_val = lit.get(elem, None)
        lit_str = f"{lit_val:.2f}" if lit_val else "~-2.2/-5.6"
        
        print(f"{elem+'_'+str(iat):>6} {z_berry:>12.4f} {z_deltap:>12.4f} {ratio:>8.2f} {lit_str:>12}")
    
    # Sum rule
    z_berry_sum = sum(CELL_VOLUME * (results[(iat, +1)][0] - results[(iat, -1)][0]) / (2 * DTAU_BOHR) for iat in range(5))
    z_deltap_sum = sum(CELL_VOLUME * (results[(iat, +1)][1] - results[(iat, -1)][1]) / (2 * DTAU_BOHR) for iat in range(5))
    print(f"\n{'Sum':>6} {z_berry_sum:>12.4f} {z_deltap_sum:>12.4f}")
    print("(Sum should be 0 by charge neutrality)")
    
    # Also compute per-atom Z* from DeltaP (how P^I changes when atom I is displaced)
    print("\n" + "=" * 70)
    print("PER-ATOM Z* FROM DELTAP (change in P^I when atom I is displaced)")
    print("=" * 70)
    print(f"\n{'Atom':>6} {'Z*_self':>12} {'Z*_total':>12} {'Fraction':>10}")
    print("-" * 45)
    
    for iat in range(5):
        elem = ATOMS[iat][0]
        per_plus = results[(iat, +1)][2]
        per_minus = results[(iat, -1)][2]
        per_eq = peratom_eq
        
        if len(per_plus) == 5 and len(per_minus) == 5:
            z_self = CELL_VOLUME * (per_plus[iat] - per_minus[iat]) / (2 * DTAU_BOHR)
            z_total = CELL_VOLUME * (results[(iat, +1)][1] - results[(iat, -1)][1]) / (2 * DTAU_BOHR)
            frac = z_self / z_total if abs(z_total) > 1e-10 else 0
            print(f"{elem+'_'+str(iat):>6} {z_self:>12.4f} {z_total:>12.4f} {frac:>10.2%}")

if __name__ == "__main__":
    main()
