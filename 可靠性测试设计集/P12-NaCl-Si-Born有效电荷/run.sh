#!/bin/bash
# ============================================================
# P12 NaCl / Si Born 有效电荷（固体响应）
# 通道①: DeltaP total ±λ (LCAO/genelpa, cal_force 1) → dF_z/dλ → Z*
# 通道②: PW berry_phase 位移 FD（阳离子沿 z ±0.005 Å）→ Z*
# 阻塞状态: 通道① 依赖 P01 的 F1 换算（λ→E），相关判定挂起（见判定段 WARNING）
# ============================================================
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASES="${BASEDIR}/cases"
RUNS="${BASEDIR}/runs"
mkdir -p "${RUNS}"
RESULTS="${RUNS}/results.txt"
: > "${RESULTS}"

# ---------------- 换算常量区（F1/F2 备忘录定稿前工作值，单点修改） ----------------
PI=3.141592653589793
A2B=1.8897261254578284              # Å → Bohr
# fcc 初基胞 gdir=3 方向有效盒长 L3 = a/√3（待备忘录固体专项核实）
# 初基胞体积 V = a³/4
FORCE_CONV=0.0194469038             # eV/Å → Ha/Bohr
DISP_ANG=0.005                      # 通道② 阳离子位移幅度 (Å)
Z_ION_NA=1                          # Na 离子项（赝势价态/形式电荷，待备忘录核实）
Z_ION_SI=4                          # Si 离子项
LAMBDAS="-0.01 -0.005 0.005 0.01"

# ---------------- 通用函数 ----------------
run_abacus() {
    local d="$1"
    ( cd "$d" && rm -rf OUT.* && \
      if [ "$NPROC" -gt 1 ]; then mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1; else "$ABACUS" > run.log 2>&1; fi )
}
get_pw_gamma()  { grep '\[DeltaP-PW\]' "$1" 2>/dev/null | tail -1 | grep -o 'γ_total=[^ ]*' | cut -d= -f2 || true; }
scf_converged() { ! grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; }
scflog()        { ls "$1"/OUT.*/running_scf.log 2>/dev/null | head -1 || true; }
get_fz() { # $1=running_scf.log $2=原子序号(1起) → 该原子 TOTAL-FORCE z 分量
    awk -v ord="$2" '
        /TOTAL-FORCE/ { inblk=1; n=0; val=""; next }
        inblk && NF>=4 && $1 ~ /^[A-Z][a-z]?[0-9]+$/ && $2 ~ /^-?[0-9.]/ {
            n++; if (n==ord) val=$4
        }
        END { if (val!="") print val; else exit 1 }
    ' "$1" 2>/dev/null || true
}

NPASS=0; NTOTAL=0
judge() {
    NTOTAL=$((NTOTAL+1))
    [ "$2" = "PASS" ] && NPASS=$((NPASS+1))
    printf '%-64s %s\n' "$1" "$2" | tee -a "${RESULTS}"
}
info() { echo "$1" | tee -a "${RESULTS}"; }
lam_tag() { echo "$1" | sed 's/-/m/;s/\./p/'; }

echo "===== P12 NaCl / Si Born 有效电荷 =====" | tee -a "${RESULTS}"
info "ABACUS=${ABACUS}  NPROC=${NPROC}"
info ""

CONVBAD=0

for SYS in NaCl Si; do
    case "$SYS" in
        NaCl) A_ANG=5.64;  Z_ION=$Z_ION_NA ;;
        Si)   A_ANG=5.431; Z_ION=$Z_ION_SI ;;
    esac
    L3_BOHR=$(awk -v a=$A_ANG -v k=$A2B 'BEGIN{printf "%.10f", a*k/1.7320508075688772}')
    V_BOHR3=$(awk -v a=$A_ANG -v k=$A2B 'BEGIN{printf "%.10f", a*a*a*k*k*k/4.0}')
    F1_LAM2E=$(awk -v pi=$PI -v l=$L3_BOHR 'BEGIN{printf "%.10f", pi/(2.0*l)}')
    info "----- 体系 ${SYS} (a=${A_ANG} Å, L3=${L3_BOHR} Bohr, V=${V_BOHR3} Bohr³) -----"

    # ---- 通道①: DeltaP ±λ，cal_force 1，dF_z/dλ → Z* ----
    declare -A FZ1=() FZ2=()
    for lam in ${LAMBDAS}; do
        tag=$(lam_tag "$lam")
        mb=0.4; awk -v l="$lam" 'BEGIN{exit !(l<0)}' && mb=0.3
        d="${RUNS}/${SYS}_deltap_lam${tag}"
        mkdir -p "$d"
        sed -e "s/SUFFIX/p12_${SYS}_${tag}/" -e "s/LAMBDA_INIT/$lam/" -e "s/MIXING_BETA/$mb/" \
            -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" -e "s|ORBITAL_DIR|${ORBITAL_DIR}|" \
            "${CASES}/INPUT_lcao.tmpl" > "$d/INPUT"
        cp "${CASES}/STRU_${SYS}" "$d/STRU"
        cp "${CASES}/KPT" "$d/KPT"
        f1=""; f2=""
        if run_abacus "$d" && scf_converged "$d"; then
            sl=$(scflog "$d")
            [ -n "$sl" ] && f1=$(get_fz "$sl" 1) && f2=$(get_fz "$sl" 2)
        fi
        if [ -z "$f1" ] || [ -z "$f2" ]; then
            info "  [INVALID] ${SYS} λ=${lam}: 未收敛或未取到 TOTAL-FORCE"
            CONVBAD=1
        else
            FZ1[$lam]=$f1; FZ2[$lam]=$f2
            info "  ${SYS} λ=${lam}: F_z(阳离子)=${f1}  F_z(原子2)=${f2} (eV/Å)"
        fi
    done
    if [ "${#FZ1[@]}" -eq 4 ]; then
        for pair in "0.005 0.01" "0.01 0.02"; do
            set -- $pair; mag=$1; span=$2
            zs1=$(awk -v fp="${FZ1[$mag]}" -v fm="${FZ1[-$mag]}" -v s="$span" -v fc=$FORCE_CONV -v f1=$F1_LAM2E \
                  'BEGIN{printf "%.6f", ((fp-fm)/s)*fc/(-f1)}')
            zs2=$(awk -v fp="${FZ2[$mag]}" -v fm="${FZ2[-$mag]}" -v s="$span" -v fc=$FORCE_CONV -v f1=$F1_LAM2E \
                  'BEGIN{printf "%.6f", ((fp-fm)/s)*fc/(-f1)}')
            info "  通道① ±${mag}: Z*(阳离子)=${zs1}  Z*(原子2)=${zs2}  [阻塞量，仅供记录]"
        done
    fi

    # ---- 通道②: PW berry_phase，阳离子沿 z 位移 ±0.005 Å ----
    DF=$(awk -v d=$DISP_ANG -v a=$A_ANG 'BEGIN{printf "%.8f", d/a}')
    GP=""; GM=""
    for sign in p m; do
        d="${RUNS}/${SYS}_pw_disp${sign}"
        mkdir -p "$d"
        if [ "$sign" = "p" ]; then sgn=1; else sgn=-1; fi
        # 笛卡尔 z 位移 δ 对应分数位移 (δ/a, δ/a, −δ/a)（fcc 初基矢下保持 Δx=Δy=0）
        awk -v df="$DF" -v s="$sgn" '{ if ($1=="0.0" && $2=="0.0" && $3=="0.0" && NF==3 && !done) \
            { printf "%.8f %.8f %.8f\n", s*df, s*df, -s*df; done=1; next } print }' \
            "${CASES}/STRU_${SYS}" > "$d/STRU"
        cp "${CASES}/KPT" "$d/KPT"
        sed -e "s/SUFFIX/p12_${SYS}_pw_${sign}/" -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" \
            "${CASES}/INPUT_pw.tmpl" > "$d/INPUT"
        g=""
        if run_abacus "$d" && scf_converged "$d"; then g=$(get_pw_gamma "$d/run.log"); fi
        if [ -z "$g" ]; then
            info "  [INVALID] ${SYS} PW 位移 ${sign}: 未收敛或未输出 [DeltaP-PW]"
            CONVBAD=1
        else
            info "  ${SYS} PW 位移 ${sign}: γ_total=${g}"
            [ "$sign" = "p" ] && GP=$g || GM=$g
        fi
    done
    ZSTAR_PW=""
    if [ -n "$GP" ] && [ -n "$GM" ]; then
        du_bohr=$(awk -v d=$DISP_ANG -v k=$A2B 'BEGIN{printf "%.10f", 2.0*d*k}')
        ZSTAR_PW=$(awk -v gp="$GP" -v gm="$GM" -v du="$du_bohr" -v v=$V_BOHR3 -v l=$L3_BOHR -v pi=$PI -v zi=$Z_ION \
            'BEGIN{dg=gp-gm; if(dg>3.141592653589793) dg-=2*3.141592653589793; if(dg<-3.141592653589793) dg+=2*3.141592653589793;
                   printf "%.6f", zi + v/(2.0*pi*l)*dg/du}')
        info "  通道② Z*_zz(阳离子, ${SYS}) = ${ZSTAR_PW} (含离子项 +${Z_ION}e，符号约定待备忘录)"
        eval "ZSTAR_PW_${SYS}=${ZSTAR_PW}"
    fi
    info ""
done

# ---------------- 判定 ----------------
info "----- 判定表 -----"

if [ "$CONVBAD" -eq 0 ]; then judge "J1 SCF 收敛巡检（两体系全部算例）" PASS; else judge "J1 SCF 收敛巡检（存在 INVALID 点）" FAIL; fi

# J2: Si |Z*_zz| < 0.1（通道②，不依赖 F1）
if [ -n "${ZSTAR_PW_Si:-}" ]; then
    if awk -v z="${ZSTAR_PW_Si}" 'BEGIN{z=(z<0?-z:z); exit !(z<0.1)}'; then
        judge "J2 Si |Z*|=${ZSTAR_PW_Si} < 0.1（通道②）" PASS
    else
        judge "J2 Si |Z*|=${ZSTAR_PW_Si} ≥ 0.1（通道②）" FAIL
    fi
else
    judge "J2 Si |Z*|<0.1（通道②数值缺失）" FAIL
fi

# NaCl 物理区间（软判，越界 WARNING 不进 SUMMARY）
if [ -n "${ZSTAR_PW_NaCl:-}" ]; then
    if awk -v z="${ZSTAR_PW_NaCl}" 'BEGIN{z=(z<0?-z:z); exit !(z>=0.9 && z<=1.2)}'; then
        info "OK: NaCl |Z*_Na|=${ZSTAR_PW_NaCl} ∈ [0.9,1.2]（通道②）"
    else
        info "WARNING: NaCl |Z*_Na|=${ZSTAR_PW_NaCl} 越出 [0.9,1.2]（软判，请检查分支选择稳定性）"
    fi
fi

# J3/J4: 依赖 F1（λ→E）换算，阻塞
info "J3 两通道 Z* 互差 ≤0.05: BLOCKED（通道① dF/dλ→Z* 需 F1 换算）"
info "J4 |ΣZ*| ≤ 0.05（中和检查，通道① 双原子）: BLOCKED（同上）"
info "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）"

info ""
info "SUMMARY: ${NPASS}/${NTOTAL} PASS"
[ "$NPASS" -eq "$NTOTAL" ]
