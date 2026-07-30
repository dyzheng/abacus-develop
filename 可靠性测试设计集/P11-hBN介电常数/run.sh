#!/bin/bash
# ============================================================
# P11 h-BN 介电常数 ε∞（固体响应）
# 通道①: DeltaP total ±λ (LCAO, gdir=3 沿 c) → dγ/dλ → α → ε=1+4πα/V
# 通道②: PW berry_phase 位移 FD (B 沿 c ±0.005 Å) → Z* 路径（README §判据）
# 阻塞状态: 依赖 P01 的 F1/F2 换算链，ε 绝对值判定挂起（见判定段 WARNING）
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
A_ANG=2.512                         # h-BN 面内晶格常数 (Å)
C_ANG=6.692                         # h-BN c 轴 (Å)
C_BOHR=$(awk -v c=$C_ANG -v k=$A2B 'BEGIN{printf "%.10f", c*k}')
# 六方原胞体积 V = sin(60°)·a²·c
V_BOHR3=$(awk -v a=$A_ANG -v c=$C_ANG -v k=$A2B 'BEGIN{printf "%.10f", 0.8660254037844386*a*a*c*k*k*k}')
F1_LAM2E=$(awk -v pi=$PI -v c=$C_BOHR 'BEGIN{printf "%.10f", pi/(2.0*c)}')   # λ(Ry)→E(Ha/Bohr): E=−π·λ/(2a)，取绝对值
F2_GAMMA2MU=$(awk -v pi=$PI -v c=$C_BOHR 'BEGIN{printf "%.10f", c/pi/2.0}') # Σγ_raw→μ(e·Bohr): μ=(a/π)·γ/F2_SPIN，F2_SPIN=2（nspin=1 自旋因子，2026-07-30 冒烟实测确认，待备忘录定稿）
DISP_ANG=0.005                      # 通道② B 位移幅度 (Å)
Z_ION_B=3                           # B 离子项形式电荷 (+3e)，待备忘录核实
LAMBDAS="-0.01 -0.005 0.005 0.01"   # 通道① λ 扫描 (Ry)

# ---------------- 通用函数 ----------------
run_abacus() {
    local d="$1"
    ( cd "$d" && rm -rf OUT.* && \
      if [ "$NPROC" -gt 1 ]; then mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1; else "$ABACUS" > run.log 2>&1; fi )
}

get_rawg()      { grep '\[rawG\]' "$1" 2>/dev/null | tail -1 | grep -o 'Σγ_raw=[^ ]*' | cut -d= -f2 || true; }
get_pw_gamma()  { grep '\[DeltaP-PW\]' "$1" 2>/dev/null | tail -1 | grep -o 'γ_total=[^ ]*' | cut -d= -f2 || true; }
scf_converged() { ! grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; }

NPASS=0; NTOTAL=0
judge() { # $1=描述 $2=PASS/FAIL
    NTOTAL=$((NTOTAL+1))
    [ "$2" = "PASS" ] && NPASS=$((NPASS+1))
    printf '%-64s %s\n' "$1" "$2" | tee -a "${RESULTS}"
}
info() { echo "$1" | tee -a "${RESULTS}"; }

sed_input() { # $1=模板 $2=输出 $3=suffix $4=lambda $5=mixing_beta
    sed -e "s/SUFFIX/$3/" -e "s/LAMBDA_INIT/$4/" -e "s/MIXING_BETA/$5/" \
        -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" -e "s|ORBITAL_DIR|${ORBITAL_DIR}|" "$1" > "$2"
}
lam_tag() { echo "$1" | sed 's/-/m/;s/\./p/'; }

echo "===== P11 h-BN 介电常数 =====" | tee -a "${RESULTS}"
info "ABACUS=${ABACUS}  NPROC=${NPROC}"
info "常量: C_BOHR=${C_BOHR}  V_BOHR3=${V_BOHR3}  F1_LAM2E=${F1_LAM2E}  F2_GAMMA2MU=${F2_GAMMA2MU}"
info ""

CONVBAD=0

# ---------------- 通道①: DeltaP ±λ 扫描（两套 k 网格） ----------------
declare -A SLOPE
for grid in 664 996; do
    declare -A GAM=()
    for lam in ${LAMBDAS}; do
        tag=$(lam_tag "$lam")
        # 负 λ 更难收敛，mixing_beta 用 0.3
        mb=0.4; awk -v l="$lam" 'BEGIN{exit !(l<0)}' && mb=0.3
        d="${RUNS}/deltap_k${grid}_lam${tag}"
        mkdir -p "$d"
        sed_input "${CASES}/INPUT_lcao.tmpl" "$d/INPUT" "p11_k${grid}_${tag}" "$lam" "$mb"
        cp "${CASES}/STRU" "${CASES}/KPT_${grid}" "$d/"
        mv "$d/KPT_${grid}" "$d/KPT"
        if run_abacus "$d" && scf_converged "$d"; then
            g=$(get_rawg "$d/run.log")
        else
            g=""
        fi
        if [ -z "$g" ]; then
            info "  [INVALID] k${grid} λ=${lam}: 未收敛或未输出 [rawG]"
            CONVBAD=1
        else
            GAM[$lam]=$g
            info "  k${grid} λ=${lam}: Σγ_raw=${g}"
        fi
    done
    if [ "${#GAM[@]}" -eq 4 ]; then
        s1=$(awk -v gp="${GAM[0.005]}" -v gm="${GAM[-0.005]}" 'BEGIN{printf "%.10f", (gp-gm)/0.01}')
        s2=$(awk -v gp="${GAM[0.01]}"  -v gm="${GAM[-0.01]}"  'BEGIN{printf "%.10f", (gp-gm)/0.02}')
        SLOPE[$grid]=$(awk -v a="$s1" -v b="$s2" 'BEGIN{printf "%.10f", (a+b)/2.0}')
        # 线性巡检（INFO，不进 SUMMARY）：两 λ 尺度斜率一致性
        lin=$(awk -v a="$s1" -v b="$s2" 'BEGIN{d=a-b; if(d<0)d=-d; m=(a+b)/2; if(m<0)m=-m; printf "%.4f", (m>1e-12)?d/m:0}')
        info "  k${grid}: dγ/dλ(±0.005)=${s1}  dγ/dλ(±0.01)=${s2}  相对不一致度=${lin}"
        # α = (F2/F1)·dγ/dλ (Ha 制 a.u.)；ε = 1+4πα/V
        eps=$(awk -v s="${SLOPE[$grid]}" -v f1=$F1_LAM2E -v f2=$F2_GAMMA2MU -v pi=$PI -v v=$V_BOHR3 \
              'BEGIN{alpha=(s<0?-s:s)*f2/f1; printf "%.6f", 1.0+4.0*pi*alpha/v}')
        info "  k${grid}: α(工作值换算) → ε_c=${eps}  [阻塞量，仅供记录]"
    fi
done

# ---------------- 通道②: PW berry_phase 位移 FD（Z* 路径，k664） ----------------
DZ=$(awk -v d=$DISP_ANG -v c=$C_ANG 'BEGIN{printf "%.8f", d/c}')
GAM_PW=""
for sign in p m; do
    d="${RUNS}/pw_disp_B${sign}"
    mkdir -p "$d"
    if [ "$sign" = "p" ]; then sgn=1; else sgn=-1; fi
    awk -v dz="$DZ" -v s="$sgn" '{ if ($1=="0.33333333333333" && $2=="0.66666666666667" && $3=="0.25" && !done) \
        { $3=sprintf("%.8f", $3+s*dz); done=1 } print }' "${CASES}/STRU" > "$d/STRU"
    cp "${CASES}/KPT_664" "$d/KPT"
    sed -e "s/SUFFIX/p11_pw_${sign}/" -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" "${CASES}/INPUT_pw.tmpl" > "$d/INPUT"
    if run_abacus "$d" && scf_converged "$d"; then
        g=$(get_pw_gamma "$d/run.log")
    else
        g=""
    fi
    if [ -z "$g" ]; then
        info "  [INVALID] PW 位移 ${sign}: 未收敛或未输出 [DeltaP-PW]"
        CONVBAD=1
    else
        info "  PW 位移 ${sign} (${DISP_ANG} Å 沿 c): γ_total=${g}"
        [ "$sign" = "p" ] && GAM_PW_P=$g || GAM_PW_M=$g
    fi
done
if [ -n "${GAM_PW_P:-}" ] && [ -n "${GAM_PW_M:-}" ]; then
    du_bohr=$(awk -v d=$DISP_ANG -v k=$A2B 'BEGIN{printf "%.10f", 2.0*d*k}')
    # Z*_el = (V/(2π·c))·dγ/du （电子项，符号约定待备忘录）；Z* = Z_ion + Z*_el
    zstar=$(awk -v gp="$GAM_PW_P" -v gm="$GAM_PW_M" -v du="$du_bohr" -v v=$V_BOHR3 -v c=$C_BOHR -v pi=$PI -v zi=$Z_ION_B \
           'BEGIN{dg=gp-gm; if(dg>3.141592653589793) dg-=2*3.141592653589793; if(dg<-3.141592653589793) dg+=2*3.141592653589793;
                  printf "%.6f", zi + v/(2.0*pi*c)*dg/du}')
    info "  通道② Z*_zz(B) = ${zstar} (含离子项 +${Z_ION_B}e，符号约定待 F1/F2 备忘录)  [阻塞量，仅供记录]"
fi

# ---------------- 判定 ----------------
info ""
info "----- 判定表 -----"

# J1: SCF 收敛巡检（全部算例）
if [ "$CONVBAD" -eq 0 ]; then judge "J1 SCF 收敛巡检（全部 DeltaP/PW 算例）" PASS; else judge "J1 SCF 收敛巡检（存在 INVALID 点，见上）" FAIL; fi

# J2: k 网格收敛 |slope_664 - slope_996| / |slope_996| ≤ 2%（比值判据，不依赖 F1/F2）
if [ -n "${SLOPE[664]:-}" ] && [ -n "${SLOPE[996]:-}" ]; then
    rel=$(awk -v a="${SLOPE[664]}" -v b="${SLOPE[996]}" 'BEGIN{d=a-b; if(d<0)d=-d; m=b; if(m<0)m=-m;
        if(m<1e-12) printf "%.6f", (d<1e-12)?0:999; else printf "%.6f", d/m}')
    if awk -v r="$rel" 'BEGIN{exit !(r<=0.02)}'; then
        judge "J2 k 网格收敛 664 vs 996: 相对差=${rel} ≤ 2%" PASS
    else
        judge "J2 k 网格收敛 664 vs 996: 相对差=${rel} > 2%" FAIL
    fi
else
    judge "J2 k 网格收敛（斜率缺失，无法判定）" FAIL
fi

# J3: 两通道 ε_c 差 ≤5% —— 阻塞于 P01（F1/F2 换算链未定稿）
info "J3 两通道 ε_c 差 ≤5%: BLOCKED（通道① ε 见上；通道② 走 Z* 路径，ε 反演关系待备忘录）"
info "J4 与文献区间（面外 ε∞≈3）对比: BLOCKED"
info "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）"

# 各向异性标注（仅 gdir=3 面外分量；面内预期更大，文献 ~4.5 vs ~3）
info "INFO 各向异性标注: 本测试仅测面外分量 ε_c；按文献面内(~4.5) > 面外(~3)，若结果反向需排查 HK 修正项"

info ""
info "SUMMARY: ${NPASS}/${NTOTAL} PASS"
[ "$NPASS" -eq "$NTOTAL" ]
