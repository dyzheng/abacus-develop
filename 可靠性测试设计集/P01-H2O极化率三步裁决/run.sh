#!/bin/bash
# P01 H2O 极化率三步裁决（旗舰测试，F1/F2 阻塞标注）
# 阶段1（裁判）：efield 有限差分 -> alpha_ref
# 阶段2（被裁）：DeltaP ±lambda 扫描 -> dgamma/dlambda
# 阶段3 裁决：经 F1/F2 换算链得 alpha_DeltaP，与 alpha_ref 对比
# 用法：./run.sh  （环境变量 ABACUS/PSEUDO_DIR/ORBITAL_DIR/NPROC 可覆盖）
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASES="${BASEDIR}/cases/h2o"
RUNS="${BASEDIR}/runs"
RESULTS="${RUNS}/results.txt"

# ============ 常量区（F1/F2 备忘录定稿前工作值，定稿后只需改这里） ============
A_BOHR=30.0                     # 盒边长 (Bohr)
PI=$(awk 'BEGIN{print atan2(0,-1)}')
# F1 工作值: E_field(Ha/Bohr) = -PI * lambda / (2 * A_BOHR)，lambda 单位 Ry
# F2 工作值: alpha_DeltaP(a.u.) = -(A_BOHR/PI) * dgamma/dE_field / F2_SPIN
# F2_SPIN=2 为 nspin=1 自旋因子（2026-07-30 冒烟实测 H2O: Σγ_raw=-12.718 rad →
# unwrap -0.1517 → /2 → μ=1.841 D vs 实验 1.855 D，吻合；仍待备忘录定稿）
F2_SPIN=2.0
# 阶段1 能量 FD: alpha_ref = 2*[E(+d)+E(-d)-2E(0)]/d^2（E 单位 Ry, d 单位 Ha,
# 系数 2 为 Ry->Ha；符号依赖 dip_cor 能量记账约定，判据用绝对值）
# ===========================================================================

EF_AMPS="-0.001 -0.0005 0.0 0.0005 0.001"
LAMS="-0.08 -0.02 0.0 0.02 0.08"

tag() { echo "$1" | sed 's/-/m/;s/\./p/'; }

run_abacus() {
    # $1 = 算例目录；运行失败不中断（提取阶段判 INVALID/FAIL）
    cd "$1"
    rm -rf OUT.*
    if [ "$NPROC" -gt 1 ]; then
        mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1 || true
    else
        "$ABACUS" > run.log 2>&1 || true
    fi
    cd "$BASEDIR"
}

get_eks() { grep "E_KohnSham" "$1"/OUT.*/running_scf.log 2>/dev/null | tail -1 | awk '{print $2}'; }
get_rawg() { grep "rawG" "$1"/run.log 2>/dev/null | tail -1 | sed -E 's/.*Σγ_raw=([^ ]+).*/\1/'; }

echo "======================================================"
echo " P01 H2O 极化率三步裁决"
echo " ABACUS=$ABACUS  NPROC=$NPROC"
echo "======================================================"
mkdir -p "$RUNS"
: > "$RESULTS"

# ---------------- 阶段1：裁判（efield FD） ----------------
echo "===== 阶段1: efield 有限差分 (efield_flag=1, dip_cor_flag=1, efield_dir=3) =====" | tee -a "$RESULTS"
for amp in $EF_AMPS; do
    d="${RUNS}/efield/ef_$(tag "$amp")"
    mkdir -p "$d"
    cp "$CASES/STRU" "$CASES/KPT" "$d/"
    cat > "$d/INPUT" <<EOF
INPUT_PARAMETERS
suffix          autotest
calculation     scf
basis_type      lcao
ecutwfc         100
gamma_only      0
nspin           1
scf_thr         1.0e-8
scf_nmax        200
smearing_method gauss
smearing_sigma  0.002
mixing_type     broyden
mixing_beta     0.4
ks_solver       genelpa
symmetry        0
efield_flag     1
dip_cor_flag    1
efield_dir      3
efield_amp      $amp
pseudo_dir  $PSEUDO_DIR
orbital_dir $ORBITAL_DIR
EOF
    echo "  [run] efield_amp=$amp -> $d"
    run_abacus "$d"
done

# 提取 E_KohnSham (Ry)
declare -A EE
for amp in $EF_AMPS; do
    d="${RUNS}/efield/ef_$(tag "$amp")"
    EE[$amp]=$(get_eks "$d")
    echo "  E(amp=$amp) = ${EE[$amp]:-MISSING} Ry" | tee -a "$RESULTS"
done

# alpha_ref 两窗口: delta=0.0005 与 0.001
AL05=$(awk -v ep="${EE[0.0005]}" -v em="${EE[-0.0005]}" -v e0="${EE[0.0]}" \
    'BEGIN{d=0.0005; printf "%.8f", 2*(ep+em-2*e0)/(d*d)}')
AL10=$(awk -v ep="${EE[0.001]}" -v em="${EE[-0.001]}" -v e0="${EE[0.0]}" \
    'BEGIN{d=0.001; printf "%.8f", 2*(ep+em-2*e0)/(d*d)}')
echo "  alpha_ref(delta=0.0005) = $AL05 a.u." | tee -a "$RESULTS"
echo "  alpha_ref(delta=0.001 ) = $AL10 a.u." | tee -a "$RESULTS"
# 若 dip_cor 记账符号导致负值，取绝对值进入判据（见 README 已知风险）
ALPHA_REF=$(awk -v a="$AL05" -v b="$AL10" 'BEGIN{a=(a<0)?-a:a; b=(b<0)?-b:b; printf "%.8f",(a+b)/2}')

# 四点线性 R^2: E 对 x=delta^2 做最小二乘（±0.0005/±0.001 四点）
R2_EF=$(awk -v e1="${EE[-0.001]}" -v e2="${EE[-0.0005]}" -v e3="${EE[0.0005]}" -v e4="${EE[0.001]}" 'BEGIN{
    n=4; x[1]=1e-6; x[2]=2.5e-7; x[3]=2.5e-7; x[4]=1e-6;
    y[1]=e1; y[2]=e2; y[3]=e3; y[4]=e4;
    for(i=1;i<=n;i++){sx+=x[i];sy+=y[i];sxx+=x[i]*x[i];sxy+=x[i]*y[i];syy+=y[i]*y[i]}
    r=(n*sxy-sx*sy)/sqrt((n*sxx-sx*sx)*(n*syy-sy*sy)); printf "%.8f", r*r}')

# ---------------- 阶段2：被裁（DeltaP ±lambda） ----------------
echo "===== 阶段2: DeltaP total 约束 lambda 扫描（测量模式 lambda_step=0） =====" | tee -a "$RESULTS"
for lam in $LAMS; do
    d="${RUNS}/deltap/lam_$(tag "$lam")"
    mkdir -p "$d"
    cp "$CASES/STRU" "$CASES/KPT" "$d/"
    # 负 lambda 更难收敛（R1），mixing_beta 降到 0.3
    MB=$(awk -v l="$lam" 'BEGIN{print (l<0)?"0.3":"0.4"}')
    cat > "$d/INPUT" <<EOF
INPUT_PARAMETERS
suffix          autotest
calculation     scf
basis_type      lcao
ecutwfc         100
gamma_only      0
nspin           1
scf_thr         1.0e-8
scf_nmax        200
smearing_method gauss
smearing_sigma  0.002
mixing_type     broyden
mixing_beta     $MB
ks_solver       genelpa
symmetry        0
deltap_switch   1
deltap_corr     1
deltap_gdir     3
deltap_rm       6.0
onsite_radius   6.0
deltap_lambda_init   $lam
deltap_lambda_step   0.0
deltap_lambda_mixing 0.1
deltap_inner_thr     1.0e-2
deltap_constraint_mode total
pseudo_dir  $PSEUDO_DIR
orbital_dir $ORBITAL_DIR
EOF
    echo "  [run] lambda=$lam (mixing_beta=$MB) -> $d"
    run_abacus "$d"
done

# 提取 Σγ_raw
declare -A GG
for lam in $LAMS; do
    d="${RUNS}/deltap/lam_$(tag "$lam")"
    GG[$lam]=$(get_rawg "$d")
    echo "  gamma_raw(lambda=$lam) = ${GG[$lam]:-MISSING} rad" | tee -a "$RESULTS"
done

# dgamma/dlambda 两窗口
SL02=$(awk -v gp="${GG[0.02]}" -v gm="${GG[-0.02]}" 'BEGIN{printf "%.8f",(gp-gm)/0.04}')
SL08=$(awk -v gp="${GG[0.08]}" -v gm="${GG[-0.08]}" 'BEGIN{printf "%.8f",(gp-gm)/0.16}')
DGDL=$(awk -v a="$SL02" -v b="$SL08" 'BEGIN{printf "%.8f",(a+b)/2}')
echo "  dgamma/dlambda: window ±0.02 -> $SL02, window ±0.08 -> $SL08, 均值 $DGDL" | tee -a "$RESULTS"

# ---------------- 阶段3：裁决 ----------------
echo "===== 阶段3: 裁决 =====" | tee -a "$RESULTS"
echo "WARNING: F1/F2 换算链待备忘录定稿（阻塞项）" | tee -a "$RESULTS"
# F1: dE/dlambda = -PI/(2*A_BOHR)  ->  dgamma/dE = dgamma/dlambda / (dE/dlambda)
DEDL=$(awk -v pi="$PI" -v a="$A_BOHR" 'BEGIN{printf "%.10f", -pi/(2*a)}')
DGDE=$(awk -v s="$DGDL" -v d="$DEDL" 'BEGIN{printf "%.10f", s/d}')
# F2: alpha_DeltaP = -(A_BOHR/PI) * dgamma/dE / F2_SPIN
ALPHA_DP=$(awk -v a="$A_BOHR" -v pi="$PI" -v s="$DGDE" -v f="$F2_SPIN" 'BEGIN{printf "%.8f", -(a/pi)*s/f}')
ALPHA_DP_ABS=$(awk -v a="$ALPHA_DP" 'BEGIN{printf "%.8f",(a<0)?-a:a}')
echo "  alpha_ref     = $ALPHA_REF a.u. (两窗口均值, |AL05|=$AL05, |AL10|=$AL10)" | tee -a "$RESULTS"
echo "  alpha_DeltaP  = $ALPHA_DP a.u. (|.|=$ALPHA_DP_ABS)" | tee -a "$RESULTS"
REL=$(awk -v a="$ALPHA_DP_ABS" -v b="$ALPHA_REF" 'BEGIN{printf "%.6f", (b>0)?((a-b<0)?(b-a)/b:(a-b)/b):999}')
echo "  相对偏差 |alpha_DeltaP-alpha_ref|/alpha_ref = $REL" | tee -a "$RESULTS"

# 反对称偏差（±0.02 与 ±0.08）
AS02=$(awk -v gp="${GG[0.02]}" -v gm="${GG[-0.02]}" 'BEGIN{d=gp-gm; if(d<0)d=-d; s=gp+gm; if(s<0)s=-s; printf "%.6f",(d>1e-12)?s/d:0}')
AS08=$(awk -v gp="${GG[0.08]}" -v gm="${GG[-0.08]}" 'BEGIN{d=gp-gm; if(d<0)d=-d; s=gp+gm; if(s<0)s=-s; printf "%.6f",(d>1e-12)?s/d:0}')
echo "  反对称偏差: ±0.02 -> $AS02, ±0.08 -> $AS08" | tee -a "$RESULTS"
echo "  efield 四点线性 R^2 = $R2_EF" | tee -a "$RESULTS"

NP=0; NT=0
judge() { # $1 描述  $2 awk 条件结果(1/0)
    NT=$((NT+1))
    if [ "$2" = "1" ]; then echo "PASS: $1" | tee -a "$RESULTS"; NP=$((NP+1))
    else echo "FAIL: $1" | tee -a "$RESULTS"; fi
}
OK1=$(awk -v r="$REL" 'BEGIN{print (r<=0.10)?1:0}')
OK2=$(awk -v a="$AS02" -v b="$AS08" 'BEGIN{m=(a>b)?a:b; print (m<0.05)?1:0}')
OK3=$(awk -v r="$R2_EF" 'BEGIN{print (r>=0.99)?1:0}')
judge "|Delta_alpha|/alpha_ref <= 10% ($REL)" "$OK1"
judge "反对称偏差 < 5% (max=$AS02,$AS08)" "$OK2"
judge "efield 四点线性 R^2 >= 0.99 ($R2_EF)" "$OK3"
INRANGE=$(awk -v a="$ALPHA_REF" 'BEGIN{print (a>=7 && a<=10)?1:0}')
if [ "$INRANGE" != "1" ]; then
    echo "WARNING: alpha_ref=$ALPHA_REF 越出 [7,10] a.u. 区间（盒子/基组充分性待查，仅警告不计 FAIL）" | tee -a "$RESULTS"
fi
echo "SUMMARY: $NP/$NT PASS" | tee -a "$RESULTS"
[ "$NP" = "$NT" ] && exit 0 || exit 1
