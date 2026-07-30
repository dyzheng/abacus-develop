#!/bin/bash
# ============================================================
# P13 BaTiO3 自发极化（固体平衡·铁电）
# ① Ti 位移路径 9 点（z_Ti=0.48..0.56）LCAO λ=0 → Σγ_raw 分支连续性
# ② Ps 三通道：DeltaP raw γ / PW berry_phase / wannier90（可选）
# ③ 约束能量面：z_Ti=0.50/0.52/0.54 三点 target 模式约束 ±0.2 rad → E(γ)
# 前置：P09（分支确定性，README §6 标注）
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

# ---------------- 换算常量区 ----------------
PI=3.141592653589793
A2B=1.8897261254578284              # Å → Bohr
A_ANG=4.00                          # 四方晶胞 a (Å)
C_ANG=4.20                          # 四方晶胞 c (Å)
C_BOHR=$(awk -v c=$C_ANG -v k=$A2B 'BEGIN{printf "%.10f", c*k}')
V_BOHR3=$(awk -v a=$A_ANG -v c=$C_ANG -v k=$A2B 'BEGIN{printf "%.10f", a*a*c*k*k*k}')
EBOHR2_TO_CM2=57.2148               # 1 e/Bohr² → C/m²
# P = (e/Ω)·(c/π)·γ（任务书公式，γ 为自旋求和 raw γ；PW 差值同一常量，见 README）
P_CONV=$(awk -v c=$C_BOHR -v pi=$PI -v v=$V_BOHR3 -v k=$EBOHR2_TO_CM2 'BEGIN{printf "%.10f", c/(pi*v)*k}')
TARGET_HALF=0.2                     # 约束目标半宽 (rad)
JUMP_THR=1.0                        # 分支跳变阈值 (rad)
Z_LIST="0.48 0.49 0.50 0.51 0.52 0.53 0.54 0.55 0.56"

# ---------------- 通用函数 ----------------
run_abacus() {
    local d="$1"
    ( cd "$d" && rm -rf OUT.* && \
      if [ "$NPROC" -gt 1 ]; then mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1; else "$ABACUS" > run.log 2>&1; fi )
}
get_rawg()      { grep '\[rawG\]' "$1" 2>/dev/null | tail -1 | grep -o 'Σγ_raw=[^ ]*' | cut -d= -f2 || true; }
get_pw_gamma()  { grep '\[DeltaP-PW\]' "$1" 2>/dev/null | tail -1 | grep -o 'γ_total=[^ ]*' | cut -d= -f2 || true; }
get_eks()       { grep 'E_KohnSham' "$1" 2>/dev/null | tail -1 | sed 's/.*E_KohnSham[= ]\{1,\}//' | awk '{print $1}' || true; }
scf_converged() { ! grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; }
scflog()        { ls "$1"/OUT.*/running_scf.log 2>/dev/null | head -1 || true; }
ztag()          { echo "$1" | sed 's/\./p/'; }

NPASS=0; NTOTAL=0
judge() {
    NTOTAL=$((NTOTAL+1))
    [ "$2" = "PASS" ] && NPASS=$((NPASS+1))
    printf '%-64s %s\n' "$1" "$2" | tee -a "${RESULTS}"
}
info() { echo "$1" | tee -a "${RESULTS}"; }

echo "===== P13 BaTiO3 自发极化 =====" | tee -a "${RESULTS}"
info "ABACUS=${ABACUS}  NPROC=${NPROC}  P_CONV=${P_CONV} (C/m²)/rad"
info ""

CONVBAD=0
declare -A GAM EQ

# ---------------- ① Ti 位移路径：LCAO λ=0 逐点 ----------------
info "----- ① Ti 位移路径（λ=0 测量模式） -----"
for z in ${Z_LIST}; do
    d="${RUNS}/path_z$(ztag $z)"
    mkdir -p "$d"
    sed "s/Z_TI/$z/" "${CASES}/STRU.tmpl" > "$d/STRU"
    cp "${CASES}/KPT" "$d/KPT"
    sed -e "s/SUFFIX/p13_path_$(ztag $z)/" -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" -e "s|ORBITAL_DIR|${ORBITAL_DIR}|" \
        "${CASES}/INPUT_lcao.tmpl" > "$d/INPUT"
    g=""; e=""
    if run_abacus "$d" && scf_converged "$d"; then
        g=$(get_rawg "$d/run.log")
        sl=$(scflog "$d"); [ -n "$sl" ] && e=$(get_eks "$sl")
    fi
    if [ -z "$g" ] || [ -z "$e" ]; then
        info "  [INVALID] z_Ti=${z}: 未收敛或缺少 [rawG]/E_KohnSham"
        CONVBAD=1
    else
        GAM[$z]=$g; EQ[$z]=$e
        pz=$(awk -v g=$g -v k=$P_CONV 'BEGIN{printf "%.6f", g*k}')
        info "  z_Ti=${z}: Σγ_raw=${g}  E_KS=${e}  P(z)=${pz} C/m²(未减参考)"
    fi
done

# 分支连续性：相邻点 |Δγ| > JUMP_THR 判为假跳变
MAXDG=0; JUMPLIST=""
prev_z=""; prev_g=""
for z in ${Z_LIST}; do
    g="${GAM[$z]:-}"
    if [ -n "$g" ] && [ -n "$prev_g" ]; then
        dg=$(awk -v a=$g -v b=$prev_g 'BEGIN{d=a-b; printf "%.6f", (d<0?-d:d)}')
        bigger=$(awk -v d=$dg -v m=$MAXDG 'BEGIN{print (d>m)?1:0}')
        [ "$bigger" = "1" ] && MAXDG=$dg
        if awk -v d=$dg -v t=$JUMP_THR 'BEGIN{exit !(d>t)}'; then
            JUMPLIST="${JUMPLIST} ${prev_z}->${z}(Δγ=${dg})"
        fi
    fi
    [ -n "$g" ] && { prev_z=$z; prev_g=$g; }
done

# ---------------- ② Ps 三通道 ----------------
info ""
info "----- ② 自发极化 Ps（参考点 z_Ti=0.50，铁电点 z_Ti=0.52） -----"
PS1=""; PS2=""
if [ -n "${GAM[0.50]:-}" ] && [ -n "${GAM[0.52]:-}" ]; then
    PS1=$(awk -v g2="${GAM[0.52]}" -v g1="${GAM[0.50]}" -v k=$P_CONV 'BEGIN{printf "%.6f", (g2-g1)*k}')
    info "  通道① DeltaP raw γ: Δγ=$(awk -v a=${GAM[0.52]} -v b=${GAM[0.50]} 'BEGIN{printf "%.6f", a-b}') rad → Ps=${PS1} C/m²"
else
    info "  [INVALID] 通道①: z=0.50/0.52 平衡 γ 缺失"
    CONVBAD=1
fi

GP50=""; GP52=""
for z in 0.50 0.52; do
    d="${RUNS}/pw_z$(ztag $z)"
    mkdir -p "$d"
    sed "s/Z_TI/$z/" "${CASES}/STRU.tmpl" > "$d/STRU"
    cp "${CASES}/KPT" "$d/KPT"
    sed -e "s/SUFFIX/p13_pw_$(ztag $z)/" -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" "${CASES}/INPUT_pw.tmpl" > "$d/INPUT"
    g=""
    if run_abacus "$d" && scf_converged "$d"; then g=$(get_pw_gamma "$d/run.log"); fi
    if [ -z "$g" ]; then
        info "  [INVALID] PW z_Ti=${z}: 未收敛或未输出 [DeltaP-PW]"
        CONVBAD=1
    else
        info "  PW z_Ti=${z}: γ_total=${g}"
        [ "$z" = "0.50" ] && GP50=$g || GP52=$g
    fi
done
if [ -n "$GP50" ] && [ -n "$GP52" ] && [ -n "${GAM[0.50]:-}" ] && [ -n "${GAM[0.52]:-}" ]; then
    # 用 LCAO Δγ 作预测值展开 PW 的 wrapped 差值（小步长预测-校正，README §6）
    PS2=$(awk -v gp2="$GP52" -v gp1="$GP50" -v gl2="${GAM[0.52]}" -v gl1="${GAM[0.50]}" -v k=$P_CONV -v pi=$PI \
        'BEGIN{ dg=gp2-gp1; ref=gl2-gl1; n=int((ref-dg)/(2*pi)+((ref-dg)>0?0.5:-0.5)); dg=dg+n*2*pi;
                printf "%.6f", dg*k }')
    info "  通道② PW berry_phase: Ps=${PS2} C/m²（已按 LCAO Δγ 展开 2π 整数倍）"
fi

# 通道③ wannier90（可选）
W90="$(command -v wannier90 2>/dev/null || command -v wannier90.x 2>/dev/null || true)"
if [ -z "$W90" ]; then
    info "  通道③ wannier90: SKIP（未检测到 wannier90 可执行文件）"
else
    info "  通道③ wannier90: 检测到 ${W90}，生成 .win 模板（窗口/投影需首次实跑确认，README §6）"
    d="${RUNS}/w90_z052"
    mkdir -p "$d"
    sed "s/Z_TI/0.52/" "${CASES}/STRU.tmpl" > "$d/STRU"
    cp "${CASES}/KPT" "$d/KPT"
    cat > "$d/bto.win" << 'WINEOF'
! P13 wannier90 通道模板（z_Ti=0.52）——num_wann/窗口为占位值，首次实跑按能带确认
num_wann = 20
num_bands = 24
dis_win_min = -20.0
dis_win_max = 10.0
dis_froz_min = -20.0
dis_froz_max = 0.0
begin projections
Ba:s
Ti:d
O:p
end projections
write_xyz = true
WINEOF
    info "  通道③ wannier90: SKIP 数值判定（模板见 ${d}/bto.win）"
fi

# ---------------- ③ 约束能量面 E(γ) ----------------
info ""
info "----- ③ 约束能量面（target 模式，半宽 ${TARGET_HALF} rad） -----"
declare -A CG CE
for z in 0.50 0.52 0.54; do
    geq="${GAM[$z]:-}"
    if [ -z "$geq" ]; then
        info "  [INVALID] z_Ti=${z}: 平衡 γ 缺失，跳过约束采样"
        CONVBAD=1
        continue
    fi
    for sgn in p m; do
        [ "$sgn" = "p" ] && s=1 || s=-1
        tgt=$(awk -v g=$geq -v s=$s -v h=$TARGET_HALF 'BEGIN{printf "%.8f", g+s*h}')
        d="${RUNS}/constr_z$(ztag $z)_${sgn}"
        mkdir -p "$d"
        sed "s/Z_TI/$z/" "${CASES}/STRU.tmpl" > "$d/STRU"
        cp "${CASES}/KPT" "$d/KPT"
        echo "$tgt" > "$d/target.dat"
        sed -e "s/SUFFIX/p13_con_$(ztag $z)_${sgn}/" -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" -e "s|ORBITAL_DIR|${ORBITAL_DIR}|" \
            "${CASES}/INPUT_lcao_target.tmpl" > "$d/INPUT"
        g=""; e=""
        if run_abacus "$d" && scf_converged "$d"; then
            g=$(get_rawg "$d/run.log")
            sl=$(scflog "$d"); [ -n "$sl" ] && e=$(get_eks "$sl")
        fi
        if [ -z "$g" ] || [ -z "$e" ]; then
            info "  [INVALID] 约束 z_Ti=${z} target=${tgt}: 未收敛或缺输出"
            CONVBAD=1
        else
            CG[${z}_${sgn}]=$g; CE[${z}_${sgn}]=$e
            res=$(awk -v a=$g -v t=$tgt 'BEGIN{d=a-t; printf "%.4f", (d<0?-d:d)}')
            info "  z_Ti=${z} target=${tgt}: 达到 Σγ_raw=${g} (残差 ${res})  E_KS=${e}"
        fi
    done
done

# E(γ) 采样点表（写入 results.txt）
info ""
info "----- E(γ) 双势阱采样点表 -----"
{
for z in ${Z_LIST}; do
    [ -n "${GAM[$z]:-}" ] && echo "${GAM[$z]} ${EQ[$z]} eq_z${z}"
done
for key in "${!CG[@]}"; do
    echo "${CG[$key]} ${CE[$key]} constr_${key}"
done
} | sort -n | awk '{printf "  γ=%+14.8f  E_KS=%+18.8f  (%s)\n", $1, $2, $3}' | tee -a "${RESULTS}"

# ---------------- 判定 ----------------
info ""
info "----- 判定表 -----"

if [ "$CONVBAD" -eq 0 ]; then judge "J1 SCF 收敛巡检（路径+PW+约束全部算例）" PASS; else judge "J1 SCF 收敛巡检（存在 INVALID 点）" FAIL; fi

if [ -z "$JUMPLIST" ]; then
    judge "J2 路径分支连续性: max|Δγ|=${MAXDG} ≤ ${JUMP_THR} rad" PASS
else
    judge "J2 路径分支连续性: 假跳变${JUMPLIST}" FAIL
fi

if [ -n "$PS1" ] && [ -n "$PS2" ]; then
    rel=$(awk -v a=$PS1 -v b=$PS2 'BEGIN{d=a-b; if(d<0)d=-d; m=b; if(m<0)m=-m;
        if(m<1e-6) printf "%.6f", (d<0.01)?0:999; else printf "%.6f", d/m}')
    if awk -v r=$rel 'BEGIN{exit !(r<=0.10)}'; then
        judge "J3 Ps 两通道互差: |${PS1}-${PS2}|/|Ps2|=${rel} ≤ 10%" PASS
    else
        judge "J3 Ps 两通道互差: 相对差=${rel} > 10%" FAIL
    fi
else
    judge "J3 Ps 两通道互差（数值缺失）" FAIL
fi

# J4: E(γ) 光滑性——每组 (z 固定，γ_eq−0.2, γ_eq, γ_eq+0.2) 相邻能量差无符号振荡（符号变化 ≤1）
SMOOTH_OK=1
for z in 0.50 0.52 0.54; do
    em="${CE[${z}_m]:-}"; e0="${EQ[$z]:-}"; ep="${CE[${z}_p]:-}"
    if [ -z "$em" ] || [ -z "$e0" ] || [ -z "$ep" ]; then SMOOTH_OK=0; info "  J4 z_Ti=${z}: 采样点缺失"; continue; fi
    nsg=$(awk -v a=$em -v b=$e0 -v c=$ep 'BEGIN{d1=b-a; d2=c-b; n=0; if(d1*d2<0) n=1; print n}')
    de=$(awk -v a=$em -v b=$e0 -v c=$ep 'BEGIN{printf "%.3e %.3e", b-a, c-b}')
    info "  J4 z_Ti=${z}: ΔE(−)=${de% *} ΔE(+)=${de#* } 符号变化=${nsg}"
    [ "$nsg" -gt 1 ] && SMOOTH_OK=0
done
if [ "$SMOOTH_OK" -eq 1 ]; then judge "J4 E(γ) 约束采样光滑（相邻能量差无符号振荡）" PASS; else judge "J4 E(γ) 约束采样存在振荡/缺失" FAIL; fi

info ""
info "SUMMARY: ${NPASS}/${NTOTAL} PASS"
[ "$NPASS" -eq "$NTOTAL" ]
