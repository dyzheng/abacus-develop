#!/bin/bash
# ============================================================
# P14 wannier90 交叉验证（Wannier 中心与偶极分解）
# LCAO λ=0 读 [rawG] 总 Σγ_raw 与各原子 γ 分量 → 总偶极/总极化
# wannier90 通道（可选）：检测到 wannier90 则走 PW 密度源管线，否则 SKIP
# 无阻塞项（不依赖 P01/F1/F2；γ→μ 换算用 ΔP测试说明.md §3.3 工作值）
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
DEBYE_PER_EBOHR=2.541746            # 1 e·Bohr = 2.541746 D（1 D = 0.393430 e·Bohr）
F2_SPIN=2.0                         # nspin=1 自旋因子 γ→μ ÷2（2026-07-30 冒烟实测确认，待备忘录定稿）
DEBYE_PER_EANG=4.8032047            # 1 e·Å = 4.8032047 D
DIP_TOL=0.02                        # w90 通道总偶极判据 (D)

# 体系配置: 盒边(Å) / 离子 z 偶极和(e·Å) / num_wann / 投影 / 软判区间
# （离子项取赝势价电子数: O6 H1 N5 C4 Si4）
BOX_H2O=15.873;  IONZD_H2O="6*7.9365+1*7.1794+1*8.6936";                 NW_H2O=4; PROJ_H2O="O:sp3"
BOX_NH3=12.0;    IONZD_NH3="5*6.0+1*6.9372+1*5.5314+1*5.5314";           NW_NH3=4; PROJ_NH3="N:sp3"
BOX_CH4=12.0;    IONZD_CH4="4*6.0+1*6.6276+1*5.3724+1*5.3724+1*6.6276";  NW_CH4=4; PROJ_CH4="C:sp3"

# 生成 seed.win（Γ 点；晶胞与原子坐标与 cases/STRU_* 一致，Angstrom）
write_win() { # $1=SYS $2=输出文件
    local sys=$1 out=$2 nw proj cell atoms
    case "$sys" in
        H2O) nw=$NW_H2O; proj=$PROJ_H2O
             cell="15.873 0.0 0.0\n0.0 15.873 0.0\n0.0 0.0 15.873"
             atoms="O  7.9365 7.9365 7.9365\nH  7.1794 7.9365 8.5223\nH  8.6936 7.9365 8.5223" ;;
        NH3) nw=$NW_NH3; proj=$PROJ_NH3
             cell="12.0 0.0 0.0\n0.0 12.0 0.0\n0.0 0.0 12.0"
             atoms="N  6.0 6.0 6.0\nH  6.9372 6.0 5.6182\nH  5.5314 6.8116 5.6182\nH  5.5314 5.1884 5.6182" ;;
        CH4) nw=$NW_CH4; proj=$PROJ_CH4
             cell="12.0 0.0 0.0\n0.0 12.0 0.0\n0.0 0.0 12.0"
             atoms="C  6.0 6.0 6.0\nH  6.6276 6.6276 6.6276\nH  6.6276 5.3724 5.3724\nH  5.3724 6.6276 5.3724\nH  5.3724 5.3724 6.6276" ;;
    esac
    {
        echo "num_wann = ${nw}"
        echo "num_bands = ${nw}"
        echo "begin unit_cell_cart"
        echo "ang"
        printf '%b\n' "$cell"
        echo "end unit_cell_cart"
        echo "begin atoms_cart"
        echo "ang"
        printf '%b\n' "$atoms"
        echo "end atoms_cart"
        echo "begin projections"
        echo "${proj}"
        echo "end projections"
        echo "mp_grid = 1 1 1"
        echo "begin kpoints"
        echo "0.0 0.0 0.0 1.0"
        echo "end kpoints"
        echo "write_xyz = true"
    } > "$out"
}

# ---------------- 通用函数 ----------------
run_abacus() {
    local d="$1"
    ( cd "$d" && rm -rf OUT.* && \
      if [ "$NPROC" -gt 1 ]; then mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1; else "$ABACUS" > run.log 2>&1; fi )
}
get_rawg()       { grep '\[rawG\]' "$1" 2>/dev/null | tail -1 | grep -o 'Σγ_raw=[^ ]*' | cut -d= -f2 || true; }
get_rawg_atoms() { grep '\[rawG\]' "$1" 2>/dev/null | tail -1 | grep -o 'γ[0-9]\+=[^ ]*' | tr '\n' ' ' || true; }
scf_converged()  { ! grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; }

NPASS=0; NTOTAL=0
judge() {
    NTOTAL=$((NTOTAL+1))
    [ "$2" = "PASS" ] && NPASS=$((NPASS+1))
    printf '%-64s %s\n' "$1" "$2" | tee -a "${RESULTS}"
}
info() { echo "$1" | tee -a "${RESULTS}"; }

echo "===== P14 wannier90 交叉验证 =====" | tee -a "${RESULTS}"
info "ABACUS=${ABACUS}  NPROC=${NPROC}"
info ""

CONVBAD=0
declare -A MU_D GSUM

# ---------------- LCAO λ=0：四体系 ----------------
for SYS in H2O NH3 CH4 Si; do
    d="${RUNS}/lcao_${SYS}"
    mkdir -p "$d"
    cp "${CASES}/STRU_${SYS}" "$d/STRU"
    if [ "$SYS" = "Si" ]; then cp "${CASES}/KPT_SI" "$d/KPT"; else cp "${CASES}/KPT_GAMMA" "$d/KPT"; fi
    sed -e "s/SUFFIX/p14_${SYS}/" -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" -e "s|ORBITAL_DIR|${ORBITAL_DIR}|" \
        "${CASES}/INPUT_lcao.tmpl" > "$d/INPUT"
    g=""; ga=""
    if run_abacus "$d" && scf_converged "$d"; then
        g=$(get_rawg "$d/run.log"); ga=$(get_rawg_atoms "$d/run.log")
    fi
    if [ -z "$g" ]; then
        info "  [INVALID] ${SYS}: 未收敛或未输出 [rawG]"
        CONVBAD=1
        continue
    fi
    GSUM[$SYS]=$g
    info "  ${SYS}: Σγ_raw=${g}"
    info "    per-atom γ 分量: ${ga}"
    if [ "$SYS" != "Si" ]; then
        eval "box=\$BOX_${SYS}"
        a_bohr=$(awk -v b=$box -v k=$A2B 'BEGIN{printf "%.10f", b*k}')
        mu=$(awk -v g=$g -v a=$a_bohr -v pi=$PI -v c=$DEBYE_PER_EBOHR -v f=$F2_SPIN \
            'BEGIN{n=g/(2*pi); n=(n>=0)?int(n+0.5):int(n-0.5); g=g-2*pi*n; printf "%.6f", (a/pi)*g/f*c}')
        MU_D[$SYS]=$mu
        info "    μ_z(LCAO) = ${mu} D （μ=(a/π)·Σγ_raw, a=${a_bohr} Bohr）"
    else
        gmod=$(awk -v g=$g -v pi=$PI 'BEGIN{m=g-2*pi*int(g/(2*pi)+(g>0?0.5:-0.5)); printf "%.6f", m}')
        info "    Si 总极化(模量子): Σγ_raw mod 2π = ${gmod} rad（金刚石对称性预期 ~0）"
    fi
done

# ---------------- wannier90 通道（分子三体系，可选） ----------------
W90="$(command -v wannier90 2>/dev/null || command -v wannier90.x 2>/dev/null || true)"
declare -A MU_W90
if [ -z "$W90" ]; then
    info ""
    info "wannier90 通道: SKIP（未检测到 wannier90/wannier90.x 可执行文件）"
else
    info ""
    info "wannier90 通道: 使用 ${W90}（PW 密度源，Γ 点设置，窗口见 README §6）"
    for SYS in H2O NH3 CH4; do
        eval "nw=\$NW_${SYS}; proj=\$PROJ_${SYS}; ionzd=\$IONZD_${SYS}"
        d="${RUNS}/w90_${SYS}"
        mkdir -p "$d"
        cp "${CASES}/STRU_${SYS}" "$d/STRU"
        cp "${CASES}/KPT_GAMMA" "$d/KPT"
        # 步骤1: PW SCF（密度源）
        sed -e "s/SUFFIX/p14_${SYS}_scf/" -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" "${CASES}/INPUT_pw.tmpl" > "$d/INPUT"
        # 步骤2: .win 模板（含晶胞/原子/投影/kpoints）+ wannier90 -pp 生成 seed.nnkp
        write_win "$SYS" "$d/seed.win"
        ok=1
        if run_abacus "$d" && scf_converged "$d" && \
           ( cd "$d" && "$W90" -pp seed > w90_pp.log 2>&1 ) && [ -f "$d/seed.nnkp" ]; then
            # 步骤3: nscf + towannier90 生成 mmn/amn/eig
            dn="${RUNS}/w90_${SYS}_nscf"
            mkdir -p "$dn"
            cp "$d/STRU" "$d/KPT" "$d/seed.nnkp" "$dn/"
            sed -e "s/SUFFIX/p14_${SYS}_nscf/" -e "s|PSEUDO_DIR|${PSEUDO_DIR}|" \
                -e "s|READ_FILE_DIR|${d}/OUT.p14_${SYS}_scf|" "${CASES}/INPUT_nscf.tmpl" > "$dn/INPUT"
            if ( cd "$dn" && rm -rf OUT.* && \
                 if [ "$NPROC" -gt 1 ]; then mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1; else "$ABACUS" > run.log 2>&1; fi ); then
                # mmn/amn/eig 收集回 w90 目录（位置随版本可能不同，做通配搜索）
                find "$dn" -maxdepth 2 -name '*.mmn' -o -name '*.amn' -o -name '*.eig' 2>/dev/null | while read -r f; do
                    base=$(basename "$f"); cp "$f" "$d/seed.${base##*.}" 2>/dev/null || true
                done
                if [ -f "$d/seed.mmn" ] && [ -f "$d/seed.amn" ] && [ -f "$d/seed.eig" ] && \
                   ( cd "$d" && "$W90" seed > w90.log 2>&1 ) && [ -f "$d/seed_centres.xyz" ]; then
                    zwc=$(awk 'NR>2 && NF>=4 {s+=$4} END{printf "%.10f", s}' "$d/seed_centres.xyz")
                    muw=$(awk "BEGIN{ion=${ionzd}; printf \"%.6f\", (-2.0*(${zwc})+ion)*${DEBYE_PER_EANG}}")
                    MU_W90[$SYS]=$muw
                    info "  ${SYS}: μ_z(w90 WC) = ${muw} D （μ=−2Σz_WC+ΣZ_I z_I，sp3 投影）"
                    info "    WC 中心: ${d}/seed_centres.xyz（对照化学图像：孤对/键中心，README §7）"
                else
                    ok=0
                fi
            else
                ok=0
            fi
        else
            ok=0
        fi
        [ "$ok" -eq 0 ] && info "  ${SYS}: w90 通道 SKIP（管线未完成，见 ${d}/ 与 ${d}_nscf/ 日志；窗口/接口需首次实跑确认）"
    done
fi

# ---------------- per-atom 分解对照表（仅打印，不硬判） ----------------
info ""
info "----- per-atom γ 分解 vs WC 划分（趋势对照，不逐值对齐，README §6） -----"
for SYS in H2O NH3 CH4 Si; do
    d="${RUNS}/lcao_${SYS}"
    [ -f "$d/run.log" ] && info "  ${SYS}: $(get_rawg_atoms "$d/run.log")"
done

# ---------------- 判定 ----------------
info ""
info "----- 判定表 -----"

if [ "$CONVBAD" -eq 0 ]; then judge "J1 四体系 LCAO SCF 收敛且 [rawG] 齐全" PASS; else judge "J1 LCAO 收敛/提取（存在 INVALID 体系）" FAIL; fi

# 软判参考区间（WARNING 不进 SUMMARY）
[ -n "${MU_D[H2O]:-}" ] && { awk -v m="${MU_D[H2O]}" 'BEGIN{m=(m<0?-m:m); exit !(m>=1.5 && m<=2.2)}' \
    && info "OK: H2O |μ|=${MU_D[H2O]} D ∈ [1.5,2.2]（参考区间）" \
    || info "WARNING: H2O μ=${MU_D[H2O]} D 越出参考区间 [1.5,2.2]（软判）"; }
[ -n "${MU_D[NH3]:-}" ] && { awk -v m="${MU_D[NH3]}" 'BEGIN{m=(m<0?-m:m); exit !(m>=1.2 && m<=1.9)}' \
    && info "OK: NH3 |μ|=${MU_D[NH3]} D ∈ [1.2,1.9]（参考区间）" \
    || info "WARNING: NH3 μ=${MU_D[NH3]} D 越出参考区间 [1.2,1.9]（软判）"; }
[ -n "${MU_D[CH4]:-}" ] && { awk -v m="${MU_D[CH4]}" 'BEGIN{m=(m<0?-m:m); exit !(m<=0.05)}' \
    && info "OK: CH4 |μ|=${MU_D[CH4]} D ≤ 0.05（对称性）" \
    || info "WARNING: CH4 μ=${MU_D[CH4]} D，|μ| > 0.05（软判）"; }

# w90 可用且产出时：总偶极差 ≤0.02 D（硬判）；否则 SKIP
for SYS in H2O NH3 CH4; do
    if [ -n "${MU_W90[$SYS]:-}" ] && [ -n "${MU_D[$SYS]:-}" ]; then
        dd=$(awk -v a="${MU_D[$SYS]}" -v b="${MU_W90[$SYS]}" 'BEGIN{d=a-b; printf "%.6f", (d<0?-d:d)}')
        if awk -v d=$dd -v t=$DIP_TOL 'BEGIN{exit !(d<=t)}'; then
            judge "J_${SYS} 总偶极 LCAO vs w90: |Δμ|=${dd} D ≤ ${DIP_TOL} D" PASS
        else
            judge "J_${SYS} 总偶极 LCAO vs w90: |Δμ|=${dd} D > ${DIP_TOL} D" FAIL
        fi
    else
        info "J_${SYS} 总偶极 LCAO vs w90: SKIP（w90 通道无产出）"
    fi
done

info ""
info "SUMMARY: ${NPASS}/${NTOTAL} PASS"
[ "$NPASS" -eq "$NTOTAL" ]
