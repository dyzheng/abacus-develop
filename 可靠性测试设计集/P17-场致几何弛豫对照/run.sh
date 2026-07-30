#!/bin/bash
# P17 场致几何弛豫对照：DeltaP 固定 lambda relax vs efield 固定 E relax（30 Bohr H2O）
# 用法: bash run.sh   （可用环境变量覆盖 ABACUS / PSEUDO_DIR / ORBITAL_DIR / NPROC）
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
PSEUDO_DIR="${PSEUDO_DIR:-/root/pporb/apns-pseudopotentials-v1}"
ORBITAL_DIR="${ORBITAL_DIR:-/root/pporb/apns-orbitals-efficiency-v1}"
NPROC="${NPROC:-1}"

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASES="$BASEDIR/cases"
RUNS="$BASEDIR/runs"
mkdir -p "$RUNS"
RESULTS="$RUNS/results.txt"

# ---- F1/F2 换算常量（备忘录定稿前工作值，修改集中于此） ----
PI=3.141592653589793
BOHR_PER_ANG=1.8897261254578284
A_BOHR=30.0                       # gdir 方向盒边长 (Bohr)
# F1: E(Ha/Bohr) = -PI*lambda/(2*A_BOHR)  (lambda 单位 Ry)
LAM_LIST="0.02 0.04"              # |lambda| (Ry), 各取正负

tag() { echo "$1" | sed 's/-/m/;s/\./p/'; }
mixb_for() { awk -v l="$1" 'BEGIN{print (l+0<0)?"0.3":"0.4"}'; }
e_of_lam() { awk -v l="$1" -v pi="$PI" -v a="$A_BOHR" 'BEGIN{printf "%.8f", -pi*l/(2*a)}'; }

run_abacus() {
    local d="$1"
    cd "$d"
    rm -rf OUT.*
    if [ "$NPROC" -gt 1 ]; then
        mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1 || echo "WARNING: abacus exit nonzero in $d"
    else
        "$ABACUS" > run.log 2>&1 || echo "WARNING: abacus exit nonzero in $d"
    fi
    cd "$BASEDIR"
}

gen_input() { # $1=模板 $2=输出 $3=lambda $4=mixing_beta $5=efield_amp
    sed -e "s#@PSEUDO_DIR@#$PSEUDO_DIR#g" \
        -e "s#@ORBITAL_DIR@#$ORBITAL_DIR#g" \
        -e "s#@LAMBDA@#$3#g" \
        -e "s#@MIXB@#$4#g" \
        -e "s#@EFIELD@#$5#g" "$1" > "$2"
}

is_conv()  { if grep -q 'SCF IS NOT CONVERGED' "$1/run.log" 2>/dev/null; then echo INVALID; else echo OK; fi; }

# 从 STRU_ION_D（Direct 坐标，relax 每步覆写，末帧即最终几何）计算 r_OH 均值与 HOH 角
# 输出: "r_OH(Ang) angle(deg)"；找不到文件输出 "NAN NAN"
geom_from_iond() { # $1=STRU_ION_D 路径
    awk -v bpa="$BOHR_PER_ANG" '
    /^LATTICE_CONSTANT/{ getline; lc=$1+0 }
    /^LATTICE_VECTORS/{ for(i=1;i<=3;i++){ getline; v[i,1]=$1+0; v[i,2]=$2+0; v[i,3]=$3+0 } }
    /^ATOMIC_POSITIONS/{ inpos=1; getline; mode=$1; next }
    inpos && $1=="O" { getline; getline; na=$1+0;
        for(i=0;i<na;i++){ getline; fx=$1; fy=$2; fz=$3;
          cx=lc*(v[1,1]*fx+v[1,2]*fy+v[1,3]*fz); cy=lc*(v[2,1]*fx+v[2,2]*fy+v[2,3]*fz); cz=lc*(v[3,1]*fx+v[3,2]*fy+v[3,3]*fz);
          ox=cx/bpa; oy=cy/bpa; oz=cz/bpa } }
    inpos && $1=="H" { getline; getline; na=$1+0;
        for(i=1;i<=na;i++){ getline; fx=$1; fy=$2; fz=$3;
          cx=lc*(v[1,1]*fx+v[1,2]*fy+v[1,3]*fz); cy=lc*(v[2,1]*fx+v[2,2]*fy+v[2,3]*fz); cz=lc*(v[3,1]*fx+v[3,2]*fy+v[3,3]*fz);
          hx[i]=cx/bpa; hy[i]=cy/bpa; hz[i]=cz/bpa } }
    END{
      if (mode!="Direct" || hx[1]=="" || hx[2]=="") { print "NAN NAN"; exit }
      v1x=hx[1]-ox; v1y=hy[1]-oy; v1z=hz[1]-oz;
      v2x=hx[2]-ox; v2y=hy[2]-oy; v2z=hz[2]-oz;
      r1=sqrt(v1x*v1x+v1y*v1y+v1z*v1z); r2=sqrt(v2x*v2x+v2y*v2y+v2z*v2z);
      cang=(v1x*v2x+v1y*v2y+v1z*v2z)/(r1*r2);
      if(cang>1)cang=1; if(cang<-1)cang=-1;
      ang=atan2(sqrt(1-cang*cang), cang)*180/3.141592653589793;
      printf "%.6f %.6f", (r1+r2)/2, ang
    }' "$1" 2>/dev/null || echo "NAN NAN"
}

# 初始参考几何（cases/STRU, Cartesian_angstrom, 直接 Å）
geom_ref() {
    awk '
    /^ATOMIC_POSITIONS/{ inpos=1; getline; mode=$1; next }
    inpos && $1=="O" { getline; getline; getline; ox=$1; oy=$2; oz=$3 }
    inpos && $1=="H" { getline; getline; na=$1+0;
        for(i=1;i<=na;i++){ getline; hx[i]=$1; hy[i]=$2; hz[i]=$3 } }
    END{
      v1x=hx[1]-ox; v1y=hy[1]-oy; v1z=hz[1]-oz;
      v2x=hx[2]-ox; v2y=hy[2]-oy; v2z=hz[2]-oz;
      r1=sqrt(v1x*v1x+v1y*v1y+v1z*v1z); r2=sqrt(v2x*v2x+v2y*v2y+v2z*v2z);
      cang=(v1x*v2x+v1y*v2y+v1z*v2z)/(r1*r2);
      ang=atan2(sqrt(1-cang*cang), cang)*180/3.141592653589793;
      printf "%.6f %.6f", (r1+r2)/2, ang
    }' "$CASES/STRU"
}

DATA="$RUNS/data.tsv"
: > "$DATA"

echo "===== P17 场致几何弛豫对照 ====="
echo "  ABACUS=$ABACUS  NPROC=$NPROC"

for L in $LAM_LIST; do
    for SGN in "" "-"; do
        LAM="$SGN$L"
        # DeltaP 固定 lambda relax
        d="$RUNS/lam_$(tag "$LAM")"
        mkdir -p "$d"; cp "$CASES/STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
        gen_input "$CASES/INPUT.relax.deltap.tmpl" "$d/INPUT" "$LAM" "$(mixb_for "$LAM")" 0.0
        run_abacus "$d"
        f="$(ls "$d"/OUT.*/STRU_ION_D 2>/dev/null | head -1 || true)"
        read -r R ANG <<< "$( [ -n "$f" ] && geom_from_iond "$f" || echo "NAN NAN")"
        echo -e "dp\t$LAM\t$R\t$ANG\t$(is_conv "$d")" >> "$DATA"

        # efield 固定 E relax (F1 换算)
        E="$(e_of_lam "$LAM")"
        d="$RUNS/ef_$(tag "$LAM")"
        mkdir -p "$d"; cp "$CASES/STRU" "$d/STRU"; cp "$CASES/KPT" "$d/KPT"
        gen_input "$CASES/INPUT.relax.efield.tmpl" "$d/INPUT" 0.0 0.4 "$E"
        run_abacus "$d"
        f="$(ls "$d"/OUT.*/STRU_ION_D 2>/dev/null | head -1 || true)"
        read -r R ANG <<< "$( [ -n "$f" ] && geom_from_iond "$f" || echo "NAN NAN")"
        echo -e "ef\t$LAM\t$R\t$ANG\t$(is_conv "$d")" >> "$DATA"
    done
done

read -r R0 ANG0 <<< "$(geom_ref)"

declare -A RR AA CC
while IFS=$'\t' read -r ch lam r ang conv; do
    RR["$ch $lam"]="$r"; AA["$ch $lam"]="$ang"; CC["$ch $lam"]="$conv"
done < "$DATA"

delta() { # $1=值 $2=参考
    case "$1|$2" in *NAN*|""*) echo NAN; return;; esac
    awk -v x="$1" -v r="$2" 'BEGIN{printf "%.6f", x-r}'
}
absd() {
    case "$1|$2" in *NAN*|""*) echo NAN; return;; esac
    awk -v x="$1" -v y="$2" 'BEGIN{d=x-y; if(d<0)d=-d; printf "%.6f", d}'
}
signs() { # $1 $2 -> same/diff/NA
    case "$1|$2" in *NAN*|""*) echo NA; return;; esac
    awk -v x="$1" -v y="$2" 'BEGIN{sx=(x>0)-(x<0); sy=(y>0)-(y<0); print (sx==sy)?"same":"DIFF"}'
}

{
echo ""
echo "----- 数据表 -----"
echo "初始几何 (cases/STRU): r_OH=$R0 Ang, angle=$ANG0 deg"
printf "%-4s %8s %12s %12s %12s %12s %8s\n" "通道" "lambda" "r_OH" "Dr" "angle" "Dangle" "SCF"
while IFS=$'\t' read -r ch lam r ang conv; do
    printf "%-4s %8s %12s %12s %12s %12s %8s\n" "$ch" "$lam" "$r" "$(delta "$r" "$R0")" "$ang" "$(delta "$ang" "$ANG0")" "$conv"
done < "$DATA"

echo ""
echo "WARNING: 判定待 F1/F2 备忘录定稿（阻塞项见 README）——以下判据仅记录数值不参与 SUMMARY"
for L in $LAM_LIST; do
    for SGN in "" "-"; do
        LAM="$SGN$L"
        DR_DP="$(delta "${RR[dp $LAM]:-NAN}" "$R0")"; DR_EF="$(delta "${RR[ef $LAM]:-NAN}" "$R0")"
        DA_DP="$(delta "${AA[dp $LAM]:-NAN}" "$ANG0")"; DA_EF="$(delta "${AA[ef $LAM]:-NAN}" "$ANG0")"
        echo "[lambda=$LAM] |Dr_dp-Dr_ef|=$(absd "$DR_DP" "$DR_EF") (<=0.005 Ang)  |Dang_dp-Dang_ef|=$(absd "$DA_DP" "$DA_EF") (<=0.5 deg)  sign(Dr)=$(signs "$DR_DP" "$DR_EF")  sign(Dang)=$(signs "$DA_DP" "$DA_EF")  [SCF dp=${CC[dp $LAM]:-?} ef=${CC[ef $LAM]:-?}]"
    done
done

echo ""
echo "原始数据: $DATA"
echo "SUMMARY: 0/0 PASS (全部判据阻塞于 F1/F2, 见 WARNING)"
} | tee "$RESULTS"

exit 0
