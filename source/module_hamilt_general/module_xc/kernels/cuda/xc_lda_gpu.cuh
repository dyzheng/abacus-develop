// GPU device functions and kernels for LDA XC functionals
// Ported from xc_funct_exch_lda.cpp and xc_funct_corr_lda.cpp
#ifndef XC_LDA_GPU_CUH
#define XC_LDA_GPU_CUH

#include <cuda_runtime.h>

namespace XC_GPU
{

// ============================================================
// LDA Exchange device functions (unpolarized)
// ============================================================

__device__ inline void d_slater(const double rs, double& ex, double& vx)
{
    const double f = -0.687247939924714;
    const double alpha = 2.0 / 3.0;
    ex = f * alpha / rs;
    vx = 4.0 / 3.0 * f * alpha / rs;
}

__device__ inline void d_slater1(const double rs, double& ex, double& vx)
{
    const double f = -0.687247939924714;
    const double alpha = 1.0;
    ex = f * alpha / rs;
    vx = 4.0 / 3.0 * f * alpha / rs;
}

// ============================================================
// LDA Correlation device functions (unpolarized)
// ============================================================

__device__ inline void d_pw(const double rs, const int iflag, double& ec, double& vc)
{
    const double a = 0.0310910;
    const double b1 = 7.59570;
    const double b2 = 3.58760;
    const double c0 = a;
    const double c1 = 0.0466440;
    const double c2 = 0.006640;
    const double c3 = 0.010430;
    const double d0 = 0.43350;
    const double d1_c = 1.44080;
    const double a1_arr[2] = {0.213700, 0.0264810};
    const double b3_arr[2] = {1.63820, -0.466470};
    const double b4_arr[2] = {0.492940, 0.133540};
    if (rs < 1 && iflag == 1)
    {
        double lnrs = log(rs);
        ec = c0 * lnrs - c1 + c2 * rs * lnrs - c3 * rs;
        vc = c0 * lnrs - (c1 + c0 / 3.0) + 2.0 / 3.0 * c2 * rs * lnrs
             - (2.0 * c3 + c2) / 3.0 * rs;
    }
    else if (rs > 100.0 && iflag == 1)
    {
        ec = -d0 / rs + d1_c / pow(rs, 1.5);
        vc = -4.0 / 3.0 * d0 / rs + 1.5 * d1_c / pow(rs, 1.5);
    }
    else
    {
        double rs12 = sqrt(rs);
        double rs32 = rs * rs12;
        double rs2 = rs * rs;
        double om = 2.0 * a * (b1 * rs12 + b2 * rs + b3_arr[iflag] * rs32 + b4_arr[iflag] * rs2);
        double dom = 2.0 * a * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3_arr[iflag] * rs32
                                + 2.0 * b4_arr[iflag] * rs2);
        double olog = log(1.0 + 1.0 / om);
        ec = -2.0 * a * (1.0 + a1_arr[iflag] * rs) * olog;
        vc = -2.0 * a * (1.0 + 2.0 / 3.0 * a1_arr[iflag] * rs) * olog
             - 2.0 / 3.0 * a * (1.0 + a1_arr[iflag] * rs) * dom / (om * (om + 1.0));
    }
}

__device__ inline void d_pz(const double rs, const int iflag, double& ec, double& vc)
{
    const double a[2] = {0.0311, 0.031091};
    const double b[2] = {-0.048, -0.046644};
    const double c[2] = {0.0020, 0.00419};
    const double d[2] = {-0.0116, -0.00983};
    const double gc[2] = {-0.1423, -0.103756};
    const double b1[2] = {1.0529, 0.56371};
    const double b2[2] = {0.3334, 0.27358};

    if (rs < 1.0)
    {
        double lnrs = log(rs);
        ec = a[iflag] * lnrs + b[iflag] + c[iflag] * rs * lnrs + d[iflag] * rs;
        vc = a[iflag] * lnrs + (b[iflag] - a[iflag] / 3.0)
             + 2.0 / 3.0 * c[iflag] * rs * lnrs + (2.0 * d[iflag] - c[iflag]) / 3.0 * rs;
    }
    else
    {
        double rs12 = sqrt(rs);
        double ox = 1.0 + b1[iflag] * rs12 + b2[iflag] * rs;
        double dox = 1.0 + 7.0 / 6.0 * b1[iflag] * rs12 + 4.0 / 3.0 * b2[iflag] * rs;
        ec = gc[iflag] / ox;
        vc = ec * dox / ox;
    }
}

__device__ inline void d_lyp(const double rs, double& ec, double& vc)
{
    const double a = 0.04918;
    const double b = 0.1320 * 2.87123400018819108;
    const double pi43 = 1.61199195401647;
    const double c = 0.2533 * pi43;
    const double d = 0.349 * pi43;

    double ecrs = b * exp(-c * rs);
    double ox = 1.0 / (1.0 + d * rs);
    ec = -a * ox * (1.0 + ecrs);
    vc = ec - rs / 3.0 * a * ox * (d * ox + ecrs * (d * ox + c));
}

// ============================================================
// LDA Exchange device functions (spin-polarized)
// ============================================================

__device__ inline void d_slater_spin(const double rho, const double zeta,
                                     double& ex, double& vxup, double& vxdw)
{
    const double f = -1.107838149573033610;
    const double alpha = 2.0 / 3.0;
    const double third = 1.0 / 3.0;
    const double p43 = 4.0 / 3.0;

    double rho13 = pow((1.0 + zeta) * rho, third);
    double exup = f * alpha * rho13;
    vxup = p43 * f * alpha * rho13;
    rho13 = pow((1.0 - zeta) * rho, third);
    double exdw = f * alpha * rho13;
    vxdw = p43 * f * alpha * rho13;
    ex = 0.5 * ((1.0 + zeta) * exup + (1.0 - zeta) * exdw);
}
// ============================================================
// LDA Correlation device functions (spin-polarized)
// ============================================================

__device__ inline void d_pw_spin(const double rs, const double zeta,
                                 double& ec, double& vcup, double& vcdw)
{
    const double a = 0.0310910, a1 = 0.213700;
    const double b1 = 7.59570, b2 = 3.58760, b3 = 1.63820, b4 = 0.492940;
    const double ap = 0.0155450, a1p = 0.205480;
    const double b1p = 14.11890, b2p = 6.19770, b3p = 3.36620, b4p = 0.625170;
    const double aa = 0.0168870, a1a = 0.111250;
    const double b1a = 10.3570, b2a = 3.62310, b3a = 0.880260, b4a = 0.496710;
    const double fz0 = 1.7099210;

    double zeta2 = zeta * zeta;
    double zeta3 = zeta2 * zeta;
    double zeta4 = zeta3 * zeta;
    double rs12 = sqrt(rs);
    double rs32 = rs * rs12;
    double rs2 = rs * rs;

    // unpolarised
    double om = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2);
    double dom = 2.0 * a * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2.0 * b4 * rs2);
    double olog = log(1.0 + 1.0 / om);
    double epwc = -2.0 * a * (1.0 + a1 * rs) * olog;
    double vpwc = -2.0 * a * (1.0 + 2.0 / 3.0 * a1 * rs) * olog
                  - 2.0 / 3.0 * a * (1.0 + a1 * rs) * dom / (om * (om + 1.0));

    // polarized
    double omp = 2.0 * ap * (b1p * rs12 + b2p * rs + b3p * rs32 + b4p * rs2);
    double domp = 2.0 * ap * (0.5 * b1p * rs12 + b2p * rs + 1.5 * b3p * rs32 + 2.0 * b4p * rs2);
    double ologp = log(1.0 + 1.0 / omp);
    double epwcp = -2.0 * ap * (1.0 + a1p * rs) * ologp;
    double vpwcp = -2.0 * ap * (1.0 + 2.0 / 3.0 * a1p * rs) * ologp
                   - 2.0 / 3.0 * ap * (1.0 + a1p * rs) * domp / (omp * (omp + 1.0));

    // antiferro
    double oma = 2.0 * aa * (b1a * rs12 + b2a * rs + b3a * rs32 + b4a * rs2);
    double doma = 2.0 * aa * (0.5 * b1a * rs12 + b2a * rs + 1.5 * b3a * rs32 + 2.0 * b4a * rs2);
    double ologa = log(1.0 + 1.0 / oma);
    double alpha = 2.0 * aa * (1.0 + a1a * rs) * ologa;
    double vpwca = 2.0 * aa * (1.0 + 2.0 / 3.0 * a1a * rs) * ologa
                   + 2.0 / 3.0 * aa * (1.0 + a1a * rs) * doma / (oma * (oma + 1.0));

    double fz = (pow(1.0 + zeta, 4.0 / 3.0) + pow(1.0 - zeta, 4.0 / 3.0) - 2.0)
                / (pow(2.0, 4.0 / 3.0) - 2.0);
    double dfz = (pow(1.0 + zeta, 1.0 / 3.0) - pow(1.0 - zeta, 1.0 / 3.0))
                 * 4.0 / (3.0 * (pow(2.0, 4.0 / 3.0) - 2.0));

    ec = epwc + alpha * fz * (1.0 - zeta4) / fz0 + (epwcp - epwc) * fz * zeta4;

    vcup = vpwc + vpwca * fz * (1.0 - zeta4) / fz0
           + (vpwcp - vpwc) * fz * zeta4
           + (alpha / fz0 * (dfz * (1.0 - zeta4) - 4.0 * fz * zeta3)
              + (epwcp - epwc) * (dfz * zeta4 + 4.0 * fz * zeta3)) * (1.0 - zeta);

    vcdw = vpwc + vpwca * fz * (1.0 - zeta4) / fz0
           + (vpwcp - vpwc) * fz * zeta4
           - (alpha / fz0 * (dfz * (1.0 - zeta4) - 4.0 * fz * zeta3)
              + (epwcp - epwc) * (dfz * zeta4 + 4.0 * fz * zeta3)) * (1.0 + zeta);
}

__device__ inline void d_pz_polarized(const double rs, double& ec, double& vc)
{
    const double a = 0.015550, b = -0.02690, c = 0.00070, d = -0.00480;
    const double gc = -0.08430, b1 = 1.39810, b2 = 0.26110;

    if (rs < 1.0)
    {
        double lnrs = log(rs);
        ec = a * lnrs + b + c * rs * lnrs + d * rs;
        vc = a * lnrs + (b - a / 3.0) + 2.0 / 3.0 * c * rs * lnrs + (2.0 * d - c) / 3.0 * rs;
    }
    else
    {
        double rs12 = sqrt(rs);
        double ox = 1.0 + b1 * rs12 + b2 * rs;
        double dox = 1.0 + 7.0 / 6.0 * b1 * rs12 + 4.0 / 3.0 * b2 * rs;
        ec = gc / ox;
        vc = ec * dox / ox;
    }
}

__device__ inline void d_pz_spin(const double rs, const double zeta,
                                 double& ec, double& vcup, double& vcdw)
{
    const double p43 = 4.0 / 3.0;
    const double third = 1.0 / 3.0;

    double ecu, vcu, ecp, vcp;
    d_pz(rs, 0, ecu, vcu);
    d_pz_polarized(rs, ecp, vcp);

    double fz = (pow(1.0 + zeta, p43) + pow(1.0 - zeta, p43) - 2.0) / (pow(2.0, p43) - 2.0);
    double dfz = p43 * (pow(1.0 + zeta, third) - pow(1.0 - zeta, third)) / (pow(2.0, p43) - 2.0);

    ec = ecu + fz * (ecp - ecu);
    vcup = vcu + fz * (vcp - vcu) + (ecp - ecu) * dfz * (1.0 - zeta);
    vcdw = vcu + fz * (vcp - vcu) + (ecp - ecu) * dfz * (-1.0 - zeta);
}

// ============================================================
// Dispatch functions: call the right functional based on func_id
// ============================================================

// Unpolarized XC dispatch (single functional)
__device__ inline void d_xc_single(const int func_id, const double rs,
                                   double& e, double& v, const double hybrid_alpha)
{
    e = v = 0.0;
    switch (func_id)
    {
        case 1:   // XC_LDA_X
        case 101: // XC_GGA_X_PBE
        case 117: // XC_GGA_X_PBE_R
        case 116: // XC_GGA_X_PBE_SOL
        case 118: // XC_GGA_X_WC
        case 106: // XC_GGA_X_B88
        case 109: // XC_GGA_X_PW91
            d_slater(rs, e, v);
            break;
        case 406: // XC_HYB_GGA_XC_PBEH (PBE0)
        {
            double ex, vx, ec, vc;
            d_slater(rs, ex, vx);
            ex *= (1.0 - hybrid_alpha);
            vx *= (1.0 - hybrid_alpha);
            d_pw(rs, 0, ec, vc);
            e = ex + ec;
            v = vx + vc;
            break;
        }
        case 130: // XC_GGA_C_PBE
        case 134: // XC_GGA_C_PW91
        case 12:  // XC_LDA_C_PW
        case 133: // XC_GGA_C_PBE_SOL
            d_pw(rs, 0, e, v);
            break;
        case 9:   // XC_LDA_C_PZ
        case 132: // XC_GGA_C_P86
            d_pz(rs, 0, e, v);
            break;
        case 131: // XC_GGA_C_LYP
            d_lyp(rs, e, v);
            break;
        default:
            break;
    }
}

// Spin-polarized XC dispatch (single functional)
__device__ inline void d_xc_spin_single(const int func_id, const double rs, const double rho,
                                        const double zeta, double& e, double& vup, double& vdw,
                                        const double hybrid_alpha)
{
    e = vup = vdw = 0.0;
    switch (func_id)
    {
        case 1: case 101: case 117: case 116: case 118: case 106: case 109:
            d_slater_spin(rho, zeta, e, vup, vdw);
            break;
        case 406:
        {
            double ex, vupx, vdwx, ec, vupc, vdwc;
            d_slater_spin(rho, zeta, ex, vupx, vdwx);
            ex *= (1.0 - hybrid_alpha);
            vupx *= (1.0 - hybrid_alpha);
            vdwx *= (1.0 - hybrid_alpha);
            d_pw_spin(rs, zeta, ec, vupc, vdwc);
            e = ex + ec; vup = vupx + vupc; vdw = vdwx + vdwc;
            break;
        }
        case 9: case 132:
            d_pz_spin(rs, zeta, e, vup, vdw);
            break;
        case 130: case 133: case 12:
            d_pw_spin(rs, zeta, e, vup, vdw);
            break;
        default:
            break;
    }
}

} // namespace XC_GPU

#endif // XC_LDA_GPU_CUH
