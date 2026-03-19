// GPU device functions for GGA exchange-correlation functionals
// Ported from xc_funct_exch_gga.cpp, xc_funct_corr_gga.cpp, xc_funct_hcth.cpp
#ifndef XC_GGA_GPU_CUH
#define XC_GGA_GPU_CUH

#include <cuda_runtime.h>
#include "xc_lda_gpu.cuh"

namespace XC_GPU
{

// ============================================================
// GGA Exchange device functions (unpolarized)
// ============================================================

__device__ inline void d_becke88(const double rho, const double grho,
                                  double& sx, double& v1x, double& v2x)
{
    const double beta = 0.00420;
    const double third = 1.0 / 3.0;
    const double two13 = 1.2599210498948730;

    double rho13 = pow(rho, third);
    double rho43 = rho13 * rho13 * rho13 * rho13;
    double xs = two13 * sqrt(grho) / rho43;
    double xs2 = xs * xs;
    double sa2b8 = sqrt(1.0 + xs2);
    double shm1 = log(xs + sa2b8);
    double dd = 1.0 + 6.0 * beta * xs * shm1;
    double dd2 = dd * dd;
    double ee = 6.0 * beta * xs2 / sa2b8 - 1.0;
    sx = two13 * grho / rho43 * (-beta / dd);
    v1x = -(4.0 / 3.0) / two13 * xs2 * beta * rho13 * ee / dd2;
    v2x = two13 * beta * (ee - dd) / (rho43 * dd2);
}

__device__ inline void d_ggax(const double rho, const double grho,
                               double& sx, double& v1x, double& v2x)
{
    const double f1 = 0.196450, f2 = 7.79560, f3 = 0.27430;
    const double f4 = 0.15080, f5 = 0.0040;
    const double fp1 = -0.0192920212964260;
    const double fp2 = 0.1616204596739950;

    double rhom43 = pow(rho, -4.0 / 3.0);
    double s = fp2 * sqrt(grho) * rhom43;
    double s2 = s * s, s3 = s2 * s, s4 = s2 * s2;
    double exps = f4 * exp(-100.0 * s2);
    double as = f3 - exps - f5 * s2;
    double sa2b8 = sqrt(1.0 + f2 * f2 * s2);
    double shm1 = log(f2 * s + sa2b8);
    double bs = 1.0 + f1 * s * shm1 + f5 * s4;
    double das = (200.0 * exps - 2.0 * f5) * s;
    double dbs = f1 * (shm1 + f2 * s / sa2b8) + 4.0 * f5 * s3;
    double dls = das / as - dbs / bs;
    sx = fp1 * grho * rhom43 * as / bs;
    v1x = -4.0 / 3.0 * sx / rho * (1.0 + s * dls);
    v2x = fp1 * rhom43 * as / bs * (2.0 + s * dls);
}

__device__ inline void d_pbex(const double rho, const double grho, const int iflag,
                               double& sx, double& v1x, double& v2x)
{
    const double third = 1.0 / 3.0;
    const double pi = 3.14159265358979323846;
    const double c1 = 0.750 / pi;
    const double c2 = 3.0936677262801360;
    const double c5 = 4.0 / 3.0;
    const double k_arr[3] = {0.8040, 1.24500, 0.8040};
    const double mu_arr[3] = {0.2195149727645171, 0.2195149727645171, 0.12345679012345679};

    double agrho = sqrt(grho);
    double kf = c2 * pow(rho, third);
    double dsg = 0.5 / kf;
    double s1 = agrho * dsg / rho;
    double s2 = s1 * s1;
    double ds = -c5 * s1;
    double f1 = s2 * mu_arr[iflag] / k_arr[iflag];
    double f2 = 1.0 + f1;
    double f3 = k_arr[iflag] / f2;
    double fx = k_arr[iflag] - f3;
    double exunif = -c1 * kf;
    sx = exunif * fx;
    double dxunif = exunif * third;
    double dfx1 = f2 * f2;
    double dfx = 2.0 * mu_arr[iflag] * s1 / dfx1;
    v1x = sx + dxunif * fx + exunif * dfx * ds;
    v2x = exunif * dfx * dsg / agrho;
    sx = sx * rho;
}

__device__ inline void d_wcx(const double rho, const double grho,
                              double& sx, double& v1x, double& v2x)
{
    const double third = 1.0 / 3.0;
    const double pi = 3.14159265358979323846;
    const double c1 = 0.75 / pi;
    const double c2 = 3.093667726280136;
    const double c5 = 4.0 * third;
    const double teneightyone = 0.123456790123;
    const double k = 0.804, mu = 0.2195149727645171, cwc = 0.00793746933516;

    double agrho = sqrt(grho);
    double kf = c2 * pow(rho, third);
    double dsg = 0.5 / kf;
    double s1 = agrho * dsg / rho;
    double s2 = s1 * s1;
    double es2 = exp(-s2);
    double ds = -c5 * s1;
    double x1 = teneightyone * s2;
    double x2 = (mu - teneightyone) * s2 * es2;
    double x3 = log(1.0 + cwc * s2 * s2);
    double f1 = (x1 + x2 + x3) / k;
    double f2 = 1.0 + f1;
    double f3 = k / f2;
    double fx = k - f3;
    double exunif = -c1 * kf;
    sx = exunif * fx;
    double dxunif = exunif * third;
    double dfx1 = f2 * f2;
    double dxds1 = teneightyone;
    double dxds2 = (mu - teneightyone) * es2 * (1.0 - s2);
    double dxds3 = 2.0 * cwc * s2 / (1.0 + cwc * s2 * s2);
    double dfx = 2.0 * s1 * (dxds1 + dxds2 + dxds3) / dfx1;
    v1x = sx + dxunif * fx + exunif * dfx * ds;
    v2x = exunif * dfx * dsg / agrho;
    sx = sx * rho;
}

// ============================================================
// GGA Exchange device functions (spin-polarized)
// ============================================================

__device__ inline void d_becke88_spin(const double rho, const double grho,
                                       double& sx, double& v1x, double& v2x)
{
    const double beta = 0.00420;
    const double third = 1.0 / 3.0;

    double rho13 = pow(rho, third);
    double rho43 = rho13 * rho13 * rho13 * rho13;
    double xs = sqrt(grho) / rho43;
    double xs2 = xs * xs;
    double sa2b8 = sqrt(1.0 + xs2);
    double shm1 = log(xs + sa2b8);
    double dd = 1.0 + 6.0 * beta * xs * shm1;
    double dd2 = dd * dd;
    double ee = 6.0 * beta * xs2 / sa2b8 - 1.0;
    sx = grho / rho43 * (-beta / dd);
    v1x = -(4.0 / 3.0) * xs2 * beta * rho13 * ee / dd2;
    v2x = beta * (ee - dd) / (rho43 * dd2);
}

// ============================================================
// GGA Correlation device functions (unpolarized)
// ============================================================

__device__ inline void d_perdew86(const double rho, const double grho,
                                   double& sc, double& v1c, double& v2c)
{
    const double p1 = 0.0232660, p2 = 7.389e-6, p3 = 8.7230, p4 = 0.4720;
    const double pc1 = 0.0016670, pc2 = 0.0025680;
    const double pci = pc1 + pc2;
    const double third = 1.0 / 3.0;
    const double pi34 = 0.62035049089940;

    double rho13 = pow(rho, third);
    double rho43 = rho13 * rho13 * rho13 * rho13;
    double rs = pi34 / rho13;
    double rs2 = rs * rs, rs3 = rs * rs2;
    double cna = pc2 + p1 * rs + p2 * rs2;
    double cnb = 1.0 + p3 * rs + p4 * rs2 + 1.e4 * p2 * rs3;
    double cn = pc1 + cna / cnb;
    double drs = -third * pi34 / rho43;
    double dcna = (p1 + 2.0 * p2 * rs) * drs;
    double dcnb = (p3 + 2.0 * p4 * rs + 3.e4 * p2 * rs2) * drs;
    double dcn = dcna / cnb - cna / (cnb * cnb) * dcnb;
    double phi = 0.1920 * pci / cn * sqrt(grho) * pow(rho, -7.0 / 6.0);
    double ephi = exp(-phi);
    sc = grho / rho43 * cn * ephi;
    v1c = sc * ((1.0 + phi) * dcn / cn - (4.0 / 3.0 - 7.0 / 6.0 * phi) / rho);
    v2c = cn * ephi / rho43 * (2.0 - phi);
}

__device__ inline void d_ggac(const double rho, const double grho,
                               double& sc, double& v1c, double& v2c)
{
    const double al = 0.090, pa = 0.0232660, pb = 7.389e-6;
    const double pc = 8.7230, pd = 0.4720;
    const double cx = -0.0016670, cxc0 = 0.0025680;
    const double cc0 = -cx + cxc0;
    const double third = 1.0 / 3.0;
    const double pi34 = 0.62035049089940;
    const double nu = 15.7559203494831440;
    const double be = nu * cc0;
    const double xkf = 1.9191582926775130;
    const double xks = 1.1283791670955130;

    double rs = pi34 / pow(rho, third);
    double rs2 = rs * rs, rs3 = rs * rs2;
    double ec, vc;
    d_pw(rs, 0, ec, vc);
    double kf = xkf / rs;
    double ks = xks * sqrt(kf);
    double t = sqrt(grho) / (2.0 * ks * rho);
    double expe = exp(-2.0 * al * ec / (be * be));
    double af = 2.0 * al / be * (1.0 / (expe - 1.0));
    double bf = expe * (vc - ec);
    double y = af * t * t;
    double xy = (1.0 + y) / (1.0 + y + y * y);
    double x = 1.0 + y + y * y;
    double qy = y * y * (2.0 + y) / (x * x);
    double s1 = 1.0 + 2.0 * al / be * t * t * xy;
    double h0 = be * be / (2.0 * al) * log(s1);
    double dh0 = be * t * t / s1 * (-7.0 / 3.0 * xy - qy * (af * bf / be - 7.0 / 3.0));
    double ddh0 = be / (2.0 * ks * ks * rho) * (xy - qy) / s1;
    x = ks / kf * t;
    double ee = -100.0 * (x * x);
    double cna = cxc0 + pa * rs + pb * rs2;
    double dcna = pa * rs + 2.0 * pb * rs2;
    double cnb = 1.0 + pc * rs + pd * rs2 + 1.e4 * pb * rs3;
    double dcnb = pc * rs + 2.0 * pd * rs2 + 3.e4 * pb * rs3;
    double cn = cna / cnb - cx;
    double dcn = dcna / cnb - cna * dcnb / (cnb * cnb);
    double h1 = nu * (cn - cc0 - 3.0 / 7.0 * cx) * t * t * exp(ee);
    double dh1 = -third * (h1 * (7.0 + 8.0 * ee) + nu * t * t * exp(ee) * dcn);
    double ddh1 = 2.0 * h1 * (1.0 + ee) * rho / grho;
    sc = rho * (h0 + h1);
    v1c = h0 + h1 + dh0 + dh1;
    v2c = ddh0 + ddh1;
}

__device__ inline void d_pbec(const double rho, const double grho, const int iflag,
                               double& sc, double& v1c, double& v2c)
{
    const double ga = 0.0310906908696548950;
    const double be[2] = {0.06672455060314922, 0.046};
    const double third = 1.0 / 3.0;
    const double pi34 = 0.62035049089940;
    const double xkf = 1.9191582926775130;
    const double xks = 1.1283791670955130;

    double rs = pi34 / pow(rho, third);
    double ec, vc;
    d_pw(rs, 0, ec, vc);
    double kf = xkf / rs;
    double ks = xks * sqrt(kf);
    double t = sqrt(grho) / (2.0 * ks * rho);
    double expe = exp(-ec / ga);
    double af = be[iflag] / ga * (1.0 / (expe - 1.0));
    double bf = expe * (vc - ec);
    double y = af * t * t;
    double xy = (1.0 + y) / (1.0 + y + y * y);
    double x = 1.0 + y + y * y;
    double qy = y * y * (2.0 + y) / (x * x);
    double s1 = 1.0 + be[iflag] / ga * t * t * xy;
    double h0 = ga * log(s1);
    double dh0 = be[iflag] * t * t / s1 * (-7.0 / 3.0 * xy - qy * (af * bf / be[iflag] - 7.0 / 3.0));
    double ddh0 = be[iflag] / (2.0 * ks * ks * rho) * (xy - qy) / s1;
    sc = rho * h0;
    v1c = h0 + dh0;
    v2c = ddh0;
}

__device__ inline void d_glyp(const double rho, const double grho,
                               double& sc, double& v1c, double& v2c)
{
    const double a = 0.049180, b = 0.1320, c = 0.25330, d = 0.3490;

    double rhom13 = pow(rho, -1.0 / 3.0);
    double om = exp(-c * rhom13) / (1.0 + d * rhom13);
    double xl = 1.0 + (7.0 / 3.0) * (c * rhom13 + d * rhom13 / (1.0 + d * rhom13));
    double ff = a * b * grho / 24.0;
    double rhom53 = rhom13 * rhom13 * rhom13 * rhom13 * rhom13;
    sc = ff * rhom53 * om * xl;
    double dom = -om * (c + d + c * d * rhom13) / (1.0 + d * rhom13);
    double x = 1.0 + d * rhom13;
    double dxl = (7.0 / 3.0) * (c + d + 2.0 * c * d * rhom13 + c * d * d * rhom13 * rhom13) / (x * x);
    double rhom43 = rhom13 * rhom13 * rhom13 * rhom13;
    v1c = -ff * rhom43 / 3.0 * (5.0 * rhom43 * om * xl + rhom53 * dom * xl + rhom53 * om * dxl);
    v2c = 2.0 * sc / grho;
}

// ============================================================
// GGA Correlation device functions (spin-polarized)
// ============================================================

__device__ inline void d_perdew86_spin(const double rho, const double zeta, const double grho,
                                        double& sc, double& v1cup, double& v1cdw, double& v2c)
{
    const double p1 = 0.0232660, p2 = 7.389e-6, p3 = 8.7230, p4 = 0.4720;
    const double pc1 = 0.0016670, pc2 = 0.0025680;
    const double pci = pc1 + pc2;
    const double third = 1.0 / 3.0;
    const double pi34 = 0.62035049089940;

    double rho13 = pow(rho, third);
    double rho43 = rho13 * rho13 * rho13 * rho13;
    double rs = pi34 / rho13;
    double rs2 = rs * rs, rs3 = rs * rs2;
    double cna = pc2 + p1 * rs + p2 * rs2;
    double cnb = 1.0 + p3 * rs + p4 * rs2 + 1.e4 * p2 * rs3;
    double cn = pc1 + cna / cnb;
    double drs = -third * pi34 / rho43;
    double dcna = (p1 + 2.0 * p2 * rs) * drs;
    double dcnb = (p3 + 2.0 * p4 * rs + 3.e4 * p2 * rs2) * drs;
    double dcn = dcna / cnb - cna / (cnb * cnb) * dcnb;
    double phi = 0.1920 * pci / cn * sqrt(grho) * pow(rho, -7.0 / 6.0);
    double dd = pow(2.0, third) * sqrt(pow((1.0 + zeta) * 0.5, 5.0 / 3.0) + pow((1.0 - zeta) * 0.5, 5.0 / 3.0));
    double ddd = pow(2.0, -4.0 / 3.0) * 5.0 * (pow((1.0 + zeta) * 0.5, 2.0 / 3.0) - pow((1.0 - zeta) * 0.5, 2.0 / 3.0)) / (3.0 * dd);
    double ephi = exp(-phi);
    sc = grho / rho43 * cn * ephi / dd;
    v1cup = sc * ((1.0 + phi) * dcn / cn - (4.0 / 3.0 - 7.0 / 6.0 * phi) / rho) - sc * ddd / dd * (1.0 - zeta) / rho;
    v1cdw = sc * ((1.0 + phi) * dcn / cn - (4.0 / 3.0 - 7.0 / 6.0 * phi) / rho) + sc * ddd / dd * (1.0 + zeta) / rho;
    v2c = cn * ephi / rho43 * (2.0 - phi) / dd;
}

__device__ inline void d_pbec_spin(const double rho, const double zeta, const double grho,
                                    const int iflag,
                                    double& sc, double& v1cup, double& v1cdw, double& v2c)
{
    const double ga = 0.0310910;
    const double be[3] = {0.0, 0.06672455060314922, 0.0460000};
    const double third = 1.0 / 3.0;
    const double pi34 = 0.62035049089940;
    const double xkf = 1.9191582926775130;
    const double xks = 1.1283791670955130;

    double rs = pi34 / pow(rho, third);
    double ec, vcup, vcdw;
    d_pw_spin(rs, zeta, ec, vcup, vcdw);
    double kf = xkf / rs;
    double ks = xks * sqrt(kf);
    double fz = 0.5 * (pow(1.0 + zeta, 2.0 / 3.0) + pow(1.0 - zeta, 2.0 / 3.0));
    double fz2 = fz * fz, fz3 = fz2 * fz;
    double dfz = (pow(1.0 + zeta, -1.0 / 3.0) - pow(1.0 - zeta, -1.0 / 3.0)) / 3.0;
    double t = sqrt(grho) / (2.0 * fz * ks * rho);
    double expe = exp(-ec / (fz3 * ga));
    double af = be[iflag] / ga * (1.0 / (expe - 1.0));
    double bfup = expe * (vcup - ec) / fz3;
    double bfdw = expe * (vcdw - ec) / fz3;
    double y = af * t * t;
    double xy = (1.0 + y) / (1.0 + y + y * y);
    double qy = y * y * (2.0 + y) / ((1.0 + y + y * y) * (1.0 + y + y * y));
    double s1 = 1.0 + be[iflag] / ga * t * t * xy;
    double h0 = fz3 * ga * log(s1);
    double dh0up = be[iflag] * t * t * fz3 / s1 * (-7.0 / 3.0 * xy - qy * (af * bfup / be[iflag] - 7.0 / 3.0));
    double dh0dw = be[iflag] * t * t * fz3 / s1 * (-7.0 / 3.0 * xy - qy * (af * bfdw / be[iflag] - 7.0 / 3.0));
    double dh0zup = (3.0 * h0 / fz - be[iflag] * t * t * fz2 / s1 * (2.0 * xy - qy * (3.0 * af * expe * ec / fz3 / be[iflag] + 2.0))) * dfz * (1.0 - zeta);
    double dh0zdw = -(3.0 * h0 / fz - be[iflag] * t * t * fz2 / s1 * (2.0 * xy - qy * (3.0 * af * expe * ec / fz3 / be[iflag] + 2.0))) * dfz * (1.0 + zeta);
    double ddh0 = be[iflag] * fz / (2.0 * ks * ks * rho) * (xy - qy) / s1;
    sc = rho * h0;
    v1cup = h0 + dh0up + dh0zup;
    v1cdw = h0 + dh0dw + dh0zdw;
    v2c = ddh0;
}

// ============================================================
// GGA dispatch: gcxc for nspin=1
// ============================================================

__device__ inline void d_gcxc(const int* func_ids, const int nfunc,
                               const double rho, const double grho,
                               double& sxc, double& v1xc, double& v2xc,
                               const double hybrid_alpha)
{
    const double small = 1.0e-6;
    const double smallg = 1.0e-10;
    sxc = v1xc = v2xc = 0.0;
    if (rho <= small || grho < smallg) return;

    for (int i = 0; i < nfunc; ++i)
    {
        double s = 0.0, v1 = 0.0, v2 = 0.0;
        switch (func_ids[i])
        {
            case 106: // XC_GGA_X_B88
                d_becke88(rho, grho, s, v1, v2); break;
            case 109: // XC_GGA_X_PW91
                d_ggax(rho, grho, s, v1, v2); break;
            case 101: // XC_GGA_X_PBE
                d_pbex(rho, grho, 0, s, v1, v2); break;
            case 117: // XC_GGA_X_PBE_R
                d_pbex(rho, grho, 1, s, v1, v2); break;
            case 116: // XC_GGA_X_PBE_SOL
                d_pbex(rho, grho, 2, s, v1, v2); break;
            case 118: // XC_GGA_X_WC
                d_wcx(rho, grho, s, v1, v2); break;
            case 406: // XC_HYB_GGA_XC_PBEH (PBE0)
            {
                double sx2, v1x2, v2x2, sc2, v1c2, v2c2;
                d_pbex(rho, grho, 0, sx2, v1x2, v2x2);
                sx2 *= (1.0 - hybrid_alpha);
                v1x2 *= (1.0 - hybrid_alpha);
                v2x2 *= (1.0 - hybrid_alpha);
                d_pbec(rho, grho, 0, sc2, v1c2, v2c2);
                s = sx2 + sc2; v1 = v1x2 + v1c2; v2 = v2x2 + v2c2;
                break;
            }
            case 132: // XC_GGA_C_P86
                d_perdew86(rho, grho, s, v1, v2); break;
            case 134: // XC_GGA_C_PW91
                d_ggac(rho, grho, s, v1, v2); break;
            case 130: // XC_GGA_C_PBE
                d_pbec(rho, grho, 0, s, v1, v2); break;
            case 133: // XC_GGA_C_PBE_SOL
                d_pbec(rho, grho, 1, s, v1, v2); break;
            case 131: // XC_GGA_C_LYP
                d_glyp(rho, grho, s, v1, v2); break;
            default:
                break;
        }
        sxc += s; v1xc += v1; v2xc += v2;
    }
}

// ============================================================
// GGA dispatch: gcx_spin + gcc_spin for nspin=2
// ============================================================

__device__ inline void d_gcx_spin(const int func_id0, const double rhoup, const double rhodw,
                                   const double grhoup2, const double grhodw2,
                                   double& sx, double& v1xup, double& v1xdw,
                                   double& v2xup, double& v2xdw,
                                   const double hybrid_alpha)
{
    const double small = 1.0e-10;
    double sxup = 0.0, sxdw = 0.0;
    sx = 0.0; v1xup = 0.0; v1xdw = 0.0; v2xup = 0.0; v2xdw = 0.0;

    double rho = rhoup + rhodw;
    if (rho <= small) return;

    switch (func_id0)
    {
        case 106: // B88
            if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                d_becke88_spin(rhoup, grhoup2, sxup, v1xup, v2xup);
            if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                d_becke88_spin(rhodw, grhodw2, sxdw, v1xdw, v2xdw);
            break;
        case 101: // PBE
            if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                d_pbex(2.0 * rhoup, 4.0 * grhoup2, 0, sxup, v1xup, v2xup);
            if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                d_pbex(2.0 * rhodw, 4.0 * grhodw2, 0, sxdw, v1xdw, v2xdw);
            break;
        case 117: // revPBE
            if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                d_pbex(2.0 * rhoup, 4.0 * grhoup2, 1, sxup, v1xup, v2xup);
            if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                d_pbex(2.0 * rhodw, 4.0 * grhodw2, 1, sxdw, v1xdw, v2xdw);
            break;
        case 406: // PBE0
            if (rhoup > small && sqrt(fabs(grhoup2)) > small)
            {
                d_pbex(2.0 * rhoup, 4.0 * grhoup2, 0, sxup, v1xup, v2xup);
                sxup *= (1.0 - hybrid_alpha);
                v1xup *= (1.0 - hybrid_alpha);
                v2xup *= (1.0 - hybrid_alpha);
            }
            if (rhodw > small && sqrt(fabs(grhodw2)) > small)
            {
                d_pbex(2.0 * rhodw, 4.0 * grhodw2, 0, sxdw, v1xdw, v2xdw);
                sxdw *= (1.0 - hybrid_alpha);
                v1xdw *= (1.0 - hybrid_alpha);
                v2xdw *= (1.0 - hybrid_alpha);
            }
            break;
        case 116: // PBEsol
            if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                d_pbex(2.0 * rhoup, 4.0 * grhoup2, 2, sxup, v1xup, v2xup);
            if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                d_pbex(2.0 * rhodw, 4.0 * grhodw2, 2, sxdw, v1xdw, v2xdw);
            break;
        default:
            break;
    }
    sx = 0.5 * (sxup + sxdw);
    v2xup = 2.0 * v2xup;
    v2xdw = 2.0 * v2xdw;
}

__device__ inline void d_gcc_spin(const int func_id0, const int func_id1,
                                   const double rho, double zeta, const double grho,
                                   double& sc, double& v1cup, double& v1cdw, double& v2c,
                                   const double hybrid_alpha)
{
    const double small = 1.0e-10;
    const double epsr = 1.0e-6;
    sc = 0.0; v1cup = 0.0; v1cdw = 0.0; v2c = 0.0;

    if (fabs(zeta) - 1.0 > small || rho <= small || sqrt(fabs(grho)) <= small)
        return;

    double x = fmin(fabs(zeta), 1.0 - epsr);
    zeta = (zeta > 0.0) ? x : -x;

    if (func_id0 == 406) // PBE0
    {
        d_pbec_spin(rho, zeta, grho, 1, sc, v1cup, v1cdw, v2c);
        return;
    }

    switch (func_id1)
    {
        case 132: // P86
            d_perdew86_spin(rho, zeta, grho, sc, v1cup, v1cdw, v2c); break;
        case 130: // PBE
            d_pbec_spin(rho, zeta, grho, 1, sc, v1cup, v1cdw, v2c); break;
        case 133: // PBEsol
            d_pbec_spin(rho, zeta, grho, 2, sc, v1cup, v1cdw, v2c); break;
        default:
            break;
    }
}

} // namespace XC_GPU
#endif // XC_GGA_GPU_CUH
