#include "source_base/module_grid/partition.h"
#include "source_base/constants.h"

#include <cmath>
#include <functional>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>

namespace Grid {
namespace Partition {

const double stratmann_a = 0.64;

double w_becke(
    int nR0,
    const double* drR,
    const double* dRR,
    int nR,
    const int* iR,
    int c
) {
    assert(nR > 0 && nR0 >= nR);
    std::vector<double> P(nR, 1.0);
    for (int i = 0; i < nR; ++i) {
        int I = iR[i];
        for (int j = i + 1; j < nR; ++j) {
            int J = iR[j];
            double mu = (drR[I] - drR[J]) / dRR[I*nR0 + J];
            double s = s_becke(mu);
            P[I] *= s;
            P[J] *= (1.0 - s); // s(-mu) = 1 - s(mu)
        }
    }
    return P[c] / std::accumulate(P.begin(), P.end(), 0.0);
}


namespace {

// Heteronuclear size-adjustment parameter a_ij (Becke 1988, Eq. 15).
// u_ij = (chi_ij - 1)/(chi_ij + 1) with chi_ij = radii[I]/radii[J];
// a_ij = u_ij/(u_ij^2 - 1) clipped to [-0.5, 0.5] so the switch-plane
// displacement stays bounded even for extreme radius ratios.
double becke_size_adjust(const double radii_I, const double radii_J) {
    if (radii_I == radii_J) {
        // Equal radii: no adjustment, mu' = mu.
        return 0.0;
    }
    const double chi = radii_I / radii_J;
    const double u = (chi - 1.0) / (chi + 1.0);
    const double a = u / (u * u - 1.0);
    return std::max(-0.5, std::min(0.5, a));
}

// Corrected switching coordinate mu'_ij for a center pair.
double mu_adjusted(const double mu, const double a) {
    return mu + a * (1.0 - mu * mu);
}

// Analytic derivative of s_becke (3rd-order iterated polynomial).
// s(mu) = 0.5*(1 - p3), p3 = p(p(p(mu))), p(x) = 0.5*x*(3 - x^2).
double s_becke_deriv(const double mu) {
    const double p1 = 0.5 * mu * (3.0 - mu * mu);
    const double p2 = 0.5 * p1 * (3.0 - p1 * p1);
    const double p3 = 0.5 * p2 * (3.0 - p2 * p2);
    const double pp1 = 1.5 * (1.0 - mu * mu);
    const double pp2 = 1.5 * (1.0 - p1 * p1);
    const double pp3 = 1.5 * (1.0 - p2 * p2);
    return -0.5 * pp3 * pp2 * pp1;
}

} // anonymous namespace

double s_becke(double mu) {
    /* 
     * Becke's iterated polynomials (3rd order)
     *
     * s(mu) = 0.5 * (1 - p(p(p(mu))))
     *
     * p(x) = 0.5 * x * (3 - x^2)
     *
     */
    double p = 0.5 * mu * (3.0 - mu*mu);
    p = 0.5 * p * (3.0 - p*p);
    p = 0.5 * p * (3.0 - p*p);
    return 0.5 * (1.0 - p);
}


double w_stratmann(
    int nR0,
    const double* drR,
    const double* dRR,
    const double* drR_thr,
    int nR,
    int* iR,
    int c
) {
    assert(nR > 0 && nR0 >= nR);
    int I = iR[c], J = 0;

    // If r falls within the exclusive zone of a center, return immediately.
    for (int j = 0; j < nR; ++j) {
        J = iR[j];
        if (drR[J] <= drR_thr[J]) {
            return static_cast<double>(I == J);
        }
    }

    // Even if the grid point does not fall within the exclusive zone of any
    // center, the normalized weight could still be 0 or 1, and this can be
    // figured out by examining the unnormalized weight alone.

    // Swap the grid center to the first position in iteration for convenience.
    // Restore the original order before return.
    std::swap(iR[0], iR[c]);

    std::vector<double> P(nR);
    for (int j = 1; j < nR; ++j) {
        J = iR[j];
        double mu = (drR[I] - drR[J]) / dRR[I*nR0 + J];
        P[j] = s_stratmann(mu);
    }
    P[0] = std::accumulate(P.begin() + 1, P.end(), 1.0,
                           std::multiplies<double>());

    if (P[0] == 0.0 || P[0] == 1.0) {
        std::swap(iR[0], iR[c]); // restore the original order
        return P[0];
    }

    // If it passes all the screening above, all unnormalized weights
    // have to be calculated in order to get the normalized weight.

    std::for_each(P.begin() + 1, P.end(), [](double& s) { s = 1.0 - s; });
    for (int i = 1; i < nR; ++i) {
        I = iR[i];
        for (int j = i + 1; j < nR; ++j) {
            J = iR[j];
            double mu = (drR[I] - drR[J]) / dRR[I*nR0 + J];
            double s = s_stratmann(mu);
            P[i] *= s;
            P[j] *= (1.0 - s); // s(-mu) = 1 - s(mu)
        }
    }

    std::swap(iR[0], iR[c]); // restore the original order
    return P[0] / std::accumulate(P.begin(), P.end(), 0.0);
}


double s_stratmann(double mu) {
    /*
     * Stratmann's piecewise cell function
     *
     * s(mu) = 0.5 * (1 - g(mu/a))
     *
     *        /             -1                          x <= -1
     *        |
     * g(x) = | (35x - 35x^3 + 21x^5 - 5x^7) / 16       |x| < 1
     *        |
     *        \             +1                          x >= +1
     *
     */
    double x = mu / stratmann_a;
    double x2 = x * x;
    double h = 0.0625 * x * (35 + x2 * (-35 + x2 * (21 - 5 * x2)));

    bool mid = std::abs(x) < 1;
    double g = !mid * (1 - 2 * std::signbit(x)) + mid * h;
    return 0.5 * (1.0 - g);
}




double w_becke_adjusted(
    int nR0,
    const double* drR,
    const double* dRR,
    const double* radii,
    int nR,
    const int* iR,
    int c
) {
    assert(nR > 0 && nR0 >= nR);
    std::vector<double> P(nR, 1.0);
    for (int i = 0; i < nR; ++i) {
        int I = iR[i];
        for (int j = i + 1; j < nR; ++j) {
            int J = iR[j];
            double mu = (drR[I] - drR[J]) / dRR[I*nR0 + J];
            const double a = becke_size_adjust(radii[I], radii[J]);
            const double mu_p = mu_adjusted(mu, a);
            double s = s_becke(mu_p);
            P[I] *= s;
            P[J] *= (1.0 - s); // s(-mu') = 1 - s(mu')
        }
    }
    return P[c] / std::accumulate(P.begin(), P.end(), 0.0);
}


void w_becke_adjusted_deriv(
    int nR0,
    const double* drR,
    const double* dRR,
    const double* radii,
    const double* eR,
    int nR,
    const int* iR,
    int c,
    int J,
    double* dw
) {
    assert(nR > 0 && nR0 >= nR);
    for (int d = 0; d < 3; ++d) {
        dw[d] = 0.0;
    }

    // Early exit: J does not participate in any pair product of the involved
    // centers, so every chain-rule term vanishes.
    bool involved = false;
    for (int i = 0; i < nR; ++i) {
        if (iR[i] == J) {
            involved = true;
            break;
        }
    }
    if (!involved) {
        return;
    }

    // Unnormalized weights P_A and their log-derivatives dlnP_A/dR_J[d].
    // P_A = prod over pairs (i,j), i<j, of the cell function of the pair
    // factor, so dlnP_A is the sum of the per-pair log-derivatives.
    std::vector<double> P(nR, 1.0);
    std::vector<double> dlnP(nR * 3, 0.0);
    for (int i = 0; i < nR; ++i) {
        int I = iR[i];
        for (int j = i + 1; j < nR; ++j) {
            int K = iR[j];
            const double d = dRR[I*nR0 + K];
            const double mu = (drR[I] - drR[K]) / d;
            const double a = becke_size_adjust(radii[I], radii[K]);
            const double mu_p = mu_adjusted(mu, a);
            const double s = s_becke(mu_p);
            const double sp = s_becke_deriv(mu_p);
            // dmu'/dmu = 1 - 2*a*mu
            const double dmup_dmu = 1.0 - 2.0 * a * mu;

            // Geometric derivative of mu w.r.t. R_J:
            //   dmu = (1/d)*(delta_{J,I}*e_I - delta_{J,K}*e_K)
            //         - (mu/d)*(R_I - R_K)/d  * (delta_{J,I} - delta_{J,K})
            // with (R_I - R_K) reconstructed from drR and eR as
            // drR[I]*eR[I] - drR[K]*eR[K].
            double dmu_dRJ[3] = {0.0, 0.0, 0.0};
            const double dII = (J == I) ? 1.0 : 0.0;
            const double dKK = (J == K) ? 1.0 : 0.0;
            const double del = dII - dKK;
            for (int dd = 0; dd < 3; ++dd) {
                // dRR geometric derivative: (R_I - R_K)_dd / d, built from
                // the direction cosines of the two centers.
                const double RIK_dd = (drR[I] * eR[3*I + dd] - drR[K] * eR[3*K + dd]) / d;
                dmu_dRJ[dd] = (dII * eR[3*I + dd] - dKK * eR[3*K + dd]) / d
                              - (mu / d) * del * RIK_dd;
            }

            // Per-pair contribution to the two involved centers.
            // Center i (index I) carries factor s(mu'), center j (K) carries
            // s(-mu') = 1 - s(mu').
            const double dmu_p_dRJ[3] = {
                dmup_dmu * dmu_dRJ[0],
                dmup_dmu * dmu_dRJ[1],
                dmup_dmu * dmu_dRJ[2]};
            for (int dd = 0; dd < 3; ++dd) {
                dlnP[i*3 + dd] += (sp / s) * dmu_p_dRJ[dd];
                dlnP[j*3 + dd] += (sp / (1.0 - s)) * (-dmu_p_dRJ[dd]);
            }
            P[i] *= s;
            P[j] *= (1.0 - s);
        }
    }

    // Normalized weight derivative:
    //   w_c = P_c / S,  S = sum_A P_A
    //   dw_c = (dP_c * S - P_c * sum_A dP_A) / S^2
    const double S = std::accumulate(P.begin(), P.end(), 0.0);
    for (int d = 0; d < 3; ++d) {
        double dPc = P[c] * dlnP[c*3 + d];
        double dS = 0.0;
        for (int A = 0; A < nR; ++A) {
            dS += P[A] * dlnP[A*3 + d];
        }
        dw[d] = (dPc * S - P[c] * dS) / (S * S);
    }
}

} // end of namespace Partition
} // end of namespace Grid
