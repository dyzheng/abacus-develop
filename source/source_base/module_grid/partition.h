#ifndef GRID_PARTITION_H
#define GRID_PARTITION_H

namespace Grid {
namespace Partition {

enum class Type {
    Becke,
    Stratmann,
};

extern const double stratmann_a;

/**
 * @brief Becke's partition weight.
 *
 * This function computes the normalized Becke's partition weight
 * for a grid point associated with a selected set of centers, given
 * the grid point's distance to centers and inter-center distance.
 *
 * @param   nR0     Total number of centers given by drR & dRR.
 * @param   drR     Distance between the grid point and centers.
 * @param   dRR     Distance between centers. dRR[I*nR0 + J] is the
 *                  distance between center I and J.
 * @param   nR      Number of centers involved in the weight calculation.
 *                  nR <= nR0. Length of iR.
 * @param   iR      Indices of centers involved.
 *                  Each element is a distinctive index in [0, nR0).
 * @param   c       iR[c] is the index of the center whom this grid point
 *                  belongs to.
 *
 * Reference:
 * Becke, A. D. (1988).
 * A multicenter numerical integration scheme for polyatomic molecules.
 * The Journal of chemical physics, 88(4), 2547-2553.
 *
 */
double w_becke(
    int nR0,
    const double* drR,
    const double* dRR,
    int nR,
    const int* iR,
    int c
);

// Becke's cell function (iterated polynomial)
double s_becke(double mu);

/**
 * @brief Becke's partition weight with heteronuclear size adjustment.
 *
 * Same partition scheme as w_becke, but the switch midpoint between centers
 * I and J is shifted when their partition radii differ (Becke 1988 size
 * adjustment).  radii[I] is the covalent/partition radius of center I.
 *
 *   u_ij = (chi_ij - 1) / (chi_ij + 1),  chi_ij = radii[I] / radii[J]
 *   a_ij = clip(u_ij / (u_ij^2 - 1), -0.5, 0.5)
 *   mu'_ij = mu_ij + a_ij * (1 - mu_ij^2)
 *
 * mu'_ij replaces mu_ij in the cell function s_becke.  For equal radii the
 * correction vanishes (a_ij = 0) and w_becke_adjusted == w_becke exactly.
 *
 * @see w_becke
 */
double w_becke_adjusted(
    int nR0,
    const double* drR,
    const double* dRR,
    const double* radii,
    int nR,
    const int* iR,
    int c
);

/**
 * @brief Analytic position derivative of the adjusted Becke weight.
 *
 * Computes dw[0..2] = d w_c / d R_J (3 cartesian components), where J is a
 * global center index in [0, nR0).  eR[3*I+d] is the d-th cartesian
 * component of the unit vector pointing from the grid point toward center I
 * (direction cosine).  The chain rule runs through mu'_ij = mu_ij + a_ij
 * (1 - mu_ij^2), the geometric derivatives of drR and dRR, and the analytic
 * derivative of the iterated Becke polynomial.
 *
 * @see w_becke_adjusted
 */
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
);


/**
 * @brief Becke's partition weight with Stratmann's scheme.
 *
 * This function is similar to w_becke, but the cell function adopts
 * the one proposed by Stratmann et al, which enables some screening.
 *
 * @see w_becke
 *
 * @param   drR_thr     Radius of exclusive zone of each center.
 *
 * Reference:
 * Stratmann, R. E., Scuseria, G. E., & Frisch, M. J. (1996).
 * Achieving linear scaling in exchange-correlation density functional
 * quadratures.
 * Chemical physics letters, 257(3-4), 213-223.
 *
 */
double w_stratmann(
    int nR0,
    const double* drR,
    const double* dRR,
    const double* drR_thr,
    int nR,
    int* iR,
    int c
);

// Stratmann's piecewise cell function
double s_stratmann(double mu);

} // end of namespace Partition
} // end of namespace Grid

#endif
