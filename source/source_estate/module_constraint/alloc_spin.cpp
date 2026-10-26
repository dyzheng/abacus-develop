#include "source_estate/module_constraint/alloc_spin.h"

#include <stdexcept>
#include <string>

namespace alloc
{
namespace
{

void check_square(const std::vector<double>& A, const int n, const char* name)
{
    // Branch: every scalar Pauli component is a dense norb x norb matrix.
    if (n < 0 || static_cast<int>(A.size()) != n * n)
    {
        throw std::invalid_argument(std::string(name) + " has wrong size");
    }
}

double dense_trace_product(const std::vector<double>& A,
                           const std::vector<double>& B,
                           const int n)
{
    double value = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            value += A[static_cast<size_t>(i) * n + j]
                     * B[static_cast<size_t>(j) * n + i];
        }
    }
    return value;
}

std::complex<double> pauli_component(const ModuleBase::Vector3<double>& vector,
                                     const int row,
                                     const int col)
{
    // Branch: the diagonal entries carry lambda_z with opposite signs.
    if (row == col)
    {
        return std::complex<double>(row == 0 ? vector.z : -vector.z, 0.0);
    }
    // Branch: upper/lower off-diagonal entries are lambda_x -/+ i lambda_y.
    if (row == 0)
    {
        return std::complex<double>(vector.x, -vector.y);
    }
    return std::complex<double>(vector.x, vector.y);
}

} // namespace

void build_alloc_spin_kernel(const std::vector<double>& K,
                             const int norb,
                             const ModuleBase::Vector3<double>& lambda,
                             std::vector<std::complex<double>>& H)
{
    check_square(K, norb, "K");
    const int nspinor = 2 * norb;
    H.assign(static_cast<size_t>(nspinor) * nspinor,
             std::complex<double>(0.0, 0.0));
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            const double k = K[static_cast<size_t>(mu) * norb + nu];
            for (int s = 0; s < 2; ++s)
            {
                for (int t = 0; t < 2; ++t)
                {
                    H[static_cast<size_t>(2 * mu + s) * nspinor + 2 * nu + t]
                        = k * pauli_component(lambda, s, t);
                }
            }
        }
    }
}

void observe_alloc_spin_moments(
    const AllocConstraintAlgebra& space,
    const std::vector<std::vector<double>>& W,
    const std::vector<double>& rho0,
    const std::vector<double>& rhox,
    const std::vector<double>& rhoy,
    const std::vector<double>& rhoz,
    std::vector<double>& charges,
    std::vector<ModuleBase::Vector3<double>>& moments)
{
    const int norb = space.norb;
    // Guard: one weight matrix per constrained component is required to pair
    // an observable with its injected operator.
    if (static_cast<int>(W.size()) != space.nconstraint)
    {
        throw std::invalid_argument("weight-matrix count does not match constraints");
    }
    check_square(rho0, norb, "rho0");
    check_square(rhox, norb, "rhox");
    check_square(rhoy, norb, "rhoy");
    check_square(rhoz, norb, "rhoz");

    charges.assign(space.nconstraint, 0.0);
    moments.assign(space.nconstraint, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    for (int alpha = 0; alpha < space.nconstraint; ++alpha)
    {
        std::vector<double> K;
        build_alloc_constraint_kernel(space, W[alpha], alpha, K);
        charges[alpha] = dense_trace_product(rho0, K, norb);
        moments[alpha].x = dense_trace_product(rhox, K, norb);
        moments[alpha].y = dense_trace_product(rhoy, K, norb);
        moments[alpha].z = dense_trace_product(rhoz, K, norb);
    }
}

void observe_alloc_spinor_moments(const std::vector<double>& K,
                                  const int norb,
                                  const std::vector<std::complex<double>>& D,
                                  double& charge,
                                  ModuleBase::Vector3<double>& moment)
{
    check_square(K, norb, "K");
    const int nspinor = 2 * norb;
    // Guard: the spinor density uses the interleaved 2*norb basis.
    if (static_cast<int>(D.size()) != nspinor * nspinor)
    {
        throw std::invalid_argument("spinor density has wrong size");
    }

    charge = 0.0;
    moment = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            const double k = K[static_cast<size_t>(mu) * norb + nu];
            for (int s = 0; s < 2; ++s)
            {
                for (int t = 0; t < 2; ++t)
                {
                    const std::complex<double> d = D[
                        static_cast<size_t>(2 * nu + t) * nspinor + 2 * mu + s];
                    // Branch: the charge channel is the identity in spin
                    // space (only the diagonal s == t components contribute).
                    if (s == t)
                    {
                        charge += (k * d).real();
                    }
                    moment.x += (k * pauli_component(
                        ModuleBase::Vector3<double>(1.0, 0.0, 0.0), s, t) * d)
                                    .real();
                    moment.y += (k * pauli_component(
                        ModuleBase::Vector3<double>(0.0, 1.0, 0.0), s, t) * d)
                                    .real();
                    moment.z += (k * pauli_component(
                        ModuleBase::Vector3<double>(0.0, 0.0, 1.0), s, t) * d)
                                    .real();
                }
            }
        }
    }
}

void build_alloc_spin_s_adjoint(
    const AllocConstraintAlgebra& space,
    const std::vector<double>& S,
    const std::vector<std::vector<double>>& W,
    const std::vector<ModuleBase::Vector3<double>>& lambda,
    const std::vector<double>& rhox,
    const std::vector<double>& rhoy,
    const std::vector<double>& rhoz,
    std::vector<double>& E)
{
    const int norb = space.norb;
    // Guard: multipliers are paired one-to-one with kernels.
    if (static_cast<int>(W.size()) != space.nconstraint
        || static_cast<int>(lambda.size()) != space.nconstraint)
    {
        throw std::invalid_argument("spin-adjoint input count does not match constraints");
    }
    check_square(rhox, norb, "rhox");
    check_square(rhoy, norb, "rhoy");
    check_square(rhoz, norb, "rhoz");

    E.assign(static_cast<size_t>(norb) * norb, 0.0);
    for (int alpha = 0; alpha < space.nconstraint; ++alpha)
    {
        const ModuleBase::Vector3<double>& mu = lambda[alpha];
        // Branch: a zero spin multiplier contributes no derivative.
        if (mu.x == 0.0 && mu.y == 0.0 && mu.z == 0.0)
        {
            continue;
        }
        std::vector<double> D_alpha(static_cast<size_t>(norb) * norb, 0.0);
        for (size_t i = 0; i < D_alpha.size(); ++i)
        {
            D_alpha[i] = mu.x * rhox[i] + mu.y * rhoy[i] + mu.z * rhoz[i];
        }
        std::vector<double> E_alpha;
        build_alloc_constraint_s_adjoint(space, S, W[alpha], alpha, D_alpha,
                                         E_alpha);
        for (size_t i = 0; i < E.size(); ++i)
        {
            E[i] += E_alpha[i];
        }
    }
}

} // namespace alloc
