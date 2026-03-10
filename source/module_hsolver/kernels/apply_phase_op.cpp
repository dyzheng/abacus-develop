#include "apply_phase_op.h"
#include "module_base/constants.h"
#include <cmath>

namespace hsolver {

// CPU implementation for complex<float>
template <>
void apply_phase_op<std::complex<float>, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* d,
    const int nrxx,
    const int nx,
    const int ny,
    const int nz,
    const int startz,
    const float dk_x,
    const float dk_y,
    const float dk_z,
    std::complex<float>* porter)
{
    for (int ir = 0; ir < nrxx; ir++) {
        // Calculate grid indices from linear index
        const int iz = ir / (nx * ny);
        const int ixy = ir % (nx * ny);
        const int iy = ixy / nx;
        const int ix = ixy % nx;

        // Calculate fractional coordinates
        const float fx = static_cast<float>(ix) / nx;
        const float fy = static_cast<float>(iy) / ny;
        const float fz = static_cast<float>(iz + startz) / nz;

        // dk is in Cartesian coordinates (2pi/a units)
        // phase = 2π * dk · f where f is fractional coordinate
        float phase = dk_x * fx + dk_y * fy + dk_z * fz;
        phase *= ModuleBase::TWO_PI;

        // Apply phase factor: exp(i*phase)
        std::complex<float> phase_factor(std::cos(phase), std::sin(phase));
        porter[ir] *= phase_factor;
    }
}

// CPU implementation for complex<double>
template <>
void apply_phase_op<std::complex<double>, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* d,
    const int nrxx,
    const int nx,
    const int ny,
    const int nz,
    const int startz,
    const double dk_x,
    const double dk_y,
    const double dk_z,
    std::complex<double>* porter)
{
    for (int ir = 0; ir < nrxx; ir++) {
        // Calculate grid indices from linear index
        const int iz = ir / (nx * ny);
        const int ixy = ir % (nx * ny);
        const int iy = ixy / nx;
        const int ix = ixy % nx;

        // Calculate fractional coordinates
        const double fx = static_cast<double>(ix) / nx;
        const double fy = static_cast<double>(iy) / ny;
        const double fz = static_cast<double>(iz + startz) / nz;

        // dk is in Cartesian coordinates (2pi/a units)
        // phase = 2π * dk · f where f is fractional coordinate
        double phase = dk_x * fx + dk_y * fy + dk_z * fz;
        phase *= ModuleBase::TWO_PI;

        // Apply phase factor: exp(i*phase)
        std::complex<double> phase_factor(std::cos(phase), std::sin(phase));
        porter[ir] *= phase_factor;
    }
}

} // namespace hsolver
