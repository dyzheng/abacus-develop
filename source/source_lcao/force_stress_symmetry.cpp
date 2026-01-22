#include "FORCE_STRESS.h"

#include "source_base/mathzone.h"

template <typename T>
void Force_Stress_LCAO<T>::forceSymmetry(const UnitCell& ucell, ModuleBase::matrix& fcs, ModuleSymmetry::Symmetry* symm)
{
    double d1, d2, d3;
    for (int iat = 0; iat < ucell.nat; iat++)
    {
        ModuleBase::Mathzone::Cartesian_to_Direct(fcs(iat, 0), fcs(iat, 1), fcs(iat, 2),
          ucell.a1.x, ucell.a1.y, ucell.a1.z, ucell.a2.x, ucell.a2.y, ucell.a2.z,
          ucell.a3.x, ucell.a3.y, ucell.a3.z, d1, d2, d3);

        fcs(iat, 0) = d1;
        fcs(iat, 1) = d2;
        fcs(iat, 2) = d3;
    }
    symm->symmetrize_vec3_nat(fcs.c);
    for (int iat = 0; iat < ucell.nat; iat++)
    {
        ModuleBase::Mathzone::Direct_to_Cartesian(fcs(iat, 0), fcs(iat, 1), fcs(iat, 2),
          ucell.a1.x, ucell.a1.y, ucell.a1.z, ucell.a2.x, ucell.a2.y, ucell.a2.z,
          ucell.a3.x, ucell.a3.y, ucell.a3.z, d1, d2, d3);

        fcs(iat, 0) = d1;
        fcs(iat, 1) = d2;
        fcs(iat, 2) = d3;
    }
    return;
}

template class Force_Stress_LCAO<double>;
template class Force_Stress_LCAO<std::complex<double>>;

