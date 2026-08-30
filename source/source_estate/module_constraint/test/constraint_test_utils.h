#ifndef CONSTRAINT_TEST_UTILS_H
#define CONSTRAINT_TEST_UTILS_H

#include <array>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "source_base/global_variable.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_cell/setup_nonlocal.h"
#include "source_io/module_parameter/parameter.h"
#include "source_estate/magnetism.h"
#include "../../test/prepare_unitcell.h"
#undef private
#undef protected

// Local GlobalV defaults for a serial PW unit test.
inline void Set_GlobalV_Default()
{
    PARAM.input.device = "cpu";
    PARAM.input.precision = "double";
    PARAM.input.nspin = 1;
    PARAM.input.nelec = 10.0;
    PARAM.input.basis_type = "pw";
    GlobalV::KPAR = 1;
    GlobalV::NPROC_IN_POOL = 1;
}

// H2O in a 10 Bohr cubic box, O at the center, H's mirrored about x=5 Bohr.
inline std::unique_ptr<UnitCell> make_h2o_ucell()
{
    UcellTestPrepare utp(
        "cubic", 2, false, false, false, "None",
        1.0, // lat0 in Bohr; tau (Bohr) == Cartesian coordinates
        {20, 0, 0, 0, 20, 0, 0, 0, 20}, // 20 Bohr box
        {"O", "H"}, {"O.upf", "H.upf"}, {"upf201", "upf201"}, {"", ""},
        {1, 2}, {16.0, 1.0}, "Cartesian",
        {10.0, 10.0, 10.0, 11.2, 10.0, 10.0, 8.8, 10.0, 10.0}, // O, H1, H2 (Bohr)
        {1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    return utp.SetUcellInfo();
}

// Cartesian (Bohr) atomic positions, global atom order.
inline std::vector<std::array<double, 3>> h2o_positions(const UnitCell& ucell)
{
    std::vector<std::array<double, 3>> pos(ucell.nat);
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            const ModuleBase::Vector3<double> p =
                ucell.atoms[it].taud[ia] * ucell.latvec * ucell.lat0;
            pos[iat] = {p.x, p.y, p.z};
            ++iat;
        }
    }
    return pos;
}

#endif
