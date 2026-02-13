// Temporary mock implementations for testing
// Most functions are already implemented in the actual codebase
// This file provides minimal mocks for UnitCell and related classes

#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"

// constructor of Atom
Atom::Atom()
{
}
Atom::~Atom()
{
}

Atom_pseudo::Atom_pseudo()
{
}
Atom_pseudo::~Atom_pseudo()
{
}

Magnetism::Magnetism()
{
}
Magnetism::~Magnetism()
{
}

InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}

pseudo::pseudo()
{
}
pseudo::~pseudo()
{
}

// constructor of UnitCell
UnitCell::UnitCell()
{
}
UnitCell::~UnitCell()
{
}
SepPot::SepPot(){}
SepPot::~SepPot(){}
Sep_Cell::Sep_Cell() noexcept {}
Sep_Cell::~Sep_Cell() noexcept {}

void UnitCell::set_iat2iwt(const int& npol_in)
{
	this->iat2iwt.resize(this->nat);
	this->npol = npol_in;
	int iat=0;
	int iwt=0;
	for(int it = 0;it < this->ntype; it++)
	{
		for(int ia=0; ia<atoms[it].na; ia++)
		{
			this->iat2iwt[iat] = iwt;
			iwt += atoms[it].nw * this->npol;
			++iat;
		}
	}
	return;
}

// Mock ElecState virtual functions
namespace elecstate {
    const double* ElecState::getRho(int spin) const { return nullptr; }
}

// Mock LCAO_Orbitals
#include "source_basis/module_ao/ORB_read.h"
LCAO_Orbitals::LCAO_Orbitals() {}
LCAO_Orbitals::~LCAO_Orbitals() {}

// Mock Numerical_Orbital_Lm
#include "source_basis/module_ao/ORB_atomic_lm.h"
Numerical_Orbital_Lm::Numerical_Orbital_Lm() {}
Numerical_Orbital_Lm::~Numerical_Orbital_Lm() {}

// Mock ORB_gaunt_table
#include "source_basis/module_ao/ORB_gaunt_table.h"
ORB_gaunt_table::ORB_gaunt_table() {}
ORB_gaunt_table::~ORB_gaunt_table() {}

// Mock cal_r_overlap_R
#include "source_io/cal_r_overlap_R.h"
cal_r_overlap_R::cal_r_overlap_R() : ParaV(nullptr) {}
cal_r_overlap_R::~cal_r_overlap_R() {}
void cal_r_overlap_R::init(const UnitCell& ucell, const Parallel_Orbitals& pv, const LCAO_Orbitals& orb) {}
ModuleBase::Vector3<std::complex<double>> cal_r_overlap_R::get_psi_L_psi(
    const ModuleBase::Vector3<double>& R1,
    const int& T1, const int& L1, const int& m1, const int& N1,
    const ModuleBase::Vector3<double>& R2,
    const int& T2, const int& L2, const int& m2, const int& N2)
{
    return ModuleBase::Vector3<std::complex<double>>(0.0, 0.0, 0.0);
}

