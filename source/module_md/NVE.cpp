#include "NVE.h"
#include "MD_func.h"
#include "../module_base/timer.h"

NVE::NVE(MD_parameters& MD_para_in, UnitCell_pseudo &unit_in) : Verlet(MD_para_in, unit_in){}

NVE::~NVE(){}

void NVE::setup()
{
    ModuleBase::TITLE("NVE", "setup");
    ModuleBase::timer::tick("NVE", "setup");
    
    Verlet::setup();

    ModuleBase::timer::tick("NVE", "setup");
}

void NVE::first_half()
{
    ModuleBase::TITLE("NVE", "first_half");
    ModuleBase::timer::tick("NVE", "first_half");

    Verlet::first_half();

    ModuleBase::timer::tick("NVE", "first_half");
}

void NVE::second_half()
{
    ModuleBase::TITLE("NVE", "second_half");
    ModuleBase::timer::tick("NVE", "second_half");

    Verlet::second_half();

    ModuleBase::timer::tick("NVE", "second_half");
}

void NVE::outputMD()
{
    Verlet::outputMD();
}

void NVE::write_restart()
{
    Verlet::write_restart();
}

void NVE::restart()
{
    Verlet::restart();
}