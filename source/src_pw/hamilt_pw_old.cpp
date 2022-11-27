//obsolete code, not compiled
//please remove the globalc::hm

#include "../module_base/global_function.h"
#include "../module_base/global_variable.h"
#include "../src_parallel/parallel_reduce.h"
#include "global.h"
#include "hamilt_pw.h"
#include "../module_base/blas_connector.h"
#include "myfunc.h"
#include "../module_base/timer.h"

int Hamilt_PW::moved = 0;

Hamilt_PW::Hamilt_PW()
{
    hpsi = new std::complex<double>[1];
    spsi = new std::complex<double>[1];
}

Hamilt_PW::~Hamilt_PW()
{
    delete[] hpsi;
    delete[] spsi;
}