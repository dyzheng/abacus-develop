#ifndef ABACUS_PARAMETER_H
#define ABACUS_PARAMETER_H

#include "module_md/MD_parameters.h"
#include "src_pw/vdwd2_parameters.h"
#include "src_pw/vdwd3_parameters.h"
#include "module_input/pw_parameters.h"
#include "module_input/lcao_parameters.h"
#include "module_input/scf_parameters.h"
#include "src_pw/exx_global.h"
#include "module_input/relax_parameters.h"
#include "module_input/output_parameters.h"
#include "module_input/elecf_parameters.h"
#include "module_input/general_parameters.h"
#include "module_input/dftu_parameters.h"
#include "module_input/berry_parameters.h"
#include "module_input/tddft_parameters.h"
#include "module_deepks/deepks_parameters.h"
#include "module_input/spectrum_parameters.h"
#include "module_input/test_parameters.h"

namespace ABACUS
{

    class ABACUS_parameters
    {
        public:
            ABACUS_parameters(){};
            ~ABACUS_parameters(){};

            /// 01 general
            General_parameters gp;
            /// 02 input path
            Input_path inputp;
            /// 03 plane wave base
            PW_parameters pwp;
            /// 04 lcao base
            LCAO_parameters lcaop;
            /// 05 scf
            SCF_parameters scfp;
            /// 06 relaxation
            Relax_parameters relaxp;
            /// 07 print
            Output_parameters outputp;
            /// 08 electric field
            Elecf_parameters elecfp;
            /// 09 exact exchange 
            Exx_Global::Exx_Info exxp;
            /// 10 molecular dynamics
            MD_parameters mdp;
            /// 11 LDA+U
            Dftu_parameters dftup;
            /// 12 vdw
            Vdwd2_Parameters vdwp2;
            Vdwd3_Parameters vdwp3;
            /// 13 berry
            Berry_parameters berryp;
            /// 14 TDDFT
            Tddft_parameters tddftp;
            /// 15 DeePKS
            Deepks_parameters deepksp;
            /// 16 test
            Test_parameters testp;
            /// 17 spectrum
            Spectrum_parameters spectp;
            /// for check parameter conflicts
            void global_check();
            /// for convert parameter among different parts
            void global_convert();
            
    };

    extern ABACUS_parameters para;
}

#endif