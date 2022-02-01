#ifndef GENERAL_PARAMETERS_H
#define GENERAL_PARAMETERS_H

#include <string>
namespace ABACUS
{
    class General_parameters
    {
        public:
            General_parameters()
            {
                calculation = "scf";
                basis_type = "pw";
                symmetry = false;
                force = false;
                stress = false;
                dft_plus_u = false;
                vdw_method = "none";
                lspinorb = false;
                tddft = false;
                nelec = 0.0;
                nbands = 0;
                nbands_istate = 0;
                nspin = 1;
                nbspline = -1;
                suffix = "ABACUS";
                dft_functional = "none";
                pseudo_rcut = 15.0;
                soc_lambda = 1.0;
                mem_saver = false;
                init_charge = "atomic";
                init_wfc = "atomic";
                init_vel = false;
                init_mesh = false;
                pw_seed = 1;
                k_par = 1;
                ntype = 0;

            };
            ~General_parameters(){};

			/// "scf" : self consistent calculation.
			/// "nscf" : non-self consistent calculation.
			/// "relax" : cell relaxations, only ions can move. 
            /// "cell-relax" : cell relaxations, both ions and lattice vectors can move.
            /// "md" : molecular dynamics
			std::string calculation;
			/// "pw" : PW;
			/// "lcao_in_pw" : LCAO in pw;
			/// "lcao" :LCAO 
            std::string basis_type;
            /// turn symmetry on or off
			bool symmetry;
            /// calculate ionic force or not
            bool force;
            /// calculate lattice stress tensor or not
            bool stress;
            /// true:DFT+U correction; false：standard DFT calcullation(default)
            bool dft_plus_u;
            /// the method of calculating vdw (none ; d2 ; d3_0 ; d3_bj")
            std::string vdw_method;
            /// consider the spin-orbit interaction
            bool lspinorb;
            /// calculate tddft or not
            bool tddft;
            /// 1: single spin; 2: up and down spin; 4: noncollinear spin
            int nspin;
            /// input number of electrons
            double nelec;
            /// number of bands
            int nbands;
            /// number of bands around Fermi level for istate calulation
            int nbands_istate;
            /// atomic; first-order; second-order; dm:coefficients of SIA
            std::string charge_extrap;
            /// the name of lattice name
            std::string latname;
            /// the order of B-spline basis(>=0) if it is -1 (default), B-spline for Sturcture Factor isnot used.
            int nbspline;
            /// suffix of out put dir
            std::string suffix;
            /// exchange correlation functional
            std::string dft_functional;
            /// cut-off radius for radial integration
            double pseudo_rcut;
            /// memory saver for many k points used
            bool mem_saver;
            /// adjust indensity of SOC effect, from [0,1]
            double soc_lambda;
            /// start charge is from 'atomic' or file
            std::string init_charge;
            /// start wavefunction is from 'atomic' or file
            std::string init_wfc;
            /// read velocity from STRU or not
            bool init_vel;
            /// 0: use our own mesh to do radial renormalization; 1: use mesh as in QE
            bool init_mesh;
            /// random seed for initializing wave functions
            int pw_seed;
            /// number of pools for k points, pw only
            int k_par;
            /// atom species number
            int ntype;
    };

    class Input_path
    {
        public:
        Input_path()
        {
            kpoint_file = "";
            read_file_dir = "auto";
            stru_file = "";
        };
        ~Input_path(){};

        /// the name of file containing k points
        std::string kpoint_file;
        /// the filename of file containing atom positions
        std::string stru_file;
        /// directory of files for reading
        std::string read_file_dir;
    };
}

#endif
