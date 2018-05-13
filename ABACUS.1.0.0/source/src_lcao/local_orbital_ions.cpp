#include "local_orbital_ions.h"
#include "src_pw/global.h"
#include "src_parallel/parallel_orbitals.h"
#include "src_lcao/sltk_atom_arrange.h"
#include "src_external/src_pdiag/pdiag_double.h"
#include "src_lcao/lcao_nnr.h"
#include "src_lcao/force_lcao.h"
#include "src_lcao/stress_lcao.h"
#include "src_lcao/istate_charge.h"
#include "src_lcao/istate_envelope.h"
#include "src_pw/vdwd2.h"
//#include "../src_siao/selinv.h" //mohan add 2012-05-13

Local_Orbital_Ions::Local_Orbital_Ions()
{}

Local_Orbital_Ions::~Local_Orbital_Ions() 
{}

void Local_Orbital_Ions::opt_ions(void)
{
    TITLE("Local_Orbital_Ions","opt_ions"); 
    timer::tick("Local_Orbital_Ions","opt_ions",'C'); 
		
	if(OUT_LEVEL=="i")
    {
        cout << setprecision(12);
        cout<< " " << setw(7)<< "ISTEP"
        <<setw(5)<< "NE"
        <<setw(18)<< "ETOT(eV)"
        <<setw(10)<< "dE(meV)"
        <<setw(10)<< "F(eV/A)"
        <<setw(10)<< "T(MIN)"
        <<endl;
    }

    // (3) Geometry optimization algorithm setup.
	if(FORCE)
	{
		//Ions_Move_Methods 
		IMM.allocate();
		//Charge_Extrapolation
		CE.allocate();
	}
    //add by zhengdy 2015/4/27 
        if(CALCULATION=="md")
	{
                              mdtype= INPUT.md_mdtype;//control md

                              if(mdtype==1||mdtype==2) {
                                      MDNVT.md_allocate();
                                      MDNVT.initMD();
                              }
                              else if(mdtype==0){
                                      MDNVE.md_allocate();
                                      MDNVE.initMD();
                              }
	}

	this->istep = 1;
	bool stop = false;
	while(istep <= NSTEP && !stop)
    {
		time_t estart = time(NULL);
		if(OUT_LEVEL=="ie" || OUT_LEVEL=="m") //xiaohui add "m" option, 2015-09-16
		{
			//add 2015-09-06, xiaohui
        		//cout << " -------------------------------------------" << endl;
			cout << " ---------------------------------------------------------" << endl;
			if(CALCULATION=="relax") 
			{
        			cout << " RELAX IONS : " << istep << endl;
				//cout << " ---------------------------------------------------------" << endl;
			}
			else if(CALCULATION=="scf")
			{
        			cout << " SELF-CONSISTENT : " << endl;
				//cout << " ---------------------------------------------------------" << endl;
			}
			else if(CALCULATION=="md") //xiaohui add 2015-09-15
			{
				if(mdtype==1||mdtype==2)
				{
    					cout<<" Molecular Dynamics (NVT) STEP "<< MDNVT.step_rst + istep<<endl;
				}
        			//cout << " STEP OF MOLECULAR DYNAMICS : " << istep << endl;
			}
        		//cout << " -------------------------------------------" << endl;
			cout << " ---------------------------------------------------------" << endl;

        		//ofs_running << " -------------------------------------------" << endl;
			ofs_running << " ---------------------------------------------------------" << endl;
			if(CALCULATION=="relax")
			{
        			ofs_running << " RELAX IONS : " << istep << endl;
				ofs_running << " ---------------------------------------------------------" << endl;
			}
			else if(CALCULATION=="scf")
			{
        			ofs_running << " SELF-CONSISTENT" << endl;
				ofs_running << " ---------------------------------------------------------" << endl;
			}
			//else if(CALCULATION=="md")
			//{
        		//	ofs_running << " STEP OF MOLECULAR DYNAMICS : " << istep << endl;
			//}
        		//ofs_running << " -------------------------------------------" << endl;
			//ofs_running << " ---------------------------------------------------------" << endl;

			/* delete 2015-09-06, xiaohui
			cout << " ---------------------------------------------------------" << endl;
			cout << " RELAX IONS : " << istep<< endl;
			cout << " ---------------------------------------------------------" << endl;
			ofs_running << " ---------------------------------------------------------" << endl;
			ofs_running << " RELAX IONS: " << istep<< endl;
			ofs_running << " ---------------------------------------------------------" << endl;
			*/
		}

        // (1) Find adjacent atoms for each atom.
    	atom_arrange::set_sr_NL();
        atom_arrange::search( SEARCH_RADIUS );
        //DONE(ofs_running,"SEARCH ADJACENT ATOMS");






        // (3) Periodic condition search
        // for each grid.
		// here 0 means GridT is not used for Vna.
		// mohan add Vna 2012-06-13
		// because it must be consistent with 
		// gtf in lcao_vna.
		// here VNA means considering the radius
		// cutoff VNA.
        GridT.set_pbc_grid(
		pw.ncx, pw.ncy, pw.ncz,
		pw.bx, pw.by, pw.bz,
		pw.nbx, pw.nby, pw.nbz,
		pw.nbxx, pw.nbzp_start, pw.nbzp, VNA);

		// (2) If k point is used here, allocate HlocR after atom_arrange.
		if(!GAMMA_ONLY_LOCAL)
		{
			// For each atom, calculate the adjacent atoms in different cells 
			// and allocate the space for H(R) and S(R).
			LNNR.cal_nnr();
			LM.allocate_HS_R(LNNR.nnr);
		}
		if(!GAMMA_ONLY_LOCAL)
		{
			// need to first calculae lgd.
			// using GridT.init.
			LNNR.cal_nnrg(GridT);
			//mohan add 2012-06-12
			//the pvnapR need nnrg.
			if(VNA>0)
			{
				UHM.GK.allocate_pvnapR();
			}
		}


		// (4) set the augmented orbitals index.
		// after ParaO and GridT, 
		// this information is used to calculate
		// the force.
		LOWF.set_trace_aug(GridT);
		
		// (5) init density kernel
		// (6) init wave functions.
    	if(GAMMA_ONLY_LOCAL)
		{
			// here we reset the density matrix dimension.
    		LOC.allocate_gamma(GridT);
			LOWF.aloc_gamma_wfc(GridT);
		}
		else
		{
			LOWF.allocate_k(GridT);
			LOC.allocate_DM_k();
		}

		//======================================
		// do the charge extrapolation before
		// the density matrix is regenerated.
		// mohan add 2011-04-08
		//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		// once the atom is move out of this processor,
		// the density matrix will not map
		// the 'moved' atom configuration,
		//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//======================================
		// HOWEVER, I ACTUALLY FOUND THIS IS 
		// A BUG, BECAUSE THE INDEX GridT.trace_lo
		// HAS BEEN REGENERATED, SO WE NEED TO
		// REALLOCATE THE DENSITY MATRIX FIRST,
		// THEN WE CAN READ IN DENSITY MATRIX,
		// AND USE DENSITY MATRIX TO DO 
		// RHO CALCULATION.
		// -- mohan 2013-03-31
		//======================================
		//if(pot.extra_pot==4 && istep>1)
		if(pot.extra_pot=="dm" && istep>1)//xiaohui modify 2015-02-01
		{
			for(int is=0; is<NSPIN; is++)
			{
				ZEROS(chr.rho[is], pw.nrxx);
				stringstream ssd;
				ssd << global_out_dir << "SPIN" << is + 1 << "_DM" ;
				// reading density matrix,
				LOC.read_dm(is, ssd.str() );
			}

			if(GAMMA_ONLY_LOCAL)
			{
				// and then construct the new density.
				UHM.GG.cal_rho();
			}
			else
			{
				UHM.GK.calculate_charge();	
			}
    		chr.renormalize_rho();
           	pot.init_pot( istep-1 );
		}

		// (9) S, T, Vnl, Vna matrix.
		UHM.set_ion();

		if(VdwD2::vdwD2)							//Peize Lin add 2014-04-04
		{
			VdwD2 vdw(ucell);
			vdw.energy();
		}
		
		// (10) self consistent
		if (CALCULATION=="scf" || CALCULATION=="md" || CALCULATION=="relax") //pengfei 2014-10-13
		{
			LOE.scf(istep-1);
		}
		else if (CALCULATION=="nscf")
		{
			LOE.nscf();
		}
		else if (CALCULATION=="istate")
		{
			IState_Charge ISC;
			ISC.begin();	
		}
		else if (CALCULATION=="ienvelope")
		{
			IState_Envelope IEP;
			IEP.begin();
		}
		else
		{
			WARNING_QUIT("Local_Orbital_Ions::opt_ions","What's the CALCULATION.");
		}
		time_t eend = time(NULL);

		//xiaohui add 2014-07-07, for second-order extrapolation
                int iat=0;
		//xiaohui modify 2015-09-30
    		//if(FORCE || CALCULATION=="md" )
    		if(CALCULATION=="relax"|| CALCULATION=="md" )
		{
                	for(int it = 0;it < ucell.ntype;it++)
                	{
                        	Atom* atom = &ucell.atoms[it];
                        	for(int ia =0;ia< ucell.atoms[it].na;ia++)
                        	{
                                	CE.pos_old2[3*iat  ] = CE.pos_old1[3*iat  ];
                                	CE.pos_old2[3*iat+1] = CE.pos_old1[3*iat+1];
                                	CE.pos_old2[3*iat+2] = CE.pos_old1[3*iat+2];

                                	CE.pos_old1[3*iat  ] = CE.pos_now[3*iat  ];
                                	CE.pos_old1[3*iat+1] = CE.pos_now[3*iat+1];
                                	CE.pos_old1[3*iat+2] = CE.pos_now[3*iat+2];

                                	CE.pos_now[3*iat  ] = atom->tau[ia].x*ucell.lat0;
                                	CE.pos_now[3*iat+1] = atom->tau[ia].y*ucell.lat0;
                                	CE.pos_now[3*iat+2] = atom->tau[ia].z*ucell.lat0;

                                	iat++;
                        	}
                	}
		}

		//2014-07-07, xiaohui
                //cout<<"CALCULATION:"<<CALCULATION<<endl;
                if(CALCULATION=="md")
                {  
                        if(mdtype==1||mdtype==2)   MDNVT.runnvt(istep);
                        else if(mdtype==0)  MDNVE.runNVE(istep);
                        if((mdtype==1||mdtype==2)&&(istep==NSTEP||stop))MDNVT.md_release();
                        if(mdtype==0&&(istep==NSTEP||stop))MDNVE.md_release();
  			//xiaohui move this line 2015-09-15
                        //cout<<"return to main function:"<<endl;

                        //MD.runMD(istep);//we need this total form
                }

		time_t fstart = time(NULL);
		if (CALCULATION=="scf" || CALCULATION=="relax")
		{
			stop = this->force_stress();
		}            
		time_t fend = time(NULL);

		//xiaohui add 2014-07-07, for second-order extrapolation
                iat=0;
    		if(FORCE || CALCULATION=="md" )
		{
                	for(int it = 0;it < ucell.ntype;it++)
                	{
                        	Atom* atom = &ucell.atoms[it];
                        	for(int ia =0;ia< ucell.atoms[it].na;ia++)
                        	{
                                	CE.pos_next[3*iat  ] = atom->tau[ia].x*ucell.lat0;
                                	CE.pos_next[3*iat+1] = atom->tau[ia].y*ucell.lat0;
                                	CE.pos_next[3*iat+2] = atom->tau[ia].z*ucell.lat0;

                                	iat++;
                        	}
                	}
		}

    		if(FORCE || CALCULATION=="md" )
		{
                	//xiaohui add CE.istep = istep 2014-07-07
                	CE.istep = istep;

                	// charge extrapolation if istep>0.
                	CE.extrapolate_charge();

			//if(pot.extra_pot==4)
			if(pot.extra_pot=="dm")//xiaohui modify 2015-02-01
                	{
                        	// done after grid technique.
                	}
                	else
                	{
                        	pot.init_pot( istep );
                	}
		}



		// need to destroy the <phi_0i | Vna | phi_Rj> matrix.
		if(!GAMMA_ONLY_LOCAL)
		{
			//mohan add 2012-06-12
			if(VNA>0)
			{
				UHM.GK.destroy_pvnapR();
			}
		}

        if(OUT_LEVEL=="i")
        {
            double etime_min = difftime(eend, estart)/60.0;
            double ftime_min = difftime(fend, fstart)/60.0;
            stringstream ss;
            ss << MOVE_IONS << istep;

			cout << setiosflags(ios::scientific)
            << " " << setw(7) << ss.str()
            << setw(5) << LOE.iter
            << setw(18) << setprecision(6) << en.etot * Ry_to_eV;

			cout << setprecision(2) << setiosflags(ios::scientific)
			<< setw(10) << IMM.get_ediff() * Ry_to_eV * 1000
			<< setw(10) << IMM.get_largest_grad() * Ry_to_eV / BOHR_TO_A;
            //<< setw(12) << IMM.get_trust_radius();

			cout << resetiosflags(ios::scientific)
//            << setw(8) << IMM.get_update_iter()
            << setprecision(2) << setw(10) << etime_min + ftime_min;
			cout << endl;
        }
		//xiaohui modifed 2013-07-22, adding "//" before ...
		//if(DIAGO_TYPE=="selinv")
		//{
		//	cout << " number of selected inversion: " << Selinv::niter_ion << endl;
		//	Selinv::niter_ion = 0;
		//}

//#ifdef __MPI //2015-09-06, xiaohui
	//2015-05-07, 2015-10-01
        //atom_arrange::delete_vector( SEARCH_RADIUS );
//#endif //2015-09-06, xiaohui

//2015-09-16
//#ifdef __MPI
//    MPI_Barrier(MPI_COMM_WORLD);
//    for (int i=0;i<ucell.ntype;i++)
//    {
//        ucell.atoms[i].bcast_atom(); // bcast tau array
//    }
//#endif

		++istep;
    }


	if(CALCULATION=="scf" || CALCULATION=="relax")
	{
		ofs_running << "\n\n --------------------------------------------" << endl;
		ofs_running << setprecision(16);
		ofs_running << " !FINAL_ETOT_IS " << en.etot * Ry_to_eV << " eV" << endl; 
		ofs_running << " --------------------------------------------\n\n" << endl;
	}

    hm.hon.clear_after_ions();

    timer::tick("Local_Orbital_Ions","opt_ions",'C'); 
    return;
}

bool Local_Orbital_Ions::force_stress(void)
{
    TITLE("Local_Orbital_Ions","force_stress");
    if(!FORCE && !STRESS)
    {
        return 1;
    }
	timer::tick("Local_Orbital_Ions","force_stress",'D');



	//return 0;

    if(FORCE)
    {
		//force_lo
		Force_LCAO FL; // init the class.
		FL.allocate (); 
		FL.start_force();

		if(STRESS)
		{
			FL.cal_stress();
		}

        // (2) move the ions according to
		// the algorithms of molecular dynamics.
		//if(CALCULATION=="md")
		//{
		//	md.init_md(istep, FL.fcs);
		//}
		// move the atoms according to CG or BFGS
		// methods.
		//else
		//{
			//---------- for test ---------------------
			//FL.fcs.zero_out(); // mohan test
			//ofstream ofs("tmp_force.txt");
			//ifstream ifs("tmp_force.txt");
			//for(int i=0; i<FL.fcs.nr*FL.fcs.nc; ++i)
			{
			//	ofs << FL.fcs.c[i] << endl;
			//	ifs >> FL.fcs.c[i];
			}
			//ofs.close();
			//ifs.close();
			//-----------------------------------------

#ifdef __MPI //2015-10-01, xiaohui
        atom_arrange::delete_vector( SEARCH_RADIUS );
#endif //2015-10-01, xiaohui

		//xiaohui add CALCULATION==relax 2015-09-30
                if(CALCULATION=="relax") IMM.cal_movement(istep, FL.fcs, en.etot);

        	if(IMM.get_converged())
        	{
            	return 1;
        	}
		//}

		// mohan update 2013-04-11
		// setup the structure factor
		// and do the density extraploation.
		// for both ionic iteratoin and
		// force calculations.

		//xiaohui modify 2014-08-09
		//pw.setup_structure_factor();

		// charge extrapolation if istep>0.
		//xiaohui modify 2014-08-09
		//CE.extrapolate_charge();

/*xiaohui modify 2014-08-09
		if(pot.extra_pot==4)
		{
			// done after grid technique.
		}
		else
		{
			pot.init_pot( istep );
		}
xiaohui modify 2014-08-09*/
	}

	timer::tick("Local_Orbital_Ions","force_stress",'D');
    return 0;
}
