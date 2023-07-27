#include "paw_atom.h"

void Paw_Atom::init_paw_atom(const int nproj_in)
{

    nproj = nproj_in;

    ca.resize(nproj);
    rhoij.resize(nproj*(nproj + 1) / 2);
    rhoijp.resize(nproj*(nproj + 1) / 2);
    rhoijselect.resize(nproj*(nproj + 1) / 2);

    this -> reset_rhoij();
}

void Paw_Atom::set_ca(std::vector<std::complex<double>> & ca_in, const double weight_in)
{
    for(int i = 0; i < nproj; i ++)
    {
        ca[i] = ca_in[i];
    }

    weight = weight_in;
}

void Paw_Atom::reset_rhoij()
{
    nrhoijsel = 0;
    for(int i = 0; i < nproj*(nproj+1)/2; i ++)
    {
        rhoij[i] = 0.0;
        rhoijp[i] = 0.0;
        rhoijselect[i] = -1;
    }    
}

void Paw_Atom::accumulate_rhoij()
{
    for(int iproj = 0; iproj < nproj; iproj ++)
    {
        int i0 = iproj * (iproj + 1) / 2;
        for(int jproj = 0; jproj < iproj+1; jproj ++)
        {
            std::complex<double> tmp = std::conj(ca[iproj]) * ca[jproj];
            rhoij[i0 + jproj] += tmp.real() * weight;
        }
    }
}

void Paw_Atom::convert_rhoij()
{
    nrhoijsel = 0;
    for(int i = 0; i < rhoij.size(); i ++)
    {
        if(std::abs(rhoij[i]) > 1e-10)
        {
            rhoijselect[nrhoijsel] = i;
            rhoijp[nrhoijsel] = rhoij[i];
            nrhoijsel ++;
        }
    }
}