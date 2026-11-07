/**
 * @file mbis_alm_cube.cpp
 * @brief Run the standalone MBIS module on an ABACUS charge-density cube and
 *        cross-check the C++ implementation against the Python prototype.
 *
 * Usage:
 *   mbis_alm_cube chg.cube [--images N] [--symmetrize]
 *   mbis_alm_cube chgs1.cube chgs2.cube [--images N] [--symmetrize]
 *
 * Build (no ABACUS dependency):
 *   g++ -std=c++14 -O2 -I source tools/mbis_alm_cube.cpp \
 *       source/source_estate/module_mbis_alm/mbis_alm.cpp -o /tmp/mbis_alm_cube
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "source_estate/module_mbis_alm/mbis_alm.h"

using namespace mbis_alm;

namespace
{

struct Cube
{
    Grid grid;
    std::vector<Atom> atoms;
    std::vector<double> rho;
};

Cube read_cube(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
        lines.push_back(line);

    Cube c;
    {
        std::istringstream ss(lines[2]);
        int nat = 0;
        ss >> nat >> c.grid.origin.x >> c.grid.origin.y >> c.grid.origin.z;
        for (int i = 0; i < 3; ++i)
        {
            std::istringstream s2(lines[3 + i]);
            s2 >> c.grid.n[i] >> c.grid.dv[i].x >> c.grid.dv[i].y >> c.grid.dv[i].z;
        }
        c.atoms.resize(nat);
        for (int i = 0; i < nat; ++i)
        {
            std::istringstream s3(lines[6 + i]);
            double zatom = 0.0;
            s3 >> zatom >> c.atoms[i].Zval >> c.atoms[i].R.x >> c.atoms[i].R.y
               >> c.atoms[i].R.z;
        }
    }

    // data: all floats after the atom lines, first n0*n1*n2
    const std::size_t nxyz = c.grid.size();
    c.rho.assign(nxyz, 0.0);
    std::size_t k = 0;
    for (std::size_t i = 6 + c.atoms.size(); i < lines.size() && k < nxyz; ++i)
    {
        std::istringstream ss(lines[i]);
        double v = 0.0;
        while (ss >> v && k < nxyz)
            c.rho[k++] = v;
    }
    if (k != nxyz)
        std::fprintf(stderr, "warning: read %zu of %zu grid values\n", k, nxyz);
    return c;
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> cubes;
    int images = 0;
    bool symmetrize = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--images") == 0 && i + 1 < argc)
            images = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--symmetrize") == 0)
            symmetrize = true;
        else
            cubes.push_back(argv[i]);
    }
    if (cubes.empty())
    {
        std::fprintf(stderr, "usage: %s cube1 [cube2] [--images N] [--symmetrize]\n", argv[0]);
        return 2;
    }

    const Cube c1 = read_cube(cubes[0]);
    std::vector<double> rho = c1.rho;
    std::vector<double> spin;
    if (cubes.size() > 1)
    {
        const Cube c2 = read_cube(cubes[1]);
        spin.assign(rho.size(), 0.0);
        for (std::size_t i = 0; i < rho.size(); ++i)
        {
            spin[i] = rho[i] - c2.rho[i];
            rho[i] += c2.rho[i];
        }
    }

    MbisConfig cfg;
    cfg.cutoff = 1.0e6;      // do not truncate the real-space tail
    cfg.max_iter = 2000;
    cfg.tol = 1e-11;
    cfg.damp = 0.3;
    cfg.images = images;
    cfg.init_zeta = 1.0;
    cfg.symmetrize = symmetrize;

    const MbisState st = compute_mbis(rho, c1.grid, c1.atoms, cfg);
    const Observables obs = observe(st, rho, spin, c1.grid);

    double ne = 0.0, mtot = 0.0;
    const double dV = c1.grid.dV();
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        ne += rho[i] * dV;
        if (!spin.empty())
            mtot += spin[i] * dV;
    }
    std::printf("grid=%dx%dx%d dV=%.6e N_elec=%.6f\n", c1.grid.n[0], c1.grid.n[1],
                c1.grid.n[2], dV, ne);
    double sumN = 0.0, sumM = 0.0;
    for (std::size_t a = 0; a < c1.atoms.size(); ++a)
    {
        sumN += obs.N[a];
        sumM += obs.M[a];
        std::printf("atom %zu: N=%.6f netQ=%+.6f zeta=%.5f M=%+.6f\n", a, obs.N[a],
                    c1.atoms[a].Zval - obs.N[a], st.zeta[a], obs.M[a]);
    }
    std::printf("sum N=%.6f (ref %.6f)  sum M=%.6f (ref %.6f)\n", sumN, ne, sumM, mtot);
    return 0;
}
