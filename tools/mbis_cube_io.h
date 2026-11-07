#ifndef TOOLS_MBIS_CUBE_IO_H
#define TOOLS_MBIS_CUBE_IO_H

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "source_estate/module_mbis_alm/mbis_alm.h"

namespace mbis_cube
{

using mbis_alm::Atom;
using mbis_alm::Grid;

struct Cube
{
    Grid grid;
    std::vector<Atom> atoms;
    std::vector<double> rho;
};

inline Cube read_cube(const std::string& path)
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
    const std::size_t nxyz = c.grid.size();
    c.rho.assign(nxyz, 0.0);
    std::size_t k = 0;
    for (std::size_t i = 6 + c.atoms.size(); i < lines.size() && k < nxyz; ++i)
    {
        std::istringstream ss2(lines[i]);
        double v = 0.0;
        while (ss2 >> v && k < nxyz)
            c.rho[k++] = v;
    }
    return c;
}

} // namespace mbis_cube

#endif
