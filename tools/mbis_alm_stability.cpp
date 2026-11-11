/** Real-cube MBIS stability: Picard/Newton fixed point + J spectral radius. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "mbis_cube_io.h"
#include "source_estate/module_mbis_alm/mbis_alm.h"
using namespace mbis_alm;
using mbis_cube::Cube;
int main(int argc, char** argv)
{
    std::vector<std::string> cubes; int images = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--images") == 0 && i + 1 < argc) images = std::atoi(argv[++i]);
        else cubes.push_back(argv[i]);
    }
    if (cubes.empty()) { std::fprintf(stderr, "usage: %s cube1 [cube2] [--images N]\n", argv[0]); return 2; }
    const Cube c1 = mbis_cube::read_cube(cubes[0]);
    std::vector<double> rho = c1.rho, spin;
    if (cubes.size() > 1)
    {
        const Cube c2 = mbis_cube::read_cube(cubes[1]);
        spin.assign(rho.size(), 0.0);
        for (std::size_t i = 0; i < rho.size(); ++i) { spin[i] = rho[i] - c2.rho[i]; rho[i] += c2.rho[i]; }
    }
    MbisConfig cfg; cfg.cutoff = 1e6; cfg.max_iter = 2000; cfg.tol = 1e-11;
    cfg.damp = 0.3; cfg.images = images; cfg.init_zeta = 1.0;
    const MbisState sp = compute_mbis(rho, c1.grid, c1.atoms, cfg);
    const MbisState sn = compute_mbis_newton_fp(rho, c1.grid, c1.atoms, cfg, 4);
    const Observables op = observe(sp, rho, spin, c1.grid);
    const Observables on = observe(sn, rho, spin, c1.grid);
    bool conv = false;
    const double rho_p = mbis_fixedpoint_spectral_radius(rho, c1.grid, c1.atoms, cfg, sp, &conv);
    const double norm2 = mbis_fixedpoint_spectral_norm(rho, c1.grid, c1.atoms, cfg, sp);
    std::printf("cosets=%d images=%d rho(J)=%.4f (conv=%d) ||J||_2=%.3f\n",
                mbis_coset_count(c1.grid, c1.atoms), images, rho_p, (int)conv, norm2);
    std::printf("atom | Picard N, M, zeta | Newton N, M, zeta\n");
    for (std::size_t a = 0; a < c1.atoms.size(); ++a)
        std::printf(" %zu   | %.5f %+.5f %.4f | %.5f %+.5f %.4f\n", a,
                    op.N[a], op.M[a], sp.zeta[a], on.N[a], on.M[a], sn.zeta[a]);
    std::printf("D_picard=%.6e  D_newton=%.6e\n", sp.D, sn.D);
    return 0;
}
