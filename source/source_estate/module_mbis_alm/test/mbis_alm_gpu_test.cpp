#include "source_estate/module_mbis_alm/mbis_alm_gpu.h"

#include "source_estate/module_mbis_alm/mbis_alm.h"
#include "source_estate/module_mbis_alm/pw_grid_bridge.h"

#include "gtest/gtest.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace mbis_alm;

namespace
{

struct Toy
{
    Grid grid;
    std::vector<Atom> atoms;
    std::vector<double> rho;
    std::vector<double> spin;
};

// Synthetic two-centre density (smooth Gaussians) on a cubic grid; mirrors the
// CPU unit-test fixture so the CPU and GPU fixed points are directly
// comparable.
Toy make_toy(int n, double spacing, int images)
{
    Toy t;
    t.grid.n = {{n, n, n}};
    t.grid.origin = Vec3{0.0, 0.0, 0.0};
    t.grid.dv = {{Vec3{spacing, 0.0, 0.0}, Vec3{0.0, spacing, 0.0}, Vec3{0.0, 0.0, spacing}}};
    const double c = 0.5 * static_cast<double>(n) * spacing;
    t.atoms = {{Vec3{c - 1.2, c, c}, 8.0}, {Vec3{c + 1.2, c, c}, 8.0}};
    const std::size_t nxyz = t.grid.size();
    t.rho.assign(nxyz, 0.0);
    t.spin.assign(nxyz, 0.0);
    for (int ix = 0; ix < n; ++ix)
        for (int iy = 0; iy < n; ++iy)
            for (int iz = 0; iz < n; ++iz)
            {
                const std::size_t ig = (static_cast<std::size_t>(ix) * n + iy) * n + iz;
                const Vec3 p = t.grid.point(ix, iy, iz);
                double rho = 0.0, spin = 0.0;
                for (int a = 0; a < 2; ++a)
                {
                    const Vec3 d = p - t.atoms[a].R;
                    const double g = std::exp(-dot(d, d) / (2.0 * 1.0 * 1.0));
                    rho += 8.0 * g;
                    spin += (a == 0 ? 1.0 : -1.0) * 4.0 * g;
                }
                t.rho[ig] = rho;
                t.spin[ig] = spin;
            }
    (void)images;
    return t;
}

MbisConfig cfg(bool use_gpu, int images)
{
    MbisConfig c;
    c.cutoff = 1.0e6; // no cutoff -> smooth reference
    c.max_iter = 400;
    c.tol = 1e-10;
    c.damp = 0.3;
    c.images = images;
    c.init_zeta = 1.0;
    c.use_gpu = use_gpu;
    return c;
}

double rel_err(double a, double b)
{
    const double d = std::abs(a - b);
    const double s = std::max(std::abs(a), std::abs(b));
    return (s > 0.0) ? d / s : d;
}

} // namespace

TEST(MbisGpu, BackendAvailable)
{
    // Informational: a CUDA build on a GPU-less node still passes the CPU
    // tests, it merely skips the comparison below.
    if (!mbis_gpu_available())
    {
        GTEST_SKIP() << "no GPU visible; device backend not exercised";
    }
    EXPECT_TRUE(mbis_gpu_available());
}

TEST(MbisGpu, MatchesCpuWithoutImages)
{
    if (!mbis_gpu_available())
    {
        GTEST_SKIP() << "no GPU visible";
    }
    Toy t = make_toy(40, 0.35, 0);
    const MbisState cpu = compute_mbis(t.rho, t.grid, t.atoms, cfg(false, 0));
    const MbisState gpu = compute_mbis(t.rho, t.grid, t.atoms, cfg(true, 0));

    ASSERT_EQ(cpu.N.size(), gpu.N.size());
    for (std::size_t a = 0; a < cpu.N.size(); ++a)
    {
        EXPECT_LT(rel_err(cpu.N[a], gpu.N[a]), 1e-8) << "N atom " << a;
        EXPECT_LT(rel_err(cpu.zeta[a], gpu.zeta[a]), 1e-8) << "zeta atom " << a;
    }
    EXPECT_LT(rel_err(cpu.D, gpu.D), 1e-8);
    double maxdw = 0.0;
    for (std::size_t a = 0; a < cpu.w.size(); ++a)
        for (std::size_t ig = 0; ig < cpu.w[a].size(); ++ig)
            maxdw = std::max(maxdw, std::abs(cpu.w[a][ig] - gpu.w[a][ig]));
    EXPECT_LT(maxdw, 1e-8);
}

TEST(MbisGpu, MatchesCpuWithImages)
{
    if (!mbis_gpu_available())
    {
        GTEST_SKIP() << "no GPU visible";
    }
    Toy t = make_toy(36, 0.5, 1);
    const MbisState cpu = compute_mbis(t.rho, t.grid, t.atoms, cfg(false, 1));
    const MbisState gpu = compute_mbis(t.rho, t.grid, t.atoms, cfg(true, 1));

    for (std::size_t a = 0; a < cpu.N.size(); ++a)
    {
        EXPECT_LT(rel_err(cpu.N[a], gpu.N[a]), 1e-8) << "N atom " << a;
        EXPECT_LT(rel_err(cpu.zeta[a], gpu.zeta[a]), 1e-8) << "zeta atom " << a;
    }
    EXPECT_LT(rel_err(cpu.D, gpu.D), 1e-8);
}

TEST(MbisGpu, ProductionLikeGridTiming)
{
    if (!mbis_gpu_available())
    {
        GTEST_SKIP() << "no GPU visible";
    }
    // 64^3 grid, 4 atoms, one image shell: representative of the production
    // WeightGrid density size (coarser than a 1e6-point FFT grid).
    Toy t = make_toy(64, 0.4, 1);
    t.atoms.push_back({Vec3{6.4, 10.4, 10.4}, 8.0});
    t.atoms.push_back({Vec3{10.4, 10.4, 10.4}, 8.0});

    MbisConfig c = cfg(false, 1);
    c.max_iter = 40;
    c.cutoff = 10.0; // production cutoff
    c.tol = 1e-9;

    auto t0 = std::chrono::steady_clock::now();
    const MbisState cpu = compute_mbis(t.rho, t.grid, t.atoms, c);
    auto t1 = std::chrono::steady_clock::now();

    c.use_gpu = true;
    auto t2 = std::chrono::steady_clock::now();
    const MbisState gpu = compute_mbis(t.rho, t.grid, t.atoms, c);
    auto t3 = std::chrono::steady_clock::now();

    const double cpu_s = std::chrono::duration<double>(t1 - t0).count();
    const double gpu_s = std::chrono::duration<double>(t3 - t2).count();
    std::printf("[mbis_gpu] 64^3 grid, 4 atoms, images=1: CPU %.4f s, GPU %.4f s"
                " (speedup %.2fx)\n",
                cpu_s,
                gpu_s,
                cpu_s / gpu_s);

    for (std::size_t a = 0; a < cpu.N.size(); ++a)
    {
        EXPECT_LT(rel_err(cpu.N[a], gpu.N[a]), 1e-7) << "N atom " << a;
    }
    // Timing is reported for the development log, not asserted: CI machines
    // vary wildly in core count/GPU model and a hard speed threshold would be
    // flaky.  Correctness on this production-like grid is the hard guarantee.
    EXPECT_GT(cpu_s, 0.0);
    EXPECT_GT(gpu_s, 0.0);
}
