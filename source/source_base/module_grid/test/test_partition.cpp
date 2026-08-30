#include "source_base/module_grid/partition.h"
#include "source_base/module_grid/radial.h"
#include "source_base/module_grid/delley.h"
#include "source_base/constants.h"

#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <numeric>
#include <chrono>
#include <random>
#include <algorithm>

#ifdef __MPI
#include <mpi.h>
#endif

using ModuleBase::PI;
using Vec3 = std::array<double, 3>;

using iclock = std::chrono::high_resolution_clock;
iclock::time_point start;
std::chrono::duration<double> dur;

double norm(const Vec3& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

Vec3 operator-(const Vec3& v1, const Vec3& v2) {
    return {v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]};
}

Vec3 operator+(const Vec3& v1, const Vec3& v2) {
    return {v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]};
}

// |r|^n * exp(-a*|r|^2)
double func_core(const Vec3& r, double a, double n) {
    double rabs = norm(r);
    return std::pow(rabs, n) * std::exp(-a * rabs * rabs);
}

// func_core integrated over all space
double ref_core(double a, double n) {
    double p = 0.5 * (n + 3);
    return 2.0 * PI * std::pow(a, -p) * std::tgamma(p);
}

// the test function is a combination of several func_core
double func(
    const Vec3& r,
    const std::vector<Vec3>& R,
    const std::vector<double>& a,
    const std::vector<double>& n
) {
    double val = 0.0;
    for (size_t i = 0; i < R.size(); i++) {
        val += func_core(r - R[i], a[i], n[i]);
    }
    return val;
}

double ref(const std::vector<double>& a, const std::vector<double>& n) {
    double val = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        val += ref_core(a[i], n[i]);
    }
    return val;
}

// A Param object specifies a test function
struct Param {
    std::vector<Vec3> R;
    std::vector<double> a;
    std::vector<double> n;
};

std::vector<Param> test_params = {
    {
        {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 2.0},
        },
        {0.5, 2.0},
        {0, 0}
    },
    {
        {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 2.0},
            {0.0, 3.0, 0.0},
        },
        {0.5, 2.0, 1.5},
        {1, 2, 0.5}
    },
    {
        {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 3.0},
            {0.0, 3.0, 0.0},
            {9.0, 0.0, 0.0},
        },
        {1.0, 2.0, 1.5, 2.0},
        {2.5, 2, 0.5, 1}
    },
    {
        {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 3.0},
            {0.0, 3.0, 0.0},
            {9.0, 0.0, 0.0},
            {1.0, 1.0, 1.0},
            {2.0, 2.0, 2.0},
            {3.0, 3.0, 3.0},
            {4.0, 4.0, 4.0},
            {5.0, 5.0, 5.0},
            {6.0, 6.0, 6.0},
        },
        {1.0, 2.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        {2.5, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
    },
};

std::vector<double> dist_R_R(const std::vector<Vec3>& R) {
    // tabulate dRR[I,J] = || R[I] - R[J] ||
    size_t nR = R.size();
    std::vector<double> dRR(nR*nR, 0.0);
    for (size_t I = 0; I < nR; I++) {
        for (size_t J = I + 1; J < nR; J++) {
            double d = norm(R[I] - R[J]);
            dRR[I*nR + J] = d;
            dRR[J*nR + I] = d;
        }
    }
    return dRR;
}

class PartitionTest: public ::testing::Test {
protected:
    PartitionTest();

    // grid & weight for one-center integration
    std::vector<double> r;
    std::vector<double> w;

    const double tol = 1e-5;
};

PartitionTest::PartitionTest() {
    // angular grid & weight
    std::vector<double> r_ang, w_ang;
    int lmax = 25;
    Grid::Angular::delley(lmax, r_ang, w_ang);

    // radial grid & weight
    std::vector<double> r_rad, w_rad;
    int nrad = 60;
    int Rcut = 7.0;
    int mult = 2;
    Grid::Radial::baker(nrad, Rcut, r_rad, w_rad, mult);

    // complete grid & weight for one-center integration
    size_t ngrid = w_rad.size() * w_ang.size();
    r.resize(3*ngrid);
    w.resize(ngrid);

    size_t ir = 0;
    for (size_t i = 0; i < w_rad.size(); i++) {
        for (size_t j = 0; j < w_ang.size(); j++) {
            r[3*ir] = r_rad[i] * r_ang[3*j];
            r[3*ir+1] = r_rad[i] * r_ang[3*j+1];
            r[3*ir+2] = r_rad[i] * r_ang[3*j+2];
            w[ir] = w_rad[i] * w_ang[j] * 4.0 * PI;
            ++ir;
        }
    }
}


TEST_F(PartitionTest, Becke) {
    dur = dur.zero();
    for (const Param& param : test_params) {
        double val = 0.0;
        double val_ref = ref(param.a, param.n);

        // tabulate || R[I] - R[J] ||
        std::vector<double> dRR(dist_R_R(param.R));

        // all centers are involved
        size_t nR = param.R.size();
        std::vector<int> iR(nR);
        std::iota(iR.begin(), iR.end(), 0);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(iR.begin(), iR.end(), g);

        for (size_t I = 0; I < nR; ++I) { // for each center
            for (size_t i = 0; i < w.size(); i++) {
                Vec3 ri = Vec3{r[3*i], r[3*i+1], r[3*i+2]} + param.R[I];

                // tabulate || r - R[J] ||
                std::vector<double> drR(nR);
                for (size_t J = 0; J < nR; ++J) {
                    drR[J] = norm(ri - param.R[J]);
                }

                // partition weight for this grid point
                start = iclock::now();
                double w_part = Grid::Partition::w_becke(
                    drR.size(), drR.data(), dRR.data(),
                    iR.size(), iR.data(), I
                );
                dur += iclock::now() - start;

                val += w_part * w[i] * func(ri, param.R, param.a, param.n);
            }
        }

        EXPECT_NEAR(val, val_ref, tol);
    }
    printf("time elapsed = %8.3e seconds\n", dur.count());
}


TEST_F(PartitionTest, Stratmann) {
    dur = dur.zero();

    for (const Param& param : test_params) {
        double val = 0.0;
        double val_ref = ref(param.a, param.n);

        // tabulate || R[I] - R[J] ||
        std::vector<double> dRR(dist_R_R(param.R));

        // all centers are involved
        size_t nR = param.R.size();
        std::vector<int> iR(nR);
        std::iota(iR.begin(), iR.end(), 0);

        // radii of exclusive zone
        std::vector<double> drR_thr(nR);
        for (size_t I = 0; I < nR; ++I) {
            double dRRmin = 1e100;
            for (size_t J = 0; J < nR; ++J) {
                if (J != I) {
                    dRRmin = std::min(dRRmin, dRR[I*nR + J]);
                }
            }
            drR_thr[I] = 0.5 * (1.0 - Grid::Partition::stratmann_a) * dRRmin;
        }

        for (size_t I = 0; I < nR; ++I) { // for each center
            for (size_t i = 0; i < w.size(); i++) {
                Vec3 ri = Vec3{r[3*i], r[3*i+1], r[3*i+2]} + param.R[I];

                // tabulate || r - R[J] ||
                std::vector<double> drR(nR);
                for (size_t J = 0; J < nR; ++J) {
                    drR[J] = norm(ri - param.R[J]);
                }

                // partition weight for this grid point
                start = iclock::now();
                double w_part = Grid::Partition::w_stratmann(
                    drR.size(), drR.data(), dRR.data(), drR_thr.data(), 
                    iR.size(), iR.data(), I
                );
                dur += iclock::now() - start;

                val += w_part * w[i] * func(ri, param.R, param.a, param.n);
            }
        }

        EXPECT_NEAR(val, val_ref, tol);
    }
    printf("time elapsed = %8.3e seconds\n", dur.count());
}


int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif

    return result;
}


// 异核修正：R_i > R_j 时中点权重应偏向大原子（小原子得权重少、切换面偏向大原子侧移）
TEST_F(PartitionTest, BeckeHeteronuclearMidpoint)
{
    // 两中心相距 2 Bohr，网格点在中点：drR={1,1}, dRR=[0,2;2,0]
    double drR[2] = {1.0, 1.0};
    double dRR[4] = {0.0, 2.0, 2.0, 0.0};
    double radii[2] = {1.5, 0.5};   // 大原子、小原子（Bragg-Slater 量级）
    int iR[2] = {0, 1};
    // 无修正时中点 w=0.5；有修正时大原子权重 > 0.5
    double w_big = Grid::Partition::w_becke_adjusted(2, drR, dRR, radii, 2, iR, 0);
    EXPECT_GT(w_big, 0.5);
    EXPECT_NEAR(w_big + Grid::Partition::w_becke_adjusted(2, drR, dRR, radii, 2, iR, 1), 1.0, 1e-12);
    // 交换半径顺序：小原子侧权重应 < 0.5（修正与半径比符号一致）
    double radii_rev[2] = {0.5, 1.5};
    double w_small = Grid::Partition::w_becke_adjusted(2, drR, dRR, radii_rev, 2, iR, 0);
    EXPECT_LT(w_small, 0.5);
}

// 解析位置导数 vs 中心差分：三中心随机几何，∂w_0/∂R_1 的 x 分量
TEST_F(PartitionTest, BeckeDerivFD)
{
    // 随机几何：中心 0/1/2 与网格点 r（不共线、距离量级 ~2-4 Bohr）
    std::mt19937 g(42);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    Vec3 R[3] = {{dist(g), dist(g), dist(g)},
                 {dist(g), dist(g), dist(g)},
                 {dist(g), dist(g), dist(g)}};
    Vec3 r = {dist(g), dist(g), dist(g)};
    double radii[3] = {1.2, 0.8, 1.0};
    int iR[3] = {0, 1, 2};
    const double delta = 1e-5;

    // 解析导数
    auto eval_geometry = [](const Vec3& rp, const Vec3* Rc) {
        std::vector<double> drR(3), dRR(9, 0.0), eR(9, 0.0);
        for (int I = 0; I < 3; ++I)
        {
            drR[I] = norm(rp - Rc[I]);
            for (int d = 0; d < 3; ++d)
            {
                eR[3*I + d] = (Rc[I][d] - rp[d]) / drR[I];
            }
            for (int J = I + 1; J < 3; ++J)
            {
                dRR[I*3 + J] = norm(Rc[I] - Rc[J]);
                dRR[J*3 + I] = dRR[I*3 + J];
            }
        }
        return std::make_tuple(drR, dRR, eR);
    };

    auto [drR, dRR, eR] = eval_geometry(r, R);
    double dw[3];
    Grid::Partition::w_becke_adjusted_deriv(3, drR.data(), dRR.data(), radii, eR.data(), 3, iR, 0, 1, dw);

    // 中心差分：扰动 R_1 的每个分量
    for (int d = 0; d < 3; ++d)
    {
        Vec3 Rp[3] = {R[0], R[1], R[2]};
        Rp[1][d] += delta;
        auto [drRp, dRRp, eRp] = eval_geometry(r, Rp);
        double wp = Grid::Partition::w_becke_adjusted(3, drRp.data(), dRRp.data(), radii, 3, iR, 0);
        Rp[1][d] -= 2.0 * delta;
        auto [drRm, dRRm, eRm] = eval_geometry(r, Rp);
        double wm = Grid::Partition::w_becke_adjusted(3, drRm.data(), dRRm.data(), radii, 3, iR, 0);
        double fd = (wp - wm) / (2.0 * delta);
        EXPECT_NEAR(dw[d], fd, 1e-6) << "component " << d;
    }

    // 对不在 iR 中的中心（J=3 在 nR0=4 全集内但不在 iR 中）导数必须为零
    std::vector<double> drR4(4), dRR4(16, 0.0), eR4(12, 0.0);
    for (int I = 0; I < 3; ++I)
    {
        drR4[I] = drR[I];
        for (int d = 0; d < 3; ++d)
        {
            eR4[3*I + d] = eR[3*I + d];
        }
        for (int J = I + 1; J < 3; ++J)
        {
            dRR4[I*4 + J] = dRR[I*3 + J];
            dRR4[J*4 + I] = dRR4[I*4 + J];
        }
    }
    drR4[3] = 3.0; // 任意占位（J=3 不参与权重）
    double dw0[3] = {9.9, 9.9, 9.9};
    Grid::Partition::w_becke_adjusted_deriv(4, drR4.data(), dRR4.data(), radii, eR4.data(), 3, iR, 0, 3, dw0);
    EXPECT_DOUBLE_EQ(dw0[0], 0.0);
    EXPECT_DOUBLE_EQ(dw0[1], 0.0);
    EXPECT_DOUBLE_EQ(dw0[2], 0.0);
}
