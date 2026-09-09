#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <vector>

#include "constraint_test_utils.h"

// Mock symbols required by the cell_info object library in unit tests.
Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
    this->start_mag = nullptr;
}
Magnetism::~Magnetism()
{
    delete[] this->start_mag;
}
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}
#include "source_estate/module_constraint/constraint_deriv.h"
#include "source_estate/module_constraint/constraint_loop.h"

using ModuleBase::PI;

// Normalized Gaussian centered at 'center' with width sigma (Bohr).
static double gaussian(const std::array<double, 3>& r,
                       const std::array<double, 3>& center,
                       const double sigma)
{
    const double dx = r[0] - center[0];
    const double dy = r[1] - center[1];
    const double dz = r[2] - center[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double norm = std::pow(2.0 * PI * sigma * sigma, 1.5);
    return std::exp(-r2 / (2.0 * sigma * sigma)) / norm;
}

class ConstraintLoopTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell;
    ModulePW::PW_Basis* rhopw = nullptr;
    std::vector<double> radii;
    std::vector<std::array<double, 3>> pos;
    std::vector<double> nelec_atom;
    std::vector<double> rho_ref;
    double dV = 0.0;
    constraint::WeightGrid* wg = nullptr; // single-constraint map {atom 0}

    void SetUp() override
    {
        Set_GlobalV_Default();
        constraint::ConstraintLoop::instance().reset();
        ucell = make_h2o_ucell();
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 40, 40, 40);
        rhopw->distribute_r();
        radii = {1.5, 0.5, 0.5}; // O, H1, H2 (Bohr)
        pos = h2o_positions(*ucell);
        nelec_atom = {8.0, 1.0, 1.0};
        dV = rhopw->omega / static_cast<double>(rhopw->nxyz);
        // Reference density: atomic superposition of Gaussians (sigma = 1.0).
        const double sigma = 1.0;
        rho_ref.assign(rhopw->nrxx, 0.0);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            const int i = ir / (rhopw->ny * rhopw->nplane);
            const int j = ir / rhopw->nplane - i * rhopw->ny;
            const int k = ir % rhopw->nplane + rhopw->startz_current;
            const ModuleBase::Vector3<double> rfrac(
                static_cast<double>(i) / rhopw->nx,
                static_cast<double>(j) / rhopw->ny,
                static_cast<double>(k) / rhopw->nz);
            const ModuleBase::Vector3<double> rc =
                rfrac * ucell->latvec * ucell->lat0;
            const std::array<double, 3> r{rc.x, rc.y, rc.z};
            for (size_t I = 0; I < pos.size(); ++I)
            {
                rho_ref[ir] += nelec_atom[I] * gaussian(r, pos[I], sigma);
            }
        }
        // Reference weight field with the same single-constraint map the
        // loop uses: fragment {atom 0} only.
        wg = new constraint::WeightGrid(*ucell, rhopw, radii);
        wg->set_constraint_atoms({{0}});
        wg->build();
    }

    void TearDown() override
    {
        delete wg;
        delete rhopw;
        constraint::ConstraintLoop::instance().reset();
    }

    // Mock SCF response, normalized to unit stiffness:
    //   rho(mu) = rho_ref - mu * w0 / S,  S = int w0^2 dV
    //   =>  Q(mu) = Q_ref - mu  (monotonic, physically negative response).
    // The raw stiffness S = int w0^2 dV ~ box volume (O weight ~ 1 over the
    // box) would push the secant below its step resolution; the normalization
    // keeps the test focused on the loop logic, not the real response scale.
    void fill_rho_mock(const std::vector<double>& mu, std::vector<double>& rho) const
    {
        rho = rho_ref;
        const double S = S_int();
        const std::vector<double>& w0 = wg->constraint_weight(0);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            rho[ir] -= mu[0] * w0[ir] / S;
        }
    }

    double S_int() const
    {
        const std::vector<double>& w0 = wg->constraint_weight(0);
        double s = 0.0;
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            s += w0[ir] * w0[ir];
        }
        return s * dV;
    }

    double Q_of(const std::vector<double>& rho) const
    {
        const std::vector<double>& w0 = wg->constraint_weight(0);
        double q = 0.0;
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            q += w0[ir] * rho[ir];
        }
        return q * dV;
    }

    // Spin-channel mock response: the split injection (V_up += mu*w,
    // V_dn -= mu*w) repels spin-up from / attracts spin-down to the
    // fragment with the same normalized stiffness S = int w0^2 dV as the
    // charge mock, so
    //   rho_up(mu) = rho_ref - mu*w0/S, rho_dn(mu) = rho_ref + mu*w0/S
    //   =>  Q_m(mu) = int w0 (rho_up - rho_dn) dr = -2 mu
    // (negative response — same direction as charge, verified in the
    // 212_PW_constraint_h2o_spin integration case).
    void fill_rho_mock_spin(const std::vector<double>& mu,
                            std::vector<double>& rho_up,
                            std::vector<double>& rho_dn) const
    {
        rho_up = rho_ref;
        rho_dn = rho_ref;
        const double S = S_int();
        const std::vector<double>& w0 = wg->constraint_weight(0);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            rho_up[ir] -= mu[0] * w0[ir] / S;
            rho_dn[ir] += mu[0] * w0[ir] / S;
        }
    }

    double Q_m_of(const std::vector<double>& rho_up,
                  const std::vector<double>& rho_dn) const
    {
        const std::vector<double>& w0 = wg->constraint_weight(0);
        double q = 0.0;
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            q += w0[ir] * (rho_up[ir] - rho_dn[ir]);
        }
        return q * dV;
    }

    // --- Stage-A mixed-channel mock helpers --------------------------------
    // The mixed test drives the loop's OWN weight grid (loop.weight_grid(),
    // fragments {{0}, {1, 2}}), so the mock response uses the same weights
    // the observer integrates — decoupling by construction:
    //   charge channel: Sigma(mu_c) = rho_ref - mu_c * w0 / S_c
    //   spin channel:   m(mu_s)    = -2 mu_s * w12 / S_m   (split-injection
    //                   repels up / attracts down, Q_s = -2 mu_s)
    // with rho_up = (Sigma + m)/2, rho_dn = (Sigma - m)/2, so Q_c sees only
    // mu_c and Q_s sees only mu_s (independent linear-response channels).
    double S_alpha(const constraint::WeightGrid& g, const int a) const
    {
        const std::vector<double>& w = g.constraint_weight(a);
        double s = 0.0;
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            s += w[ir] * w[ir];
        }
        return s * dV;
    }

    double Q_w(const constraint::WeightGrid& g,
               const int a,
               const std::vector<double>& rho) const
    {
        const std::vector<double>& w = g.constraint_weight(a);
        double q = 0.0;
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            q += w[ir] * rho[ir];
        }
        return q * dV;
    }

    void fill_rho_mock_mixed(const std::vector<double>& mu,
                             const constraint::WeightGrid& g,
                             std::vector<double>& rho_up,
                             std::vector<double>& rho_dn) const
    {
        const double S_c = S_alpha(g, 0);
        const double S_m = S_alpha(g, 1);
        const std::vector<double>& w0 = g.constraint_weight(0);
        const std::vector<double>& w12 = g.constraint_weight(1);
        rho_up.resize(rhopw->nrxx);
        rho_dn.resize(rhopw->nrxx);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            const double sig = rho_ref[ir] - mu[0] * w0[ir] / S_c;
            const double m = -2.0 * mu[1] * w12[ir] / S_m;
            rho_up[ir] = 0.5 * (sig + m);
            rho_dn[ir] = 0.5 * (sig - m);
        }
    }

    // Mixed charge+spin specs through the extended configure core (the A1
    // staging-guard site): returns ConfigStatus::OK only once the A4 guard
    // removal landed, so the G4 test fails while the guard rejects.
    static constraint::ConfigStatus configure_mixed(
        const std::string& json,
        constraint::ConstraintConfig& cfg,
        std::vector<constraint::ConstraintSpec>& specs,
        std::vector<std::string>& warnings,
        std::string& error,
        const int nat,
        const int nspin)
    {
        return constraint::configure_constraint(
            cfg, specs, warnings, true, "charge", "becke", "delta", json, 5.0,
            1e-4, nat, nspin, error);
    }

    constraint::ConstraintConfig make_cfg(const double delta,
                                          const double mu_max = 5.0,
                                          const double thr = 1e-4,
                                          const std::string& type = "charge") const
    {
        constraint::ConstraintConfig cfg;
        cfg.enabled = true;
        cfg.type = type;
        cfg.weight_type = "becke";
        cfg.target_mode = "delta";
        cfg.mu_max = mu_max;
        cfg.thr = thr;
        constraint::ConstraintTarget t;
        t.value = delta;
        t.atoms = {0};
        cfg.targets = {t};
        return cfg;
    }
};

TEST_F(ConstraintLoopTest, ReferenceThenConstrained)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, make_cfg(0.1), radii, 10.0);
    ASSERT_TRUE(loop.enabled());
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::REFERENCE);

    // SCF iteration 1: reference run, mu = 0, injection is a no-op.
    ModuleBase::matrix veff(1, rhopw->nrxx);
    ModuleBase::matrix veff_ref = veff;
    loop.inject_potential(1, veff, veff);
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(veff(0, ir), veff_ref(0, ir));
    }
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    bool conv = true; // SCF converged
    loop.on_scf_converged(1, conv);

    // Reference recorded, targets built, first secant step taken, SCF forced
    // to continue.
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    EXPECT_EQ(loop.status(), constraint::MuStatus::RUNNING);
    EXPECT_FALSE(conv);
    EXPECT_EQ(loop.targets().size(), 1u);
    EXPECT_NEAR(loop.targets()[0], Q_of(rho_ref) + 0.1, 1e-10);
    EXPECT_LT(loop.mu()[0], 0.0); // increasing Q needs mu < 0
    EXPECT_NE(loop.last_audit_line().find("CONSTRAINT_AUDIT"),
              std::string::npos);
    EXPECT_EQ(loop.outer_steps(), 1);
}

TEST_F(ConstraintLoopTest, ConvergesOnLinearResponse)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    const double delta = 0.01;
    loop.init(*ucell, rhopw, make_cfg(delta), radii, 10.0);

    const double q_ref = Q_of(rho_ref);
    const double mu_star = -delta; // exact root of the normalized response

    std::vector<double> rho;
    int iter = 1;
    bool conv = false;
    // Reference SCF.
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(iter, rho_ptr, 1);
    conv = true;
    loop.on_scf_converged(iter, conv);
    EXPECT_FALSE(conv); // forced to continue
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 20)
    {
        ++iter;
        fill_rho_mock(loop.mu(), rho);
        const double* rp[1] = {rho.data()};
        loop.observe(iter, rp, 1);
        conv = true;
        loop.on_scf_converged(iter, conv);
        ++guard;
    }
    EXPECT_LT(guard, 20);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_TRUE(conv); // outer loop done: SCF ends normally
    EXPECT_NEAR(loop.mu()[0], mu_star, 1e-6);
    EXPECT_NEAR(Q_of(rho), loop.targets()[0], 1e-4);
}

TEST_F(ConstraintLoopTest, SpinChannelConvergesOnLinearResponse)
{
    // Phase-2 spin channel: the split injection (V_up += mu*w, V_dn -= mu*w)
    // repels spin-up from / attracts spin-down to the fragment for mu > 0,
    // driving m = rho_up - rho_dn down.  Normalized linear response gives
    // Q_m(mu) = -2 mu, so the root for target delta is mu* = -delta/2 — the
    // same sign pattern as the charge channel (verified in the
    // 212_PW_constraint_h2o_spin integration case, dQ/dmu ~ -1.4 e/Ry).
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    const double delta = 0.02;
    loop.init(*ucell, rhopw, make_cfg(delta, 5.0, 1e-4, "spin"), radii, 10.0);
    ASSERT_TRUE(loop.enabled());

    std::vector<double> rho_up, rho_dn;
    // Reference SCF (mu = 0): the free run must NOT converge to the
    // unnatural target (anti-fake convergence, T4a'): at mu = 0 the
    // observed m = 0 != target = delta, so the loop must stay RUNNING and
    // the first secant step must move mu away from zero.
    fill_rho_mock_spin({0.0}, rho_up, rho_dn);
    const double* rho_ptr[2] = {rho_up.data(), rho_dn.data()};
    int iter = 1;
    bool conv = false;
    loop.observe(iter, rho_ptr, 2);
    EXPECT_NEAR(loop.charges()[0], 0.0, 1e-10); // natural m at mu=0
    conv = true;
    loop.on_scf_converged(iter, conv);
    EXPECT_FALSE(conv); // forced to continue — not a free-run convergence
    EXPECT_EQ(loop.status(), constraint::MuStatus::RUNNING);
    EXPECT_LT(loop.mu()[0], 0.0); // increasing m needs mu < 0 (charge-like)

    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 20)
    {
        ++iter;
        fill_rho_mock_spin(loop.mu(), rho_up, rho_dn);
        const double* rp[2] = {rho_up.data(), rho_dn.data()};
        loop.observe(iter, rp, 2);
        conv = true;
        loop.on_scf_converged(iter, conv);
        ++guard;
    }
    EXPECT_LT(guard, 20);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_TRUE(conv);
    EXPECT_NEAR(loop.mu()[0], -delta / 2.0, 1e-6); // root of Q_m = -2 mu
    EXPECT_NEAR(Q_m_of(rho_up, rho_dn), loop.targets()[0], 1e-4);
    EXPECT_NEAR(loop.targets()[0], delta, 1e-10); // m_ref = 0
}

TEST_F(ConstraintLoopTest, MixedConvergesOnLinearResponse)
{
    // G4 gate: the mixed charge+spin run is served end-to-end by the stage-A
    // loop.  The spec list goes through the extended configure core — the
    // A1 staging-guard site — so this test FAILS while the guard still
    // rejects a mixed run (guard-removal falsifiability: the ASSERT below is
    // the red/green boundary) and converges after the A4 removal.
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<std::string> warnings;
    std::string error;
    const std::string json = R"({"constraints": [
        {"type": "charge", "target": 0.01, "atoms": [0]},
        {"type": "spin", "target": 0.02, "atoms": [1, 2]}]})";
    const constraint::ConfigStatus st = configure_mixed(
        json, cfg, specs, warnings, error, ucell->nat, 2);
    ASSERT_EQ(st, constraint::ConfigStatus::OK) << error;
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Charge);
    EXPECT_EQ(specs[1].kind, constraint::ConstraintKind::Spin);

    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, cfg, specs, radii, 10.0);
    ASSERT_TRUE(loop.enabled());
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::REFERENCE);

    // Reference SCF (mu = 0): the free run must NOT reach either target
    // (charge delta != 0 and the natural spin reference m = 0 != delta) —
    // anti-fake convergence per channel.  Both multipliers must move away
    // from zero after the first secant step.
    const constraint::WeightGrid& g = loop.weight_grid();
    std::vector<double> rho_up, rho_dn;
    fill_rho_mock_mixed({0.0, 0.0}, g, rho_up, rho_dn);
    const double qc_ref = Q_w(g, 0, rho_ref);
    const double* rho_ptr[2] = {rho_up.data(), rho_dn.data()};
    int iter = 1;
    bool conv = false;
    loop.observe(iter, rho_ptr, 2);
    EXPECT_NEAR(loop.charges()[0], qc_ref, 1e-6);
    EXPECT_NEAR(loop.charges()[1], 0.0, 1e-10);
    conv = true;
    loop.on_scf_converged(iter, conv);
    EXPECT_FALSE(conv); // forced to continue — no free-run convergence
    EXPECT_EQ(loop.status(), constraint::MuStatus::RUNNING);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    EXPECT_LT(loop.mu()[0], 0.0); // charge: increasing Q needs mu < 0
    EXPECT_LT(loop.mu()[1], 0.0); // spin: increasing m needs mu < 0

    // Drive both channels to their independent linear-response roots:
    // Q_c(mu_c) = Q_ref - mu_c (root -delta_c) and Q_s(mu_s) = -2 mu_s
    // (root -delta_s/2), so mu* = (-0.01, -0.01).
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 30)
    {
        ++iter;
        fill_rho_mock_mixed(loop.mu(), g, rho_up, rho_dn);
        const double* rp[2] = {rho_up.data(), rho_dn.data()};
        loop.observe(iter, rp, 2);
        conv = true;
        loop.on_scf_converged(iter, conv);
        ++guard;
    }
    ASSERT_LT(guard, 30);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_TRUE(conv); // outer loop done: SCF ends normally
    EXPECT_NEAR(loop.mu()[0], -0.01, 1e-6);
    EXPECT_NEAR(loop.mu()[1], -0.01, 1e-6);
    EXPECT_NEAR(loop.targets()[0], qc_ref + 0.01, 1e-8);
    EXPECT_NEAR(loop.targets()[1], 0.02, 1e-8); // m_ref = 0
    EXPECT_NEAR(loop.charges()[0], loop.targets()[0], 1e-3);
    EXPECT_NEAR(loop.charges()[1], loop.targets()[1], 1e-3);

    // M5 kind= audit labels: the mixed run is tagged per constraint.
    const std::string& line = loop.last_audit_line();
    EXPECT_NE(line.find("c[0] kind=charge"), std::string::npos);
    EXPECT_NE(line.find("c[1] kind=spin"), std::string::npos);
}

TEST_F(ConstraintLoopTest, MixedFuseHonorsPerComponentCap)
{
    // A0 decision D3: per-constraint mu_max must reach the fuse logic of the
    // correct component.  Charge (cap 5.0, root -0.01) converges while spin
    // (cap 0.05, root -0.25) pins at ITS OWN small cap and fuses the run as
    // UNREACHABLE.  A buggy single scalar cap (e.g. the legacy cfg.mu_max
    // mirror = 5.0, the first spec's cap) would let the spin channel run to
    // its root and CONVERGE — this test discriminates the per-component path.
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<std::string> warnings;
    std::string error;
    const std::string json = R"({"constraints": [
        {"type": "charge", "target": 0.01, "atoms": [0], "mu_max": 5.0},
        {"type": "spin", "target": 0.5, "atoms": [1, 2], "mu_max": 0.05}]})";
    const constraint::ConfigStatus st = configure_mixed(
        json, cfg, specs, warnings, error, ucell->nat, 2);
    ASSERT_EQ(st, constraint::ConfigStatus::OK) << error;
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_DOUBLE_EQ(specs[0].mu_max, 5.0);
    EXPECT_DOUBLE_EQ(specs[1].mu_max, 0.05);

    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, cfg, specs, radii, 10.0);
    ASSERT_TRUE(loop.enabled());
    const constraint::WeightGrid& g = loop.weight_grid();
    std::vector<double> rho_up, rho_dn;
    int iter = 1;
    bool conv = false;
    fill_rho_mock_mixed({0.0, 0.0}, g, rho_up, rho_dn);
    const double* rho_ptr[2] = {rho_up.data(), rho_dn.data()};
    loop.observe(iter, rho_ptr, 2);
    conv = true;
    loop.on_scf_converged(iter, conv);
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 30)
    {
        ++iter;
        fill_rho_mock_mixed(loop.mu(), g, rho_up, rho_dn);
        const double* rp[2] = {rho_up.data(), rho_dn.data()};
        loop.observe(iter, rp, 2);
        conv = true;
        loop.on_scf_converged(iter, conv);
        ++guard;
    }
    ASSERT_LT(guard, 30);
    EXPECT_EQ(loop.status(), constraint::MuStatus::UNREACHABLE);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_TRUE(conv); // fused: SCF terminates cleanly
    // Spin pinned at ITS own cap (0.05), not at the charge spec's 5.0.
    EXPECT_DOUBLE_EQ(loop.mu()[1], -0.05);
    // Charge channel converged independently of the spin cap.
    EXPECT_NEAR(loop.mu()[0], -0.01, 1e-4);
    EXPECT_LT(std::abs(loop.mu()[0]), 0.05); // not pinned at the spin cap
}

TEST_F(ConstraintLoopTest, FuseUnreachable)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    // Small mu cap: pins after a few steps, plateau fuse triggers quickly.
    loop.init(*ucell, rhopw, make_cfg(0.1, 0.2, 1e-4), radii, 10.0);

    std::vector<double> rho;
    int iter = 1;
    bool conv = false;
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(iter, rho_ptr, 1);
    conv = true;
    loop.on_scf_converged(iter, conv);
    EXPECT_FALSE(conv);
    int guard = 0;
    // Unresponsive channel: rho does not react to mu at all.
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 50)
    {
        ++iter;
        rho = rho_ref;
        const double* rp[1] = {rho.data()};
        loop.observe(iter, rp, 1);
        conv = true;
        loop.on_scf_converged(iter, conv);
        ++guard;
    }
    EXPECT_LT(guard, 50);
    EXPECT_EQ(loop.status(), constraint::MuStatus::UNREACHABLE);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_TRUE(conv); // fused: SCF terminates cleanly
    EXPECT_DOUBLE_EQ(loop.mu()[0], -0.2); // pinned at the cap
}

TEST_F(ConstraintLoopTest, InjectMatchesObserver)
{
    // Shared-instance principle through the loop: after inject with mu,
    // int rho * dV_injected dr == mu * Q (single constraint).
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, make_cfg(0.1), radii, 10.0);
    // Drive past the reference so mu != 0.
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    bool conv = true; // SCF converged (reference step)
    loop.on_scf_converged(1, conv);
    ASSERT_LT(loop.mu()[0], 0.0);

    std::vector<double> rho;
    fill_rho_mock(loop.mu(), rho);
    ModuleBase::matrix veff(1, rhopw->nrxx);
    ModuleBase::matrix veff_smooth(1, rhopw->nrxx);
    loop.inject_potential(2, veff, veff_smooth);
    const std::vector<double>& w0 = wg->constraint_weight(0);
    double integral = 0.0;
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        integral += rho[ir] * veff(0, ir);
    }
    integral *= dV;
    EXPECT_NEAR(integral, loop.mu()[0] * Q_of(rho), 1e-10 * std::abs(Q_of(rho)));
    // veff_smooth got the same injection.
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(veff_smooth(0, ir), veff(0, ir));
    }
}

TEST_F(ConstraintLoopTest, DisabledNoop)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg;
    cfg.enabled = false;
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    EXPECT_FALSE(loop.enabled());

    ModuleBase::matrix veff(1, rhopw->nrxx);
    ModuleBase::matrix veff_ref = veff;
    loop.inject_potential(1, veff, veff);
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    bool conv = true;
    loop.on_scf_converged(1, conv);
    EXPECT_TRUE(conv); // never overrides convergence
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::IDLE);
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(veff(0, ir), veff_ref(0, ir));
    }
}

TEST_F(ConstraintLoopTest, IgnoresUnconvergedScf)
{
    // Two-stage gating: the outer step must not run while the SCF is not
    // converged, otherwise the secant chases the mixing noise.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, make_cfg(0.1), radii, 10.0);
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    bool conv = false; // SCF NOT converged
    loop.on_scf_converged(1, conv);
    // Nothing happened: still in the reference phase, mu untouched, no audit.
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::REFERENCE);
    EXPECT_EQ(loop.outer_steps(), 0);
    EXPECT_DOUBLE_EQ(loop.mu()[0], 0.0);
    EXPECT_TRUE(loop.last_audit_line().empty());
    // The same non-converged call during the constrained phase is a no-op.
    conv = true;
    loop.on_scf_converged(1, conv);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    EXPECT_EQ(loop.outer_steps(), 1);
    loop.observe(2, rho_ptr, 1);
    conv = false;
    loop.on_scf_converged(2, conv);
    EXPECT_EQ(loop.outer_steps(), 1); // unchanged
}

TEST_F(ConstraintLoopTest, ComputeForceZeroWhenDisabled)
{
    // The loop is disabled (never initialized in this test): compute_force
    // must be a no-op and leave the buffer untouched (zero).
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    EXPECT_FALSE(loop.enabled());
    const double* rho_ptr[1] = {rho_ref.data()};
    ModuleBase::matrix F(ucell->nat, 3);
    F.fill_out(7.0);
    loop.compute_force(rho_ptr, 1, F);
    for (int J = 0; J < ucell->nat; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_DOUBLE_EQ(F(J, d), 7.0); // untouched
        }
    }
}

TEST_F(ConstraintLoopTest, ComputeForceConvergedMatchesKernel)
{
    // Drive the outer loop to convergence on the normalized linear mock
    // response (mu* = -delta), then compare compute_force against a direct
    // constraint_force call with the same density and the loop's mu.  The
    // loop builds its own WeightGrid from the same cell/radii as the
    // fixture's, so the two grids are bit-identical (deterministic build):
    // an exact match proves the wiring (density pointer, channel, shared
    // weight field) and that "observable == injection operator" carries
    // over to the force.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    const double delta = 0.01;
    loop.init(*ucell, rhopw, make_cfg(delta), radii, 10.0);

    std::vector<double> rho;
    int iter = 1;
    bool conv = false;
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(iter, rho_ptr, 1);
    conv = true;
    loop.on_scf_converged(iter, conv);
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 20)
    {
        ++iter;
        fill_rho_mock(loop.mu(), rho);
        const double* rp[1] = {rho.data()};
        loop.observe(iter, rp, 1);
        conv = true;
        loop.on_scf_converged(iter, conv);
        ++guard;
    }
    ASSERT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    ASSERT_NE(loop.mu()[0], 0.0);

    fill_rho_mock(loop.mu(), rho);
    const double* rho_arr[1] = {rho.data()};
    ModuleBase::matrix Floop(ucell->nat, 3);
    loop.compute_force(rho_arr, 1, Floop);
    // The fixture's weight grid builds only the weight field; the direct
    // kernel call below needs the position-derivative grid too (the loop
    // path builds it lazily inside compute_force).
    wg->build_derivatives();
    ModuleBase::matrix Fdir(ucell->nat, 3);
    constraint::constraint_force(*wg, rho_arr, 1,
                                 constraint::DensityChannel::Charge,
                                 loop.mu(), Fdir);
    for (int J = 0; J < ucell->nat; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_DOUBLE_EQ(Floop(J, d), Fdir(J, d))
                << "J " << J << " d " << d;
        }
    }
    // The converged force is nonzero (mu* != 0 and the density overlaps the
    // weight derivative field): the wiring does something.
    double norm = 0.0;
    for (int J = 0; J < ucell->nat; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            norm += Floop(J, d) * Floop(J, d);
        }
    }
    EXPECT_GT(norm, 0.0);
}

TEST_F(ConstraintLoopTest, ComputeForceZeroWhenMuZero)
{
    // delta = 0 -> target == reference charge -> the outer loop converges
    // with mu = 0.  The constraint force must then be EXACTLY zero (the
    // kernel short-circuits zero multipliers), which is the mu = 0
    // reference-phase check of the force wiring.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, make_cfg(0.0), radii, 10.0);
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    bool conv = true;
    loop.on_scf_converged(1, conv);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_DOUBLE_EQ(loop.mu()[0], 0.0);

    std::vector<double> rho;
    fill_rho_mock(loop.mu(), rho);
    const double* rho_arr[1] = {rho.data()};
    ModuleBase::matrix F(ucell->nat, 3);
    loop.compute_force(rho_arr, 1, F);
    for (int J = 0; J < ucell->nat; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_DOUBLE_EQ(F(J, d), 0.0);
        }
    }
}
