#include "gtest/gtest.h"

#include <cmath>
#include <cstdlib>
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
            1e-4, 0.05, 0.0, 0.0, nat, nspin, error);
    }

    // One mock SCF iteration in the real hook order:
    //   fill rho(mu_obs) -> observe -> on_iteration(iter, drho) ->
    //   on_scf_converged.
    // Returns the on_iteration() verdict (the mix-reset request); the SCF
    // verdict the esolver would take after on_scf_converged() goes to
    // conv_out.  'mu_obs' is the multiplier the mock density is built at:
    // loop.mu()[0] for a settled density, a perturbed value to simulate a
    // settling bounce.  'scf_conv_in' is the drho-based verdict handed in
    // (true unless the test wants to keep the SCF running).
    bool mock_iteration(constraint::ConstraintLoop& loop,
                        std::vector<double>& rho,
                        const int iter,
                        const double mu_obs,
                        const double drho,
                        bool& conv_out,
                        const bool scf_conv_in = true)
    {
        fill_rho_mock({mu_obs}, rho);
        const double* rp[1] = {rho.data()};
        loop.observe(iter, rp, 1);
        const bool reset = loop.on_iteration(iter, drho);
        bool conv = scf_conv_in;
        loop.on_scf_converged(iter, conv);
        conv_out = conv;
        return reset;
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

// ---------------------------------------------------------------------------
// Online energy branch guard (L10 lineage, constraint_branch_tol > 0)
// ---------------------------------------------------------------------------
//
// The guard compares the constrained energy E_tot = E_KS + sum_a mu_a(Q_a -
// t_a) at every converged SCF against the mu = 0 reference energy E_ref.  A
// dip below E_ref beyond the tolerance means the SCF left the reference
// electronic branch: the run must be fused as BRANCH_FLIP instead of being
// reported CONVERGED.  The mock response (Q = Q_ref - mu) is unchanged; only
// the energy fed to the guard is synthetic, which is exactly the seam the
// esolver drives through set_scf_energy().

TEST_F(ConstraintLoopTest, BranchGuardDefaultOffDoesNotFuse)
{
    // Default-off contract: without constraint_branch_tol the guard is a
    // pure no-op (legacy runs stay bit-identical), so even a nonsense energy
    // must not change the verdict.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.01);
    ASSERT_DOUBLE_EQ(cfg.branch_tol, 0.0); // default
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    EXPECT_FALSE(loop.branch_guard_armed());

    std::vector<double> rho;
    int iter = 1;
    bool conv = false;
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(iter, rho_ptr, 1);
    loop.set_scf_energy(-1.0e6); // nonsense: must be ignored
    conv = true;
    loop.on_scf_converged(iter, conv);
    EXPECT_FALSE(loop.reference_recorded());
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 20)
    {
        ++iter;
        fill_rho_mock(loop.mu(), rho);
        const double* rp[1] = {rho.data()};
        loop.observe(iter, rp, 1);
        loop.set_scf_energy(-1.0e6);
        conv = true;
        loop.on_scf_converged(iter, conv);
        ++guard;
    }
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
}

// Synthetic energy feed: place the guard's E_tot exactly de_target above the
// reference by subtracting the constraint term the loop will add back.
static void feed_guard_energy(constraint::ConstraintLoop& loop,
                              const double de_target)
{
    double e_con = 0.0;
    for (size_t a = 0; a < loop.mu().size(); ++a)
    {
        e_con += loop.mu()[a] * (loop.charges()[a] - loop.targets()[a]);
    }
    loop.set_scf_energy(loop.reference_energy() + de_target - e_con);
}

TEST_F(ConstraintLoopTest, BranchGuardRisingEnergyConverges)
{
    // Well-behaved run: the constrained energy rises above the reference by
    // the linear-response amount 0.5 |delta| |mu| (the II-1 FeO shape).  The
    // guard must not fire.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    const double delta = 0.01;
    constraint::ConstraintConfig cfg = make_cfg(delta);
    cfg.branch_tol = 1e-3;
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    ASSERT_TRUE(loop.branch_guard_armed());

    const double e_ref = -100.0;
    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    loop.set_scf_energy(e_ref);
    bool conv = true;
    loop.on_scf_converged(1, conv);
    ASSERT_FALSE(conv);
    ASSERT_TRUE(loop.reference_recorded());
    EXPECT_DOUBLE_EQ(loop.reference_energy(), e_ref);

    std::vector<double> rho;
    int iter = 1;
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 20)
    {
        ++iter;
        fill_rho_mock(loop.mu(), rho);
        const double* rp[1] = {rho.data()};
        loop.observe(iter, rp, 1);
        feed_guard_energy(loop, 0.5 * delta * std::abs(loop.mu()[0]));
        conv = true;
        loop.on_scf_converged(iter, conv);
        EXPECT_GE(loop.guard_energy(), loop.reference_energy());
        ++guard;
    }
    EXPECT_LT(guard, 20);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_NEAR(loop.mu()[0], -delta, 1e-6);
}

TEST_F(ConstraintLoopTest, BranchGuardToleratesDipWithinTolerance)
{
    // Boundary: a dip of exactly half the tolerance is numerical noise, not a
    // branch flip — the run must still converge.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.01);
    cfg.branch_tol = 1e-3;
    loop.init(*ucell, rhopw, cfg, radii, 10.0);

    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    loop.set_scf_energy(-50.0);
    bool conv = true;
    loop.on_scf_converged(1, conv);

    std::vector<double> rho;
    int iter = 1;
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 20)
    {
        ++iter;
        fill_rho_mock(loop.mu(), rho);
        const double* rp[1] = {rho.data()};
        loop.observe(iter, rp, 1);
        feed_guard_energy(loop, -0.5 * cfg.branch_tol);
        conv = true;
        loop.on_scf_converged(iter, conv);
        ++guard;
    }
    EXPECT_LT(guard, 20);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
}

TEST_F(ConstraintLoopTest, BranchGuardPreemptsTargetReachedOnFlippedBranch)
{
    // No-silent-acceptance discriminator: at the step where the mock response
    // lands EXACTLY on the target (|Q - t| < thr, so the residual test alone
    // would report CONVERGED) the energy is fed 2*tol BELOW the reference.
    // The guard runs before the outer step, so the verdict must be
    // BRANCH_FLIP — never CONVERGED.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.01);
    cfg.branch_tol = 1e-3;
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    ASSERT_TRUE(loop.enabled());

    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    loop.set_scf_energy(-50.0);
    bool conv = true;
    loop.on_scf_converged(1, conv);
    ASSERT_FALSE(conv);

    std::vector<double> rho;
    int iter = 1;
    int guard = 0;
    bool fused_at_target = false;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 20)
    {
        ++iter;
        fill_rho_mock(loop.mu(), rho);
        const double* rp[1] = {rho.data()};
        loop.observe(iter, rp, 1);
        const bool at_target
            = std::abs(loop.charges()[0] - loop.targets()[0]) < cfg.thr;
        feed_guard_energy(loop, at_target ? -2.0 * cfg.branch_tol : 0.0);
        conv = true;
        loop.on_scf_converged(iter, conv);
        if (at_target)
        {
            fused_at_target = true;
        }
        ++guard;
    }
    EXPECT_LT(guard, 20);
    EXPECT_TRUE(fused_at_target); // the residual test WAS satisfied ...
    EXPECT_LT(std::abs(loop.charges()[0] - loop.targets()[0]), cfg.thr);
    EXPECT_EQ(loop.status(), constraint::MuStatus::BRANCH_FLIP); // ... and rejected
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_TRUE(conv); // fused: the SCF terminates cleanly
    EXPECT_LT(loop.guard_energy(),
              loop.reference_energy() - cfg.branch_tol);
}

TEST_F(ConstraintLoopTest, BranchGuardRefusesMissingEnergyDeathTest)
{
    // Wiring guard: an armed branch guard without an energy input must abort
    // loudly (exit 1) rather than silently running unguarded.
    EXPECT_EXIT(
        {
            constraint::ConstraintConfig cfg = make_cfg(0.01);
            cfg.branch_tol = 1e-3;
            constraint::ConstraintLoop& loop
                = constraint::ConstraintLoop::instance();
            loop.init(*ucell, rhopw, cfg, radii, 10.0);
            const double* rp[1] = {rho_ref.data()};
            loop.observe(1, rp, 1);
            bool conv = true;
            loop.on_scf_converged(1, conv); // no set_scf_energy() call
        },
        ::testing::ExitedWithCode(1), "");
}

TEST_F(ConstraintLoopTest, BranchGuardRefusesFixedMuDeathTest)
{
    // The fixed-mu experiment has no mu = 0 reference SCF, so an armed guard
    // there would never fire; refuse the combination loudly (exit 1).
    EXPECT_EXIT(
        {
            setenv("ABA_CONSTRAINT_FIXED_MU", "0.01", 1);
            constraint::ConstraintConfig cfg = make_cfg(0.01);
            cfg.branch_tol = 1e-3;
            constraint::ConstraintLoop& loop
                = constraint::ConstraintLoop::instance();
            loop.init(*ucell, rhopw, cfg, radii, 10.0);
        },
        ::testing::ExitedWithCode(1), "");
}

TEST_F(ConstraintLoopTest, OnsiteMomentsReachTheAuditLine)
{
    // II-1b instrument wiring: per-atom on-site moments supplied by the
    // esolver appear as the fragment sum in the audit line; with none supplied
    // (the default, and the case without DFT+U) the token is absent so the
    // historical output is unchanged.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, make_cfg(0.01), radii, 10.0);

    const double* rho_ptr[1] = {rho_ref.data()};
    loop.observe(1, rho_ptr, 1);
    bool conv = true;
    loop.on_scf_converged(1, conv); // reference step, no moments supplied
    ASSERT_TRUE(loop.onsite_moments().empty());
    EXPECT_EQ(loop.last_audit_line().find("onsite="), std::string::npos);

    // Supply the esolver-side moments (indexed by global atom; the constraint
    // fragment is atom 0 only) and run one more converged SCF.
    loop.set_onsite_moments({3.5, -3.5, 0.25});
    std::vector<double> rho;
    fill_rho_mock(loop.mu(), rho);
    const double* rp[1] = {rho.data()};
    loop.observe(2, rp, 1);
    conv = true;
    loop.on_scf_converged(2, conv);
    ASSERT_EQ(loop.last_audit().onsite.size(), 1u);
    EXPECT_DOUBLE_EQ(loop.last_audit().onsite[0], 3.5);
    EXPECT_NE(loop.last_audit_line().find("onsite=3.5"), std::string::npos);
    // Informational only: the moments never touch the solver state.
    EXPECT_TRUE(loop.enabled());
}
// ---------------------------------------------------------------------------
// Dual-iteration schedule (plan 2026-09-11-dual-iteration-strategy.md, §2).
// Six tests cover the new INNER schedule: gating + budget, the mix-reset
// contract, the settle check (pass and bounce), the anti-fake-convergence
// discipline, the OUTER legacy contract, and the configuration guards.
// ---------------------------------------------------------------------------

TEST_F(ConstraintLoopTest, InnerScheduleGating)
{
    // Gate (drho < constraint_inner_thr) + budget (constraint_inner_nmax).
    // The mock response is exactly linear, so a loose conv_tol would
    // converge after two updates; thr is pinned at 1e-12 and the target is
    // far enough (delta = 0.5, step_max = 0.05) that the solver keeps
    // stepping RUNNING, which is what lets the budget be exercised.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.5, 5.0, 1e-12);
    cfg.mu_schedule = "inner";
    cfg.inner_thr = 1e-4;
    cfg.inner_nmax = 3;
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    ASSERT_TRUE(loop.enabled());
    EXPECT_EQ(loop.schedule(), "inner");
    EXPECT_TRUE(loop.inner_active());
    EXPECT_EQ(loop.inner_steps(), 0);

    std::vector<double> rho;
    bool conv = false;
    // Reference SCF (mu = 0): the OUTER path records it and takes the first
    // history-free probe step.  The gate is open (drho << inner_thr) but the
    // reference phase must stay free of in-SCF updates.
    EXPECT_FALSE(mock_iteration(loop, rho, 1, 0.0, 1e-9, conv));
    ASSERT_FALSE(conv);
    ASSERT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    EXPECT_EQ(loop.inner_steps(), 0);

    // drho above the gate -> the hook is inert: no update, no reset request.
    // Above the gate the SCF is by construction not converged (inner_thr >
    // scf_thr), so the esolver hands in scf_conv_in = false and no outer step
    // runs either.
    EXPECT_FALSE(mock_iteration(loop, rho, 2, loop.mu()[0], 1e-1, conv,
                                /*scf_conv_in=*/false));
    EXPECT_EQ(loop.inner_steps(), 0);
    EXPECT_FALSE(conv);

    // drho below the gate -> exactly one in-SCF update (and one reset).
    EXPECT_TRUE(mock_iteration(loop, rho, 3, loop.mu()[0], 1e-6, conv));
    EXPECT_EQ(loop.inner_steps(), 1);

    // Two more updates fit the budget (inner_nmax = 3) ...
    EXPECT_TRUE(mock_iteration(loop, rho, 4, loop.mu()[0], 1e-6, conv));
    EXPECT_TRUE(mock_iteration(loop, rho, 5, loop.mu()[0], 1e-6, conv));
    EXPECT_EQ(loop.inner_steps(), 3);

    // ... the fourth gated iteration is refused, and the run degrades to the
    // OUTER schedule loudly instead of spinning forever.
    EXPECT_FALSE(mock_iteration(loop, rho, 6, loop.mu()[0], 1e-6, conv));
    EXPECT_EQ(loop.inner_steps(), 3);
    EXPECT_FALSE(loop.inner_active()); // degraded, never silently continued
}

TEST_F(ConstraintLoopTest, InnerMixResetOnUpdate)
{
    // The mix-reset contract: on_iteration() returns true iff mu actually
    // moved, and the caller then resets the Broyden/DIIS history.  A step
    // that reports CONVERGED leaves mu untouched (MuSolver's convergence
    // check precedes any update), so it must NOT request a reset.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1, 5.0, 0.005);
    cfg.mu_schedule = "inner";
    cfg.inner_thr = 1e-4;
    cfg.inner_nmax = 10;
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    ASSERT_TRUE(loop.enabled());

    std::vector<double> rho;
    bool conv = false;
    // Reference SCF (SCF converged): the OUTER path records it and takes the
    // history-free probe step (mu = -0.05) but never resets the mixing.
    EXPECT_FALSE(mock_iteration(loop, rho, 1, 0.0, 1e-9, conv));
    ASSERT_FALSE(conv);
    const double mu_after_ref = loop.mu()[0];
    EXPECT_NEAR(mu_after_ref, -0.05, 1e-9);
    EXPECT_EQ(loop.inner_steps(), 0);

    // Above the gate: no update -> no reset request, mu untouched.
    EXPECT_FALSE(mock_iteration(loop, rho, 2, mu_after_ref, 1e-1, conv,
                                /*scf_conv_in=*/false));
    EXPECT_DOUBLE_EQ(loop.mu()[0], mu_after_ref);
    EXPECT_EQ(loop.inner_steps(), 0);

    // Below the gate, residual still above tol: mu moves -> exactly one
    // reset request (the fixed-point map changed).
    EXPECT_TRUE(mock_iteration(loop, rho, 3, mu_after_ref, 1e-6, conv,
                               /*scf_conv_in=*/false));
    EXPECT_EQ(loop.inner_steps(), 1);
    const double mu_star = loop.mu()[0];
    EXPECT_NEAR(mu_star, -0.1, 1e-9); // root of the normalized response

    // The next gated iteration lands on the target: CONVERGED without moving
    // mu -> no reset request.  The SCF is held open this iteration, so the
    // settle check is deferred to the next one.
    EXPECT_FALSE(mock_iteration(loop, rho, 4, mu_star, 1e-6, conv,
                                /*scf_conv_in=*/false));
    EXPECT_DOUBLE_EQ(loop.mu()[0], mu_star);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    EXPECT_EQ(loop.settle_failures(), 0);

    // Settle check on the relaxed density: the target survived -> the run is
    // genuinely done.  No outer step was ever taken after the reference
    // (outer_steps == 1), so the INNER path produced the convergence.
    EXPECT_FALSE(mock_iteration(loop, rho, 5, mu_star, 1e-9, conv));
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_EQ(loop.settle_failures(), 0);
    EXPECT_TRUE(conv); // the run really terminated here
    EXPECT_EQ(loop.outer_steps(), 1);
    // NOTE: the emitted "[constraint] settle step N ... (phase=constrained)"
    // header line goes to ofs_running and is not captured by this unit test;
    // it is verified end-to-end on cases 211/212 (spec
    // 2026-09-14-dual-iteration-inner-schedule.md section 3.5).
}

TEST_F(ConstraintLoopTest, InnerSettleCheck)
{
    // Settle check (§2.4): an inner CONVERGED verdict is provisional until
    // the density has settled with mu frozen.  A bounce withdraws the claim
    // and resumes the inner loop; a second bounce degrades to OUTER.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1, 5.0, 0.005);
    cfg.mu_schedule = "inner";
    cfg.inner_thr = 1e-4;
    cfg.inner_nmax = 10;
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    ASSERT_TRUE(loop.enabled());

    std::vector<double> rho;
    bool conv = false;
    // Reference SCF -> probe step to mu = -0.05.
    mock_iteration(loop, rho, 1, 0.0, 1e-9, conv);
    ASSERT_FALSE(conv);
    ASSERT_NEAR(loop.mu()[0], -0.05, 1e-9);

    // One gated update lands mu on the target (-0.1) but stays RUNNING.
    EXPECT_TRUE(mock_iteration(loop, rho, 2, loop.mu()[0], 1e-6, conv,
                               /*scf_conv_in=*/false));
    ASSERT_EQ(loop.inner_steps(), 1);
    ASSERT_NEAR(loop.mu()[0], -0.1, 1e-9);

    // Next gated iteration: residual is zero -> CONVERGED, settle armed.  The
    // SCF is held open (scf_conv_in = false) so the settle check is deferred
    // to the next iteration, where the density has had a chance to relax.
    EXPECT_FALSE(mock_iteration(loop, rho, 3, loop.mu()[0], 1e-6, conv,
                                /*scf_conv_in=*/false));
    ASSERT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    ASSERT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    ASSERT_EQ(loop.settle_failures(), 0);
    EXPECT_FALSE(conv);

    // Bounce #1: the relaxed density no longer meets the target (simulated
    // by observing a density built at a perturbed mu).  The verdict must be
    // withdrawn, the SCF kept running and the inner loop resumed.
    EXPECT_FALSE(mock_iteration(loop, rho, 4, -0.07, 1e-9, conv));
    EXPECT_EQ(loop.settle_failures(), 1);
    EXPECT_EQ(loop.status(), constraint::MuStatus::RUNNING);
    EXPECT_TRUE(loop.inner_active()); // one bounce: still INNER
    EXPECT_FALSE(conv);

    // The inner loop re-converges on the settled density (mu is still the
    // frozen -0.1).
    EXPECT_FALSE(mock_iteration(loop, rho, 5, loop.mu()[0], 1e-6, conv,
                                /*scf_conv_in=*/false));
    ASSERT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.settle_failures(), 1);

    // Bounce #2: two failed settle checks -> degrade to OUTER.  The SCF is
    // settled, so the outer secant takes over on this density immediately.
    mock_iteration(loop, rho, 6, -0.07, 1e-9, conv);
    EXPECT_EQ(loop.settle_failures(), 2);
    EXPECT_FALSE(loop.inner_active()); // degraded, loudly
    EXPECT_GT(loop.outer_steps(), 1);  // the OUTER path took over
}

TEST_F(ConstraintLoopTest, InnerAntiFakeConvergence)
{
    // T4a discipline in INNER mode: an open density gate is not target
    // convergence.  At mu = 0 the observed Q is the natural reference Q_ref,
    // which sits exactly 'delta' away from the target, so the reference
    // iteration must stay RUNNING (never CONVERGED) and the first gated
    // iteration must move mu away from zero.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.01, 5.0, 1e-12);
    cfg.mu_schedule = "inner";
    cfg.inner_thr = 1e-4;
    cfg.inner_nmax = 10;
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    ASSERT_TRUE(loop.enabled());

    std::vector<double> rho;
    bool conv = false;
    // Reference: gate open (drho = 1e-9 << inner_thr), yet no convergence
    // and no in-SCF update.
    EXPECT_FALSE(mock_iteration(loop, rho, 1, 0.0, 1e-9, conv));
    EXPECT_FALSE(conv);
    EXPECT_EQ(loop.status(), constraint::MuStatus::RUNNING);
    EXPECT_EQ(loop.inner_steps(), 0);
    // The free density really is off-target: this is why a stop here would
    // be a fake convergence.
    EXPECT_NEAR(loop.targets()[0] - loop.charges()[0], 0.01, 1e-10);
    EXPECT_LT(loop.mu()[0], 0.0); // already stepped away from mu = 0

    int iter = 1;
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 30)
    {
        ++iter;
        mock_iteration(loop, rho, iter, loop.mu()[0], 1e-9, conv);
        ++guard;
    }
    EXPECT_LT(guard, 30);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_NEAR(loop.mu()[0], -0.01, 1e-9); // root for delta = 0.01
    EXPECT_NEAR(loop.charges()[0], loop.targets()[0], 1e-10);
    EXPECT_GT(loop.inner_steps(), 0); // the INNER path did the work
}

TEST_F(ConstraintLoopTest, OuterLegacyBitIdentical)
{
    // OUTER is the default and must behave exactly as before the feature:
    // the per-iteration hook never fires (even with the gate wide open), the
    // mu trajectory and the outer-step count are unchanged.  Byte-level
    // regression is covered by ConvergesOnLinearResponse (untouched) still
    // passing; here the schedule-specific invariants are pinned.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.01); // defaults
    ASSERT_EQ(cfg.mu_schedule, "outer");
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    EXPECT_FALSE(loop.inner_active());
    EXPECT_EQ(loop.schedule(), "outer");

    std::vector<double> rho;
    bool conv = false;
    // drho = 0 would satisfy any positive gate: a sabotaged gate must still
    // not turn the OUTER schedule into an INNER one.
    EXPECT_FALSE(mock_iteration(loop, rho, 1, 0.0, 0.0, conv));
    ASSERT_FALSE(conv);
    EXPECT_EQ(loop.outer_steps(), 1);
    EXPECT_EQ(loop.inner_steps(), 0);

    int iter = 1;
    int guard = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && guard < 20)
    {
        ++iter;
        EXPECT_FALSE(mock_iteration(loop, rho, iter, loop.mu()[0], 0.0, conv));
        EXPECT_EQ(loop.inner_steps(), 0);
        EXPECT_EQ(loop.settle_failures(), 0);
        ++guard;
    }
    EXPECT_LT(guard, 20);
    EXPECT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_TRUE(conv);
    EXPECT_NEAR(loop.mu()[0], -0.01, 1e-6);
    EXPECT_EQ(loop.outer_steps(), guard + 1); // reference + one per iteration
}

TEST_F(ConstraintLoopTest, InnerGuards)
{
    // (1) Configuration guards: an unusable INNER schedule must abort
    // loudly rather than silently behave like OUTER or trip immediately.
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<std::string> warnings;
    std::string error;
    const std::string json = R"({"targets": [0.1], "atoms": [[0]]})";
    auto configure = [&](const std::string& schedule, const double thr,
                         const int nmax) {
        return constraint::configure_constraint(
            cfg, specs, warnings, true, "charge", "becke", "delta", json, 5.0,
            1e-4, 0.05, 0.0, 0.0, 3, 1, error, schedule, thr, nmax);
    };
    // Branch: an unknown schedule is refused (no silent OUTER fallback).
    EXPECT_EQ(configure("bogus", 1e-3, 20), constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_mu_schedule"), std::string::npos);
    // Branch: INNER with a non-positive density gate.
    EXPECT_EQ(configure("inner", 0.0, 20), constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_inner_thr"), std::string::npos);
    // Branch: INNER with a non-positive update budget.
    EXPECT_EQ(configure("inner", 1e-3, 0), constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_inner_nmax"), std::string::npos);
    // Branch: OUTER keeps ignoring both knobs -> the legacy runs stay valid.
    EXPECT_EQ(configure("outer", 0.0, 0), constraint::ConfigStatus::OK);
    EXPECT_EQ(configure("inner", 1e-3, 20), constraint::ConfigStatus::OK);

    // (2) Sabotage detector: the gate is a STRICT inequality.  A sabotage
    // that widened it (drho <= inner_thr) or zeroed it is exactly what this
    // boundary pins down: drho == inner_thr must NOT update, drho just below
    // it must.
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig lcfg = make_cfg(0.5, 5.0, 1e-12);
    lcfg.mu_schedule = "inner";
    lcfg.inner_thr = 1e-3;
    lcfg.inner_nmax = 10;
    loop.init(*ucell, rhopw, lcfg, radii, 10.0);

    std::vector<double> rho;
    bool conv = false;
    mock_iteration(loop, rho, 1, 0.0, 1e-9, conv); // reference: inert
    ASSERT_EQ(loop.inner_steps(), 0);
    const double mu_after_ref = loop.mu()[0];

    // The two boundary iterations hand in scf_conv_in = false so that the
    // OUTER secant cannot move mu: the assertion isolates the schedule gate.
    // drho exactly at the gate: strict '<' -> no update.
    EXPECT_FALSE(mock_iteration(loop, rho, 2, mu_after_ref, 1e-3, conv,
                                /*scf_conv_in=*/false));
    EXPECT_EQ(loop.inner_steps(), 0);
    EXPECT_DOUBLE_EQ(loop.mu()[0], mu_after_ref);

    // drho just below the gate: open -> exactly one update.
    EXPECT_TRUE(mock_iteration(loop, rho, 3, mu_after_ref, 1e-3 * (1.0 - 1e-9),
                               conv, /*scf_conv_in=*/false));
    EXPECT_EQ(loop.inner_steps(), 1);
    EXPECT_NE(loop.mu()[0], mu_after_ref);
}
