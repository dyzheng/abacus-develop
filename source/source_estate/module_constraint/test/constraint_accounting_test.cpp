#include "gtest/gtest.h"

#include <string>
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
#include "source_estate/module_constraint/constraint_accounting.h"
#include "source_estate/module_constraint/weight_grid.h"

class ConstraintAccountingTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell;
    ModulePW::PW_Basis* rhopw = nullptr;
    std::vector<double> radii;
    constraint::WeightGrid* wg = nullptr;

    void SetUp() override
    {
        Set_GlobalV_Default();
        ucell = make_h2o_ucell();
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 20, 20, 20);
        rhopw->distribute_r();
        radii = {1.5, 0.5, 0.5};
        wg = new constraint::WeightGrid(*ucell, rhopw, radii);
        wg->build();
    }

    void TearDown() override
    {
        delete wg;
        delete rhopw;
    }
};

TEST_F(ConstraintAccountingTest, EsconKnown)
{
    // E_con = sum mu_alpha (Q_alpha - t_alpha): 0.5*(-0.1) = -0.05.
    const std::vector<double> mu = {0.5};
    const std::vector<double> Q = {1.9};
    const std::vector<double> target = {2.0};
    const constraint::ConstraintAudit a =
        constraint::ConstraintAccounting::audit(*wg, mu, Q, target, 10.0);
    EXPECT_NEAR(a.e_con, -0.05, 1e-12);
    EXPECT_NEAR(a.max_residual, 0.1, 1e-12);
    EXPECT_DOUBLE_EQ(a.total_charge, 1.9);
    EXPECT_DOUBLE_EQ(a.nelec, 10.0);
    EXPECT_NEAR(a.maxdev, 0.0, 1e-10);
}

TEST_F(ConstraintAccountingTest, EsconMultiComponent)
{
    // mu={0.5,-0.25}, Q={1.9,3.2}, t={2.0,3.0}:
    // e_con = 0.5*(-0.1) + (-0.25)*(0.2) = -0.05 - 0.05 = -0.10
    const std::vector<double> mu = {0.5, -0.25};
    const std::vector<double> Q = {1.9, 3.2};
    const std::vector<double> target = {2.0, 3.0};
    const constraint::ConstraintAudit a =
        constraint::ConstraintAccounting::audit(*wg, mu, Q, target, 10.0);
    EXPECT_NEAR(a.e_con, -0.10, 1e-12);
    EXPECT_NEAR(a.max_residual, 0.2, 1e-12);
    ASSERT_EQ(a.residual.size(), 2u);
    EXPECT_NEAR(a.residual[0], -0.1, 1e-12);
    EXPECT_NEAR(a.residual[1], 0.2, 1e-12);
}

TEST_F(ConstraintAccountingTest, ZeroWhenConverged)
{
    // Q == target: no constraint energy, zero residual.
    const std::vector<double> mu = {0.7, -1.2};
    const std::vector<double> Q = {2.0, 3.0};
    const std::vector<double> target = {2.0, 3.0};
    const constraint::ConstraintAudit a =
        constraint::ConstraintAccounting::audit(*wg, mu, Q, target, 10.0);
    EXPECT_DOUBLE_EQ(a.e_con, 0.0);
    EXPECT_DOUBLE_EQ(a.max_residual, 0.0);
    EXPECT_DOUBLE_EQ(a.total_charge, 5.0);
}

TEST_F(ConstraintAccountingTest, AuditLineMachineReadable)
{
    const std::vector<double> mu = {0.5};
    const std::vector<double> Q = {1.9};
    const std::vector<double> target = {2.0};
    const constraint::ConstraintAudit a =
        constraint::ConstraintAccounting::audit(*wg, mu, Q, target, 10.0);
    const std::string line = constraint::ConstraintAccounting::audit_line(a);
    // key=value tokens present with the right numbers.
    EXPECT_NE(line.find("CONSTRAINT_AUDIT"), std::string::npos);
    EXPECT_NE(line.find("nconstraint=1"), std::string::npos);
    EXPECT_NE(line.find("e_con=-0.05"), std::string::npos);
    EXPECT_NE(line.find("max_residual=0.1"), std::string::npos);
    EXPECT_NE(line.find("total_charge=1.9"), std::string::npos);
    EXPECT_NE(line.find("nelec=10"), std::string::npos);
    // Per-constraint detail line.
    EXPECT_NE(line.find("c[0]"), std::string::npos);
    EXPECT_NE(line.find("q=1.9"), std::string::npos);
    EXPECT_NE(line.find("t=2"), std::string::npos);
    EXPECT_NE(line.find("mu=0.5"), std::string::npos);
    EXPECT_NE(line.find("res=-0.1"), std::string::npos);
}
