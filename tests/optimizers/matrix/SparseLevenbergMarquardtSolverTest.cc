#include "gtest/gtest.h"

#include "SparseLSMTask.h"
#include "sparse/SparseLevenbergMarquardtSolver.h"

#include <cmath>

TEST(SparseLSMTaskTest, JacobianStoresOnlyNonZeroEntries) {
    double xValue = 3.0;
    double yValue = -1.0;
    double zValue = 5.0;

    Variable x(&xValue);
    Variable y(&yValue);
    Variable z(&zValue);

    Function* f1 = new Subtraction(x.clone(), new Constant(1.0));
    Function* f2 = new Addition(y.clone(), new Constant(2.0));
    Function* f3 = new Subtraction(new Addition(x.clone(), z.clone()), new Constant(4.0));

    SparseLSMTask task({f1, f2, f3}, {&x, &y, &z});

    auto [residuals, jacobian] = task.linearizeFunction();

    EXPECT_EQ(jacobian.nonZeros(), 4u);
    EXPECT_DOUBLE_EQ(residuals(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(residuals(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(residuals(2, 0), 4.0);
    EXPECT_DOUBLE_EQ(jacobian(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(0, 2), 0.0);
}

TEST(SparseLSMTaskTest, JacobianKeepsStructuralEntriesAtNumericalZero) {
    double xValue = 0.0;
    double yValue = 0.0;

    Variable x(&xValue);
    Variable y(&yValue);

    Function* f = new Multiplication(x.clone(), y.clone());

    SparseLSMTask task({f}, {&x, &y});

    auto [initialResiduals, initialJacobian] = task.linearizeFunction();

    EXPECT_EQ(initialJacobian.nonZeros(), 2u);
    EXPECT_DOUBLE_EQ(initialResiduals(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(initialJacobian(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(initialJacobian(0, 1), 0.0);

    task.setError({2.0, 3.0});
    auto [updatedResiduals, updatedJacobian] = task.linearizeFunction();

    EXPECT_EQ(updatedJacobian.nonZeros(), 2u);
    EXPECT_DOUBLE_EQ(updatedResiduals(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(updatedJacobian(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(updatedJacobian(0, 1), 2.0);
}

TEST(SparseLMSolverTest, ConvergesOnSparseLeastSquaresSystem) {
    double xValue = 0.0;
    double yValue = 0.0;
    double zValue = 0.0;

    Variable x(&xValue);
    Variable y(&yValue);
    Variable z(&zValue);

    Function* f1 = new Subtraction(x.clone(), new Constant(1.0));
    Function* f2 = new Addition(y.clone(), new Constant(2.0));
    Function* f3 = new Subtraction(z.clone(), new Constant(3.0));
    Function* f4 = new Subtraction(new Addition(x.clone(), z.clone()), new Constant(4.0));

    SparseLSMTask task({f1, f2, f3, f4}, {&x, &y, &z});
    SparseLMSolver optimizer(200, 1e-3, 1e-10, 1e-10);

    optimizer.setTask(&task);
    optimizer.optimize();

    const std::vector<double> result = optimizer.getResult();

    EXPECT_TRUE(optimizer.isConverged());
    EXPECT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0], 1.0, 1e-6);
    EXPECT_NEAR(result[1], -2.0, 1e-6);
    EXPECT_NEAR(result[2], 3.0, 1e-6);
    EXPECT_NEAR(optimizer.getCurrentError(), 0.0, 1e-8);
}

TEST(SparseLMSolverTest, DoesNotReportConvergenceAtStationaryPointWithResidual) {
    double xValue = 0.0;

    Variable x(&xValue);

    Function* residual = new Addition(
        new Power(x.clone(), new Constant(2.0)),
        new Constant(1.0));

    SparseLSMTask task({residual}, {&x});
    SparseLMSolver optimizer(50, 1e-3, 1e-10, 1e-10, 1e-10);

    optimizer.setTask(&task);
    optimizer.optimize();

    EXPECT_FALSE(optimizer.isConverged());
    EXPECT_EQ(optimizer.getIterationCount(), 0);
    EXPECT_NEAR(optimizer.getCurrentError(), 1.0, 1e-12);
}

TEST(SparseLMSolverTest, ReusingOptimizerResetsDampingForRankDeficientProblems) {
    double xValue = 0.0;
    double yValue = 0.0;

    Variable x(&xValue);
    Variable y(&yValue);

    Function* residual = new Subtraction(
        new Addition(x.clone(), y.clone()),
        new Constant(1.0));

    SparseLSMTask task({residual}, {&x, &y});
    SparseLMSolver optimizer(20, 1e-3, 1e-10, 1e-10, 1e-12);

    for (int iteration = 0; iteration < 900; ++iteration) {
        xValue = 0.0;
        yValue = 0.0;

        optimizer.setTask(&task);
        optimizer.optimize();

        EXPECT_TRUE(std::isfinite(optimizer.getCurrentError()));
    }
}
