#include "gtest/gtest.h"

#include "SparseLSMTask.h"
#include "sparse/SparseDogLegSolver.h"

#include <memory>
#include <vector>

namespace {

struct SparseLinearProblem {
    double xValue = 0.0;
    double yValue = 0.0;
    double zValue = 0.0;
    Variable x;
    Variable y;
    Variable z;
    std::unique_ptr<SparseLSMTask> task;

    SparseLinearProblem()
        : x(&xValue), y(&yValue), z(&zValue)
    {
        std::vector<Function*> residuals;
        residuals.push_back(new Subtraction(x.clone(), new Constant(2.0)));
        residuals.push_back(new Addition(y.clone(), new Constant(3.0)));
        residuals.push_back(new Subtraction(z.clone(), new Constant(4.0)));
        residuals.push_back(new Subtraction(
            new Addition(x.clone(), z.clone()),
            new Constant(6.0)));

        task = std::make_unique<SparseLSMTask>(
            residuals,
            std::vector<Variable*>{&x, &y, &z});
    }
};

} // namespace

TEST(SparseDogLegSolverTest, ConvergesOnSparseLinearLeastSquaresSystem) {
    SparseLinearProblem problem;
    SparseDogLegSolver optimizer(50, 10.0, 100.0, 1e-4, 1e-10, 1e-10, 1e-12);

    optimizer.setTask(problem.task.get());
    optimizer.optimize();

    const std::vector<double> result = optimizer.getResult();
    EXPECT_TRUE(optimizer.isConverged());
    ASSERT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0], 2.0, 1e-6);
    EXPECT_NEAR(result[1], -3.0, 1e-6);
    EXPECT_NEAR(result[2], 4.0, 1e-6);
    EXPECT_NEAR(optimizer.getCurrentError(), 0.0, 1e-8);
}
