#include "gtest/gtest.h"

#include "SparseSystemTask.h"
#include "sparse/SparseNewtonSolver.h"

#include <memory>
#include <vector>

namespace {

struct SparseNonlinearSystem {
    double xValue = 3.0;
    double yValue = 0.0;
    double zValue = 1.0;
    Variable x;
    Variable y;
    Variable z;
    std::unique_ptr<SparseSystemTask> task;

    SparseNonlinearSystem()
        : x(&xValue), y(&yValue), z(&zValue)
    {
        std::vector<Function*> functions;
        functions.push_back(new Subtraction(
            new Multiplication(x.clone(), x.clone()),
            new Constant(4.0)));
        functions.push_back(new Addition(y.clone(), new Constant(3.0)));
        functions.push_back(new Subtraction(z.clone(), new Constant(4.0)));

        task = std::make_unique<SparseSystemTask>(
            functions,
            std::vector<Variable*>{&x, &y, &z});
    }
};

} // namespace

TEST(SparseNewtonSolverTest, ConvergesOnSparseNonlinearSystem) {
    SparseNonlinearSystem problem;
    SparseNewtonSolver optimizer(20, 1e-10, 1e-10, 1e-12, 1e-12);

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
