#include "gtest/gtest.h"
#include "StochasticGradientOptimizer.h"
#include "LSMFORLMTask.h"

TEST(StochasticGradientOptimizerTest, SingleVariableQuadratic) {
    double a = 0.0;
    Variable x(&a);
    Function* f = new Power(
        new Subtraction(x.clone(), new Constant(3.0)),
        new Constant(2.0));

    std::vector<Variable*> vars = {&x};
    LSMFORLMTask task({f}, vars);

    StochasticGradientOptimizer optimizer(0.001, 5000, 1e-6);
    optimizer.setTask(&task);
    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    double error = optimizer.getCurrentError();

    EXPECT_NEAR(result[0], 3.0, 0.5);
    EXPECT_LT(error, 1.0);

    delete f;
}

TEST(StochasticGradientOptimizerTest, TwoVariableQuadratic) {
    double a = 1.0;
    double b = 2.0;
    Variable x(&a);
    Variable y(&b);

    Function* f1 = new Subtraction(x.clone(), new Constant(2.0));
    Function* f2 = new Addition(y.clone(), new Constant(5.0));

    std::vector<Variable*> vars = {&x, &y};
    LSMFORLMTask task({f1, f2}, vars);

    StochasticGradientOptimizer optimizer(0.001, 10000, 1e-6);
    optimizer.setTask(&task);
    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    double error = optimizer.getCurrentError();

    EXPECT_NEAR(result[0], 2.0, 0.5);
    EXPECT_NEAR(result[1], -5.0, 0.5);
    EXPECT_LT(error, 1.0);

    delete f1;
    delete f2;
}

TEST(StochasticGradientOptimizerTest, AlreadyOptimal) {
    double a = 3.0;
    Variable x(&a);
    Function* f = new Power(
        new Subtraction(x.clone(), new Constant(3.0)),
        new Constant(2.0));

    std::vector<Variable*> vars = {&x};
    LSMFORLMTask task({f}, vars);

    StochasticGradientOptimizer optimizer(0.01, 1000, 1e-6);
    optimizer.setTask(&task);
    optimizer.optimize();

    EXPECT_TRUE(optimizer.isConverged());
    EXPECT_NEAR(optimizer.getCurrentError(), 0.0, 1e-6);

    delete f;
}
