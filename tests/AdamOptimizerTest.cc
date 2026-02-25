#include "gtest/gtest.h"
#include "AdamOptimizer.h"
#include "LSMFORLMTask.h"

TEST(AdamOptimizerTest, SingleVariableQuadratic) {
    double a = 0.0;
    Variable x(&a);
    Function* f = new Power(
        new Subtraction(x.clone(), new Constant(3.0)),
        new Constant(2.0));

    std::vector<Variable*> vars = {&x};
    LSMFORLMTask task({f}, vars);

    AdamOptimizer optimizer(0.1, 0.9, 0.999, 1e-8, 1e-6, 2000);
    optimizer.setTask(&task);
    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    double error = optimizer.getCurrentError();

    EXPECT_NEAR(result[0], 3.0, 1e-1);
    EXPECT_NEAR(error, 0.0, 0.1);
}

TEST(AdamOptimizerTest, TwoVariableQuadratic) {
    double a = 1.0;
    double b = 2.0;
    Variable x(&a);
    Variable y(&b);

    Function* f1 = new Subtraction(x.clone(), new Constant(2.0));
    Function* f2 = new Addition(y.clone(), new Constant(5.0));

    std::vector<Variable*> vars = {&x, &y};
    LSMFORLMTask task({f1, f2}, vars);

    AdamOptimizer optimizer(0.1, 0.9, 0.999, 1e-8, 1e-6, 2000);
    optimizer.setTask(&task);
    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();

    EXPECT_NEAR(result[0], 2.0, 1e-1);
    EXPECT_NEAR(result[1], -5.0, 1e-1);
}

TEST(AdamOptimizerTest, AlreadyOptimal) {
    double a = 3.0;
    Variable x(&a);
    Function* f = new Power(
        new Subtraction(x.clone(), new Constant(3.0)),
        new Constant(2.0));

    std::vector<Variable*> vars = {&x};
    LSMFORLMTask task({f}, vars);

    AdamOptimizer optimizer;
    optimizer.setTask(&task);
    optimizer.optimize();

    EXPECT_TRUE(optimizer.isConverged());
    EXPECT_NEAR(optimizer.getCurrentError(), 0.0, 1e-6);
}

TEST(AdamOptimizerTest, Himmelblau) {
    double x_value = 0.0;
    double y_value = 0.0;
    Variable x(&x_value);
    Variable y(&y_value);

    Function* x_sq = new Power(x.clone(), new Constant(2.0));
    Function* term1 = new Power(
        new Subtraction(new Addition(x_sq, y.clone()), new Constant(11.0)),
        new Constant(2.0));
    Function* y_sq = new Power(y.clone(), new Constant(2.0));
    Function* term2 = new Power(
        new Subtraction(new Addition(x.clone(), y_sq), new Constant(7.0)),
        new Constant(2.0));
    Function* himmelblau = new Addition(term1, term2);

    std::vector<Variable*> vars = {&x, &y};
    LSMFORLMTask task({himmelblau}, vars);

    AdamOptimizer optimizer(0.01, 0.9, 0.999, 1e-8, 1e-8, 10000);
    optimizer.setTask(&task);
    optimizer.optimize();

    double error = optimizer.getCurrentError();
    EXPECT_LT(error, 1.0);

    delete himmelblau;
}
