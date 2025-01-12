#include "gtest/gtest.h"

#include "NewtonGaussSolver.h"
#include "TaskF.h"

TEST(OptimizerTest, SingleVariableQuadraticFunction) {
    //f(x) = (x - 3)^2
    double a = 0.0;
    Variable x(&a);
    Constant c(3);
    Constant d(2);
    Subtraction e(&x, &c);
    Function *f = new Power(&e, &d);
    std::vector<Variable*> variables = { &x };
    LSMTask task({f}, variables);
    NewtonGaussSolver optimizer(10000); // maxIterations=1000
    optimizer.setTask(&task);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_TRUE(converged);

    EXPECT_NEAR(result[0], 3.0, 1e-2);

    EXPECT_NEAR(finalError, 0.0, 1e-4);
}
/*
TEST(OptimizerTest, MultiVariableQuadraticFunction) {
    // f(x, y) = (x - 2)^2 + (y + 5)^2
    double a = 1.0;
    double b = 2.0;
    Variable x(&a);
    Variable y(&b);
    Constant c(2.0);
    Constant d(5.0);
    Subtraction e(&x, &c);
    Addition g(&y, &d);
    Power h(&e, &c);
    Power i(&g, &c);
    Function* f = new Addition(&h, &i);
    std::vector<Variable*> variables = { &x, &y };
    LSMTask task( {f}, variables);

    NewtonGaussSolver optimizer(10000); // maxIterations=1000
    optimizer.setTask(&task);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    //EXPECT_TRUE(converged);

    EXPECT_NEAR(result[0], 2.0, 1e-2);
    EXPECT_NEAR(result[1], -5.0, 1e-2);

    EXPECT_NEAR(finalError, 0.0, 1e-4);
}

TEST(OptimizerTest, DoesNotConvergeWithHighLearningRate) {
    double a = 0.0;
    Variable x(&a);
    Constant b(1.0);
    Constant c(2.0);
    Subtraction d(&x, &b);
    Function* f = new Power(&d, &c); // (x - 1)^2
    std::vector<Variable*> variables = { &x };
    LSMTask task({f}, variables);

    NewtonGaussSolver optimizer(10000); // maxIterations=100
    optimizer.setTask(&task);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_FALSE(converged);
    EXPECT_GT(finalError, 1.0);
}

TEST(OptimizerTest, OptimizeWithoutSettingTask) {
    NewtonGaussSolver optimizer(10000);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_FALSE(converged);

    EXPECT_TRUE(result.empty());

    EXPECT_DOUBLE_EQ(finalError, 0.0);
}

TEST(OptimizerTest, AlreadyOptimal) {
    //f(x) = x - 4
    //f(x) = y + 5
    double a = 4.0;
    Variable x(&a);
    Constant b(4.0);
    Constant c(2.0);
    Subtraction g(&x, &b);
    Function* f = new Power(&g, &c);
    std::vector<Variable*> variables = { &x };
    LSMTask task({f}, variables);

    NewtonGaussSolver optimizer(10000);
    optimizer.setTask(&task);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_TRUE(converged);

    EXPECT_DOUBLE_EQ(result[0], 4.0);

    EXPECT_DOUBLE_EQ(finalError, 0.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}*/