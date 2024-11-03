//
// Created by Eugene Bychkov on 03.11.2024.
//
#include "gtest/gtest.h"
#include "NewtonOptimizer.h"

TEST(OptimizerTest, Rosenbrock){
    double a = 1.0;
    double b = 100.0;
    double x_value = 0.0;
    double y_value = 0.0;
    Variable* x = new Variable(&x_value);
    Variable* y = new Variable(&y_value);
    Constant* a_constant = new Constant(a);
    Function* term1 = new Power(new Subtraction(a_constant, x), new Constant(2.0));
    Constant* b_constant = new Constant(b);
    Function* x_squared = new Power(x, new Constant(2.0));
    Function* term2 = new Multiplication(b_constant, new Power(new Subtraction(y, x_squared), new Constant(2.0)));
    Function* rosenbrock = new Addition(term1, term2);

    std::vector<Variable*> variables = {x, y};
    TaskF task(rosenbrock, variables);
    NewtonOptimizer optimizer(1000); // maxIterations=1000
    optimizer.setTask(&task);
    optimizer.optimize();
    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();
    EXPECT_TRUE(converged);
    EXPECT_NEAR(result[0], 1.0, 1e-2);
    EXPECT_NEAR(result[1], 1.0, 1e-2);
    EXPECT_NEAR(finalError, 0.0, 1e-4);
}
TEST(OptimizerTest, Himmelblau){
    double x_value = 0.0;
    double y_value = 0.0;
    Variable* x = new Variable(&x_value);
    Variable* y = new Variable(&y_value);
    Function* x_squared = new Power(x, new Constant(2.0));
    Function* term1 = new Power(new Subtraction(new Addition(x_squared, y), new Constant(11.0)), new Constant(2.0));
    Function* y_squared = new Power(y, new Constant(2.0));
    Function* term2 = new Power(new Subtraction(new Addition(x, y_squared), new Constant(7.0)), new Constant(2.0));
    Function* himmelblau = new Addition(term1, term2);
    std::vector<Variable*> variables = {x, y};
    TaskF task(himmelblau, variables);
    NewtonOptimizer optimizer(1000); // maxIterations=1000
    optimizer.setTask(&task);
    optimizer.optimize();
    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    EXPECT_TRUE(converged);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}