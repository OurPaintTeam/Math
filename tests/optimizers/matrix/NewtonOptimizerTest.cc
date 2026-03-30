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

TEST(OptimizerTest, PerpendicularityTest) {
    // Define variables for the endpoints of the two line segments
    double x1_value = 0.0, y1_value = 0.0;
    double x2_value = 1.0, y2_value = 0.0;
    double x3_value = 0.0, y3_value = 1.0;
    double x4_value = 1.0, y4_value = 1.0;

    Variable* x1 = new Variable(&x1_value);
    Variable* y1 = new Variable(&y1_value);
    Variable* x2 = new Variable(&x2_value);
    Variable* y2 = new Variable(&y2_value);
    Variable* x3 = new Variable(&x3_value);
    Variable* y3 = new Variable(&y3_value);
    Variable* x4 = new Variable(&x4_value);
    Variable* y4 = new Variable(&y4_value);

    // Define the vectors representing the two line segments
    Function* segment1_x = new Subtraction(x2, x1); // x2 - x1
    Function* segment1_y = new Subtraction(y2, y1); // y2 - y1
    Function* segment2_x = new Subtraction(x4, x3); // x4 - x3
    Function* segment2_y = new Subtraction(y4, y3); // y4 - y3

    // Perpendicularity condition: dot product of the two vectors should be 0
    Function* dot_product = new Addition(
            new Multiplication(segment1_x, segment2_x),
            new Multiplication(segment1_y, segment2_y)
    ); // (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)

    // The error function: we want the dot product to be 0 for perpendicularity
    Function* error = new Power(dot_product, new Constant(2.0)); // (dot_product)^2 to minimize error

    // Define the task with the error function and the variables
    std::vector<Variable*> variables = {x1, y1, x2, y2, x3, y3, x4, y4};
    TaskF task(error, variables);

    // Initialize optimizer and run optimization
    NewtonOptimizer optimizer(1000); // maxIterations=1000
    optimizer.setTask(&task);
    optimizer.optimize();

    // Get the result and check for convergence
    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();

    // Expect the optimizer to have converged
    EXPECT_TRUE(converged);

    // Optionally, verify the result (dot product should be close to 0)
    EXPECT_NEAR(result[0] * result[3] + result[1] * result[4], 0.0, 1e-5);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}