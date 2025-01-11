//
// Created by Eugene Bychkov on 11.11.2024.
//
#include "gtest/gtest.h"
#include "LMForTest.h"
#include "ErrorFunctions.h"

TEST(OptimizerTestLMS, Himmelblau){
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
    LSMFORLMTask task({himmelblau}, variables);
    LevenbergMarquardtSolver optimizer;
    optimizer.setTask(&task);
    optimizer.optimize();
    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    EXPECT_TRUE(converged);
}

TEST(OptimizerTestLMS, PerpendicularityTest) {
    // Define variables for the endpoints of the two line segments
    double x1_value = 20.0, y1_value = 20.0;
    double x2_value = 30.0, y2_value = 30.0;
    double x3_value = 20.0, y3_value = 30.0;
    double x4_value = 30.0, y4_value = 40.0;

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
    Function* min1 = new Subtraction(x1, x2);
    Function* min2 = new Subtraction(y1, y2);
    Function* pow1 = new Power(min1, new Constant(2));
    Function* pow2 = new Power(min2, new Constant(2));
    Function* summ = new Addition(pow1, pow2);
    Function* sqqrt = new Power(summ, new Constant(0.5));
    Function* sub2 =  new Subtraction(sqqrt, new Constant(20));
    Function* min11 = new Subtraction(x3, x4);
    Function* min22 = new Subtraction(y3, y4);
    Function* pow11 = new Power(min11, new Constant(2));
    Function* pow22 = new Power(min22, new Constant(2));
    Function* summ1 = new Addition(pow11, pow22);
    Function* sqqrt1 = new Power(summ1, new Constant(0.5));
    Function* sub21 =  new Subtraction(sqqrt1, new Constant(20));
    // Define the task with the error function and the variables
    std::vector<Variable*> variables = {x1, y1, x2, y2, x3, y3, x4, y4};
    LSMFORLMTask task({dot_product, sub2, sub21}, variables);

    // Initialize optimizer and run optimization
    LevenbergMarquardtSolver optimizer; // maxIterations=1000
    optimizer.setTask(&task);
    optimizer.optimize();
    // Get the result and check for convergence
    std::vector<double> result = optimizer.getResult();
    for (int i = 0; i < result.size(); i++) {
        std::cout << result[i] << std::endl;
    }
    bool converged = optimizer.isConverged();

    // Expect the optimizer to have converged
    EXPECT_TRUE(converged);
}
TEST(TestsForCAD, PerpLenghtSetTest) {
    // Define variables for the endpoints of the two line segments
    double x1_value = 20.0, y1_value = 20.0;
    double x2_value = 30.0, y2_value = 30.0;
    double x3_value = 20.0, y3_value = 30.0;
    double x4_value = 30.0, y4_value = 40.0;
    std::vector<Variable*> variables = {
            new Variable(&x1_value), new Variable(&y1_value),
            new Variable(&x2_value), new Variable(&y2_value),
            new Variable(&x3_value), new Variable(&y3_value),
            new Variable(&x4_value), new Variable(&y4_value),
    };
    std::vector<Variable*> section1 = {
            variables[0], variables[1], variables[2], variables[3]
    };
    std::vector<Variable*> section2 = {
            variables[4], variables[5], variables[6], variables[7]
    };
    std::vector<Variable*> ppReq = {
            variables[0], variables[1], variables[4], variables[5]
    };
    PointPointDistanceError* f1 = new PointPointDistanceError(section1, 100);
    PointPointDistanceError* f2 = new PointPointDistanceError(section2, 100);
    SectionSectionPerpendicularError* f3 = new SectionSectionPerpendicularError(variables);
    PointOnPointError* f4 = new PointOnPointError(ppReq);
    LSMFORLMTask task({f1,f2,f3,f4}, variables);
    // Initialize optimizer and run optimization
    LevenbergMarquardtSolver optimizer; // maxIterations=1000
    optimizer.setTask(&task);
    optimizer.optimize();
    std::vector<double> result = optimizer.getResult();
    for (int i = 0; i < result.size(); i++) {
        std::cout << result[i] << std::endl;
    }
    bool converged = optimizer.isConverged();
    double error = optimizer.getCurrentError();
    std::cout << error << std::endl;
    EXPECT_TRUE(converged);
    EXPECT_NEAR(error, 0.0,1e-6);
}
TEST(TestsForCAD, PerpLenghtSetTestBigSize) {
    // Define variables for the endpoints of the two line segments
    double x1_value = 200.0, y1_value = 200.0;
    double x2_value = 300.0, y2_value = 300.0;
    double x3_value = 200.0, y3_value = 300.0;
    double x4_value = 300.0, y4_value = 400.0;
    std::vector<Variable*> variables = {
            new Variable(&x1_value), new Variable(&y1_value),
            new Variable(&x2_value), new Variable(&y2_value),
            new Variable(&x3_value), new Variable(&y3_value),
            new Variable(&x4_value), new Variable(&y4_value),
    };
    std::vector<Variable*> section1 = {
            variables[0], variables[1], variables[2], variables[3]
    };
    std::vector<Variable*> section2 = {
            variables[4], variables[5], variables[6], variables[7]
    };
    std::vector<Variable*> ppReq = {
            variables[0], variables[1], variables[4], variables[5]
    };
    PointPointDistanceError* f1 = new PointPointDistanceError(section1, 100);
    PointPointDistanceError* f2 = new PointPointDistanceError(section2, 100);
    SectionSectionPerpendicularError* f3 = new SectionSectionPerpendicularError(variables);
    PointOnPointError* f4 = new PointOnPointError(ppReq);
    LSMFORLMTask task({f1,f2,f3,f4}, variables);
    // Initialize optimizer and run optimization
    LevenbergMarquardtSolver optimizer; // maxIterations=1000
    optimizer.setTask(&task);
    optimizer.optimize();
    std::vector<double> result = optimizer.getResult();
    for (int i = 0; i < result.size(); i++) {
        std::cout << result[i] << std::endl;
    }
    bool converged = optimizer.isConverged();
    double error = optimizer.getCurrentError();
    std::cout << error << std::endl;
    EXPECT_TRUE(converged);
    EXPECT_NEAR(error, 0.0,1e-6);
}
TEST(TestsForCAD, PPDistanceTest) {
    // Define variables for the endpoints of the two line segments
    double x1_value = 20.0, y1_value = 20.0;
    double x2_value = 30.0, y2_value = 30.0;
    double x3_value = 200.0, y3_value = 30.0;
    double x4_value = 300.0, y4_value = 400.0;
    std::vector<Variable*> variables = {
            new Variable(&x1_value), new Variable(&y1_value),
            new Variable(&x2_value), new Variable(&y2_value),
            new Variable(&x3_value), new Variable(&y3_value),
            new Variable(&x4_value), new Variable(&y4_value),
    };
    std::vector<Variable*> ppReq = {
            variables[0], variables[1], variables[2], variables[3]
    };
    PointPointDistanceError* f4 = new PointPointDistanceError(ppReq, 20);
    LSMFORLMTask task({f4}, ppReq);
    // Initialize optimizer and run optimization
    LevenbergMarquardtSolver optimizer; // maxIterations=1000
    optimizer.setTask(&task);
    optimizer.optimize();
    std::vector<double> result = optimizer.getResult();
    for (int i = 0; i < result.size(); i++) {
        std::cout << result[i] << std::endl;
    }
    bool converged = optimizer.isConverged();
    double error = optimizer.getCurrentError();
    std::cout << error << std::endl;
    EXPECT_TRUE(converged);
    EXPECT_NEAR(error, 0.0,1e-6);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}