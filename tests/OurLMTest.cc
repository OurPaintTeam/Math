#include "gtest/gtest.h"
#include "LevenbergMarquardtSolver.h"
#include "ErrorFunctions.h"

TEST(OptimizerTestOURLMS, Himmelblau){
    double x_value = 0.0;
    double y_value = 0.0;
    Variable x(&x_value);
    Variable y(&y_value);
    Function* x_squared = new Power(x.clone(), new Constant(2.0));
    Function* term1 = new Power(new Subtraction(new Addition(x_squared, y.clone()), new Constant(11.0)), new Constant(2.0));
    Function* y_squared = new Power(y.clone(), new Constant(2.0));
    Function* term2 = new Power(new Subtraction(new Addition(x.clone(), y_squared), new Constant(7.0)), new Constant(2.0));
    Function* himmelblau = new Addition(term1, term2);
    std::vector<Variable*> variables = {&x, &y};
    LSMTask task({himmelblau}, variables);
    LMSolver optimizer;
    optimizer.setTask(&task);
    optimizer.optimize();
    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    EXPECT_TRUE(converged);
}

TEST(OptimizerTestOURLMS, PerpendicularityTest) {
    // Define variables for the endpoints of the two line segments
    double x1_value = 20.0, y1_value = 20.0;
    double x2_value = 30.0, y2_value = 30.0;
    double x3_value = 20.0, y3_value = 30.0;
    double x4_value = 30.0, y4_value = 40.0;

    Variable x1(&x1_value);
    Variable y1(&y1_value);
    Variable x2(&x2_value);
    Variable y2(&y2_value);
    Variable x3(&x3_value);
    Variable y3(&y3_value);
    Variable x4(&x4_value);
    Variable y4(&y4_value);

    // Define the vectors representing the two line segments
    Function* segment1_x = new Subtraction(x2.clone(), x1.clone()); // x2 - x1
    Function* segment1_y = new Subtraction(y2.clone(), y1.clone()); // y2 - y1
    Function* segment2_x = new Subtraction(x4.clone(), x3.clone()); // x4 - x3
    Function* segment2_y = new Subtraction(y4.clone(), y3.clone()); // y4 - y3

    // Perpendicularity condition: dot product of the two vectors should be 0
    Function* dot_product = new Addition(
            new Multiplication(segment1_x, segment2_x),
            new Multiplication(segment1_y, segment2_y)
    ); // (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)
    Function* min1 = new Subtraction(x1.clone(), x2.clone());
    Function* min2 = new Subtraction(y1.clone(), y2.clone());
    Function* pow1 = new Power(min1, new Constant(2));
    Function* pow2 = new Power(min2, new Constant(2));
    Function* summ = new Addition(pow1, pow2);
    Function* sqqrt = new Power(summ, new Constant(0.5));
    Function* sub2 =  new Subtraction(sqqrt, new Constant(20));
    Function* min11 = new Subtraction(x3.clone(), x4.clone());
    Function* min22 = new Subtraction(y3.clone(), y4.clone());
    Function* pow11 = new Power(min11, new Constant(2));
    Function* pow22 = new Power(min22, new Constant(2));
    Function* summ1 = new Addition(pow11, pow22);
    Function* sqqrt1 = new Power(summ1, new Constant(0.5));
    Function* sub21 =  new Subtraction(sqqrt1, new Constant(20));
    // Define the task with the error function and the variables
    std::vector<Variable*> variables = {&x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4};
    LSMTask task({dot_product, sub2, sub21}, variables);

    // Initialize optimizer and run optimization
    LMSolver optimizer; // maxIterations=1000
    optimizer.setTask(&task);
    optimizer.optimize();
    // Get the result and check for convergence
    std::vector<double> result = optimizer.getResult();
    //for (int i = 0; i < result.size(); i++) {
    //    std::cout << result[i] << std::endl;
    //}
    bool converged = optimizer.isConverged();

    // Expect the optimizer to have converged
    EXPECT_TRUE(converged);
}
/*
TEST(TestsForLMCAD, PerpLenghtSetTest) {
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
	std::vector<Variable*> variables2 = {
			variables[0]->clone(), variables[1]->clone(),
			variables[2]->clone(), variables[3]->clone(),
			variables[4]->clone(), variables[5]->clone(),
			variables[6]->clone(), variables[7]->clone()
	};
    std::vector<Variable*> section1 = {
            variables[0]->clone(), variables[1]->clone(), variables[2]->clone(), variables[3]->clone()
    };
    std::vector<Variable*> section2 = {
            variables[4]->clone(), variables[5]->clone(), variables[6]->clone(), variables[7]->clone()
    };
    std::vector<Variable*> ppReq = {
            variables[0]->clone(), variables[1]->clone(), variables[4]->clone(), variables[5]->clone()
    };
    PointPointDistanceError* f1 = new PointPointDistanceError(section1, 100);
    PointPointDistanceError* f2 = new PointPointDistanceError(section2, 100);
    SectionSectionPerpendicularError* f3 = new SectionSectionPerpendicularError(variables);
    PointOnPointError* f4 = new PointOnPointError(ppReq);
    LSMTask task({f1}, variables);
    // Initialize optimizer and run optimization
	LMSolver optimizer; // maxIterations=1000
	optimizer.setTask(&task);
	optimizer.optimize();
	std::vector<double> result = optimizer.getResult();
	//for (int i = 0; i < result.size(); i++) {
	//    std::cout << result[i] << std::endl;
	//}
	bool converged = optimizer.isConverged();
	double error = optimizer.getCurrentError();
	//std::cout << error << std::endl;
	EXPECT_TRUE(converged);
	EXPECT_NEAR(error, 0.0,1e-6);
}

TEST(TestsForLMCAD, PerpLenghtSetTestBigSize) {
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
            variables[0]->clone(), variables[1]->clone(), variables[2]->clone(), variables[3]->clone()
    };
    std::vector<Variable*> section2 = {
            variables[4]->clone(), variables[5]->clone(), variables[6]->clone(), variables[7]->clone()
    };
    std::vector<Variable*> ppReq = {
            variables[0]->clone(), variables[1]->clone(), variables[4]->clone(), variables[5]->clone()
    };
    PointPointDistanceError* f1 = new PointPointDistanceError(section1, 100);
    PointPointDistanceError* f2 = new PointPointDistanceError(section2, 100);
    SectionSectionPerpendicularError* f3 = new SectionSectionPerpendicularError(variables);
    PointOnPointError* f4 = new PointOnPointError(ppReq);
    LSMTask task({f1,f2,f3,f4}, variables);
    // Initialize optimizer and run optimization
    LMSolver optimizer; // maxIterations=1000
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
}*/