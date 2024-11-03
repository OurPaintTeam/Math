//
// Created by Eugene Bychkov on 03.11.2024.
//
#include <gtest/gtest.h>
#include "Function.h"
#include "LSMTask.h"

class TestFunction : public ::testing::Test {
protected:
    Variable* var;
    Constant* const2;
    Constant* const3;
    Function* addFunc;
    Function* powerFunc;

    void SetUp() override {
        double x = 0.0;
        var = new Variable(&x);
        const2 = new Constant(2.0);
        const3 = new Constant(3.0);
        addFunc = new Addition(var, const2);
        powerFunc = new Power(addFunc, const3);
    }

    void TearDown() override {
        delete var;
        delete const2;
        delete const3;
        delete addFunc;
        delete powerFunc;
    }
};

TEST_F(TestFunction, EvaluateConstant) {
    EXPECT_DOUBLE_EQ(const2->evaluate(), 2.0);
}

TEST_F(TestFunction, EvaluateVariable) {
    var->setValue(5.0);
    EXPECT_DOUBLE_EQ(var->evaluate(), 5.0);
}

TEST_F(TestFunction, EvaluateAddition) {
    var->setValue(5.0);
    EXPECT_DOUBLE_EQ(addFunc->evaluate(), 7.0); // 5 + 2 = 7
}

TEST_F(TestFunction, EvaluatePower) {
    var->setValue(2.0);
    EXPECT_DOUBLE_EQ(powerFunc->evaluate(), 64.0); // (2 + 2)^3 = 4^3 = 64
}

TEST_F(TestFunction, DerivativeVariable) {
    EXPECT_DOUBLE_EQ(var->derivative(var)->evaluate(), 1.0); // d(x)/dx = 1
}

TEST_F(TestFunction, DerivativeAddition) {
    EXPECT_DOUBLE_EQ(addFunc->derivative(var)->evaluate(), 1.0); // d(x + 2)/dx = 1
}

TEST_F(TestFunction, GradientLSMTask) {
    std::vector<Function*> functions = { powerFunc };
    std::vector<Variable*> variables = { var };
    LSMTask task(functions, variables);

    var->setValue(1.0);
    Matrix<> grad = task.gradient();
    EXPECT_DOUBLE_EQ(grad(0, 0), 1458.0); // d((x + 2)^6)/dx at x=1 := 1458
}

TEST_F(TestFunction, HessianLSMTask) {
    std::vector<Function*> functions = { powerFunc };
    std::vector<Variable*> variables = { var };
    LSMTask task(functions, variables);

    var->setValue(1.0);
    Matrix<> hess = task.hessian();
    EXPECT_DOUBLE_EQ(hess(0, 0), 2430.0); // d^2((x + 2)^3)/dx^2 at x=1
}

TEST_F(TestFunction, JacobianLSMTask) {
    std::vector<Function*> functions = { powerFunc };
    std::vector<Variable*> variables = { var };
    LSMTask task(functions, variables);

    Matrix<> jac = task.jacobian();
    EXPECT_DOUBLE_EQ(jac(0, 0), 192.0); // d((x + 2)^3)/dx at x=1
}

TEST_F(TestFunction, ComplexSystemGradient) {
    double x_val = 1.0, y_val = 2.0, z_val = 3.0;
    Variable* x = new Variable(&x_val);
    Variable* y = new Variable(&y_val);
    Variable* z = new Variable(&z_val);

    // Define some functions based on these variables
    Function* func1 = new Addition(new Multiplication(x, y), new Power(z, const2));  // x * y + z^2
    Function* func2 = new Power(new Addition(x, y), const3);                        // (x + y)^3
    Function* func3 = new Addition(new Power(x, const2), new Multiplication(y, z)); // x^2 + y * z

    // Group functions and variables for LSMTask
    std::vector<Function*> functions = { func1, func2, func3 };
    std::vector<Variable*> variables = { x, y, z };
    LSMTask task(functions, variables);

    // Set up the gradient matrix
    Matrix<> jac = task.jacobian();

    // Expected values might require analytical calculations or approximations
    EXPECT_EQ(jac(0, 0), 44);
    EXPECT_EQ(jac(1, 0), 1458);
    EXPECT_EQ(jac(2, 0), 28);
    EXPECT_EQ(jac(0, 1), 22);
    EXPECT_EQ(jac(1, 1), 1458);
    EXPECT_EQ(jac(2, 1), 42);
    EXPECT_EQ(jac(0, 2), 132);
    EXPECT_EQ(jac(1, 2), 0);
    EXPECT_EQ(jac(2, 2), 28);
}
