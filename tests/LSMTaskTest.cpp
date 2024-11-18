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
};

// Тесты для функции evaluate
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

// Тесты для производных
TEST_F(TestFunction, DerivativeVariable) {
    EXPECT_DOUBLE_EQ(var->derivative(var)->evaluate(), 1.0); // d(x)/dx = 1
}

TEST_F(TestFunction, DerivativeAddition) {
    EXPECT_DOUBLE_EQ(addFunc->derivative(var)->evaluate(), 1.0); // d(x + 2)/dx = 1
}

// Тесты для LSMTask
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
    var->setValue(1.0);
    Matrix<> jac = task.jacobian();
    EXPECT_DOUBLE_EQ(jac(0, 0), 27.0); // d((x + 2)^3)/dx at x=1
}

// Тесты для метода linearizeFunction
TEST_F(TestFunction, LinearizeFunctionSimple) {
    Variable* x = new Variable(new double(1.0));
    Function* linearFunc = x;  // f(x) = x

    std::vector<Function*> functions = { linearFunc };
    std::vector<Variable*> variables = { x };
    LSMTask task(functions, variables);

    auto [residuals, jacobian] = task.linearizeFunction();

    // Residual for f(x) = x at x = 1 is 1
    EXPECT_DOUBLE_EQ(residuals(0, 0), 1.0);

    // Derivative of f(x) = x is 1
    EXPECT_DOUBLE_EQ(jacobian(0, 0), 1.0);
}

TEST_F(TestFunction, LinearizeFunctionWithConstant) {
    Variable* x = new Variable(new double(2.0));
    Function* func = new Addition(x, const2);  // f(x) = x + 2

    std::vector<Function*> functions = { func };
    std::vector<Variable*> variables = { x };
    LSMTask task(functions, variables);

    auto [residuals, jacobian] = task.linearizeFunction();

    // Residual for f(x) = x + 2 at x = 2 is 4
    EXPECT_DOUBLE_EQ(residuals(0, 0), 4.0);

    // Derivative of f(x) = x + 2 is 1
    EXPECT_DOUBLE_EQ(jacobian(0, 0), 1.0);
}

TEST_F(TestFunction, LinearizeFunctionNonLinear) {
    Variable* x = new Variable(new double(2.0));
    Function* func = new Power(new Addition(x, const2), const3);  // f(x) = (x + 2)^3

    std::vector<Function*> functions = { func };
    std::vector<Variable*> variables = { x };
    LSMTask task(functions, variables);

    auto [residuals, jacobian] = task.linearizeFunction();

    // Residual for f(x) = (x + 2)^3 at x = 2 is 64
    EXPECT_DOUBLE_EQ(residuals(0, 0), 64.0);

    // Derivative of f(x) = (x + 2)^3 is 3 * (x + 2)^2, at x = 2 it is 72
    EXPECT_DOUBLE_EQ(jacobian(0, 0), 48.0);

}

TEST_F(TestFunction, LinearizeMultipleFunctions) {
    double x_val = 1.0, y_val = 2.0;
    Variable* x = new Variable(&x_val);
    Variable* y = new Variable(&y_val);

    Function* func1 = new Addition(x, y);  // f1(x, y) = x + y
    Function* func2 = new Multiplication(x, y);  // f2(x, y) = x * y

    std::vector<Function*> functions = { func1, func2 };
    std::vector<Variable*> variables = { x, y };
    LSMTask task(functions, variables);

    auto [residuals, jacobian] = task.linearizeFunction();

    // Residuals for f1(x, y) = x + y at (1, 2) is 3
    EXPECT_DOUBLE_EQ(residuals(0, 0), 3.0);

    // Residuals for f2(x, y) = x * y at (1, 2) is 2
    EXPECT_DOUBLE_EQ(residuals(1, 0), 2.0);

    // Jacobian for f1(x, y) = x + y is [1, 1]
    EXPECT_DOUBLE_EQ(jacobian(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(0, 1), 1.0);

    // Jacobian for f2(x, y) = x * y is [2, 1]
    EXPECT_DOUBLE_EQ(jacobian(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(jacobian(1, 1), 1.0);
}

TEST_F(TestFunction, LinearizeFunctionZeroValues) {
    Variable* x = new Variable(new double(0.0));
    Function* func = new Power(new Addition(x, const2), const3);  // f(x) = (x + 2)^3

    std::vector<Function*> functions = { func };
    std::vector<Variable*> variables = { x };
    LSMTask task(functions, variables);

    auto [residuals, jacobian] = task.linearizeFunction();

    // Residual for f(x) = (x + 2)^3 at x = 0 is 8
    EXPECT_DOUBLE_EQ(residuals(0, 0), 8.0);

    // Derivative of f(x) = (x + 2)^3 is 3 * (x + 2)^2, at x = 0 it is 12
    EXPECT_DOUBLE_EQ(jacobian(0, 0), 12.0);
}
