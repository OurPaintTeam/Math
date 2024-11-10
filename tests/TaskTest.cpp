//
// Created by Eugene Bychkov on 25.10.2024.
//
#include <gtest/gtest.h>
#include "TaskF.h"
#include "Function.h"

TEST(TaskFTest, SingleVariableQuadratic) {
    double x_value = 3.0;
    Variable x(&x_value);
    std::vector<Variable*> variables = {&x};

    Constant* exponent = new Constant(2.0);
    Power* function = new Power(&x, exponent);

    TaskF task(function, variables);

    double error = task.getError();
    EXPECT_DOUBLE_EQ(error, x_value * x_value);

    Matrix<> grad = task.gradient();
    EXPECT_EQ(grad.rows_size(), 1);
    EXPECT_EQ(grad.cols_size(), 1);
    EXPECT_DOUBLE_EQ(grad(0, 0), 2 * x_value);

    Matrix<> hessian = task.hessian();
    EXPECT_EQ(hessian.rows_size(), 1);
    EXPECT_EQ(hessian.cols_size(), 1);
    EXPECT_DOUBLE_EQ(hessian(0, 0), 2.0);
}
TEST(TaskFFTest, TwoVariableQuadratic) {
    double x_value = 3.0;
    double y_value = 4.0;
    Variable x(&x_value);
    Variable y(&y_value);
    std::vector<Variable*> variables = {&x, &y};

    Power* x_squared = new Power(&x, new Constant(2.0));
    Power* y_squared = new Power(&y, new Constant(2.0));
    Addition* function = new Addition(x_squared, y_squared);

    TaskF task(function, variables);

    double error = task.getError();
    EXPECT_DOUBLE_EQ(error, x_value * x_value + y_value * y_value);

    Matrix<> grad = task.gradient();
    EXPECT_EQ(grad.rows_size(), 2);
    EXPECT_EQ(grad.cols_size(), 1);
    EXPECT_DOUBLE_EQ(grad(0, 0), 2 * x_value);
    EXPECT_DOUBLE_EQ(grad(1, 0), 2 * y_value);

    Matrix<> hessian = task.hessian();
    EXPECT_EQ(hessian.rows_size(), 2);
    EXPECT_EQ(hessian.cols_size(), 2);
    EXPECT_DOUBLE_EQ(hessian(0, 0), 2.0); // d^2f/dx^2
    EXPECT_DOUBLE_EQ(hessian(0, 1), 0.0); // d^2f/dxdy
    EXPECT_DOUBLE_EQ(hessian(1, 0), 0.0); // d^2f/dydx
    EXPECT_DOUBLE_EQ(hessian(1, 1), 2.0); // d^2f/dy^2
}

TEST(TaskFTest, TwoVariableMultiplication) {
    double x_value = 2.0;
    double y_value = 5.0;
    Variable x(&x_value);
    Variable y(&y_value);
    std::vector<Variable*> variables = {&x, &y};

    Multiplication* function = new Multiplication(&x, &y);

    TaskF task(function, variables);

    double error = task.getError();
    EXPECT_DOUBLE_EQ(error, x_value * y_value);

    Matrix<> grad = task.gradient();
    EXPECT_EQ(grad.rows_size(), 2);
    EXPECT_EQ(grad.cols_size(), 1);
    EXPECT_DOUBLE_EQ(grad(0, 0), y_value); // df/dx = y
    EXPECT_DOUBLE_EQ(grad(1, 0), x_value); // df/dy = x

    Matrix<> hessian = task.hessian();
    EXPECT_EQ(hessian.rows_size(), 2);
    EXPECT_EQ(hessian.cols_size(), 2);
    EXPECT_DOUBLE_EQ(hessian(0, 0), 0.0); // d^2f/dx^2
    EXPECT_DOUBLE_EQ(hessian(0, 1), 1.0); // d^2f/dxdy
    EXPECT_DOUBLE_EQ(hessian(1, 0), 1.0); // d^2f/dydx
    EXPECT_DOUBLE_EQ(hessian(1, 1), 0.0); // d^2f/dy^2
}