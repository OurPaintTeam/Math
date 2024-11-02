//
// Created by Eugene Bychkov on 25.10.2024.
//
#include "Function.h"
#include <gtest/gtest.h>
#include <cmath>

bool almost_equal(double a, double b, double epsilon = 1e-6) {
    return std::fabs(a - b) < epsilon;
}

class FunctionTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FunctionTest, TestSquareFunction) {
    double x_val = 3.0;
    Variable x(&x_val);
    Function* f = new Power(x.clone(), new Constant(2.0));
    double expected_f = 9.0;
    double computed_f = f->evaluate();
    EXPECT_TRUE(almost_equal(computed_f, expected_f));
    Function* df_dx = f->derivative(&x);
    double expected_df_dx = 6.0;
    double computed_df_dx = df_dx->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
    delete df_dx;
    delete f;
}

TEST_F(FunctionTest, TestSumOfSquares) {
    double x_val = 3.0;
    double y_val = 4.0;
    Variable x(&x_val);
    Variable y(&y_val);
    Function* x_squared = new Power(x.clone(), new Constant(2.0));
    Function* y_squared = new Power(y.clone(), new Constant(2.0));
    Function* f = new Addition(x_squared, y_squared);
    double expected_f = 25.0;
    double computed_f = f->evaluate();
    EXPECT_TRUE(almost_equal(computed_f, expected_f));
    Function* df_dx = f->derivative(&x);
    double expected_df_dx = 6.0;
    double computed_df_dx = df_dx->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
    Function* df_dy = f->derivative(&y);
    double expected_df_dy = 8.0;
    double computed_df_dy = df_dy->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dy, expected_df_dy));
    delete df_dy;
    delete df_dx;
    delete f;
}

TEST_F(FunctionTest, TestExponentialFunction) {
    double x_val = 2.0;
    double y_val = 3.0;
    Variable x(&x_val);
    Variable y(&y_val);
    Function* xy = new Multiplication(x.clone(), y.clone());
    Function* f = new Exponential(xy);
    double expected_f = std::exp(6.0);
    double computed_f = f->evaluate();
    EXPECT_TRUE(almost_equal(computed_f, expected_f));
    Function* df_dx = f->derivative(&x);
    double expected_df_dx = y_val * expected_f;
    double computed_df_dx = df_dx->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
    Function* df_dy = f->derivative(&y);
    double expected_df_dy = x_val * expected_f;
    double computed_df_dy = df_dy->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dy, expected_df_dy));
    delete df_dy;
    delete df_dx;
    delete f;
}

TEST_F(FunctionTest, TestDivisionFunction) {
    double x_val = 5.0;
    double y_val = 2.0;
    Variable x(&x_val);
    Variable y(&y_val);
    Function* numerator = new Addition(x.clone(), y.clone());
    Function* denominator = new Subtraction(x.clone(), y.clone());
    Function* f = new Division(numerator, denominator);
    double expected_f = 7.0 / 3.0;
    double computed_f = f->evaluate();
    EXPECT_TRUE(almost_equal(computed_f, expected_f));
    Function* df_dx = f->derivative(&x);
    double expected_df_dx = (-2.0 * y_val) / std::pow(x_val - y_val, 2);
    double computed_df_dx = df_dx->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
    Function* df_dy = f->derivative(&y);
    double expected_df_dy = (2.0 * x_val) / std::pow(x_val - y_val, 2);
    double computed_df_dy = df_dy->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dy, expected_df_dy));
    delete df_dy;
    delete df_dx;
    delete f;
}

TEST_F(FunctionTest, TestMultipleVariablesFunction) {
    double x_val = 2.0;
    double y_val = 3.0;
    double z_val = 4.0;
    Variable x(&x_val);
    Variable y(&y_val);
    Variable z(&z_val);
    Function* xy = new Multiplication(x.clone(), y.clone());
    Function* z_squared = new Power(z.clone(), new Constant(2.0));
    Function* f = new Addition(xy, z_squared);
    double expected_f = 22.0;
    double computed_f = f->evaluate();
    EXPECT_TRUE(almost_equal(computed_f, expected_f));
    Function* df_dx = f->derivative(&x);
    double expected_df_dx = y_val;
    double computed_df_dx_val = df_dx->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dx_val, expected_df_dx));
    Function* df_dy = f->derivative(&y);
    double expected_df_dy = x_val;
    double computed_df_dy_val = df_dy->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dy_val, expected_df_dy));
    Function* df_dz = f->derivative(&z);
    double expected_df_dz = 2.0 * z_val;
    double computed_df_dz_val = df_dz->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dz_val, expected_df_dz));
    delete df_dz;
    delete df_dy;
    delete df_dx;
    delete f;
}

TEST_F(FunctionTest, TestLogarithmFunction) {
    double x_val = std::exp(1.0);
    Variable x(&x_val);
    Function* f = new Logarithm(x.clone());
    double expected_f = 1.0;
    double computed_f = f->evaluate();
    EXPECT_TRUE(almost_equal(computed_f, expected_f));
    Function* df_dx = f->derivative(&x);
    double expected_df_dx = 1.0 / x_val;
    double computed_df_dx = df_dx->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
    delete df_dx;
    delete f;
}

TEST_F(FunctionTest, TestExponentialOfSquareFunction) {
    double x_val = 1.0;
    Variable x(&x_val);
    Function* x_squared = new Power(x.clone(), new Constant(2.0));
    Function* f = new Exponential(x_squared);
    double expected_f = std::exp(1.0);
    double computed_f = f->evaluate();
    EXPECT_TRUE(almost_equal(computed_f, expected_f));
    Function* df_dx = f->derivative(&x);
    double expected_df_dx = 2.0 * x_val * expected_f;
    double computed_df_dx = df_dx->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
    delete df_dx;
    delete f;
}


TEST_F(FunctionTest, TestComplexFunction) {
    double x_val = 1.0;
    double y_val = 2.0;
    Variable x(&x_val);
    Variable y(&y_val);
    Function* x_squared = new Power(x.clone(), new Constant(2.0));
    Function* y_squared = new Power(y.clone(), new Constant(2.0));
    Function* sum = new Addition(x_squared, y_squared);
    Function* xy = new Multiplication(x.clone(), y.clone());
    Function* exponential = new Exponential(xy);
    Function* f = new Multiplication(sum, exponential);
    double expected_f = (std::pow(x_val, 2) + std::pow(y_val, 2)) * std::exp(x_val * y_val);
    double computed_f = f->evaluate();
    EXPECT_TRUE(almost_equal(computed_f, expected_f));
    Function* df_dx = f->derivative(&x);

    double expected_df_dx = (2.0 * x_val) * std::exp(x_val * y_val) +
                            (std::pow(x_val, 2) + std::pow(y_val, 2)) * y_val * std::exp(x_val * y_val);
    double computed_df_dx = df_dx->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));

    Function* df_dy = f->derivative(&y);
    double expected_df_dy = (2.0 * y_val) * std::exp(x_val * y_val) +
                            (std::pow(x_val, 2) + std::pow(y_val, 2)) * x_val * std::exp(x_val * y_val);
    double computed_df_dy = df_dy->evaluate();
    EXPECT_TRUE(almost_equal(computed_df_dy, expected_df_dy));

    delete df_dy;
    delete df_dx;
    delete f;
}

TEST_F(FunctionTest, TestDivisionByZero) {
    double x_val = 1.0;
    double y_val = 1.0;
    Variable x(&x_val);
    Variable y(&y_val);

    Function* numerator = new Constant(1.0);
    Function* denominator = new Subtraction(y.clone(), y.clone());
    Function* f = new Division(numerator, denominator);

    EXPECT_THROW({
                     f->evaluate();
                 }, std::runtime_error);

    delete f;
}

TEST_F(FunctionTest, TestLogarithmOfNonPositive) {
    double x_val = -1.0;
    Variable x(&x_val);

    Function* f = new Logarithm(x.clone());

    EXPECT_THROW({
                     f->evaluate();
                 }, std::runtime_error);

    delete f;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}