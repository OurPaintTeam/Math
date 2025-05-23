#include "Function.h"
#include <gtest/gtest.h>
#include <cmath>

bool almost_equal(double a, double b, double epsilon = 1e-6) {
	return std::fabs(a - b) < epsilon;
}

TEST(FunctionTest, TestSquareFunction) {
	double x_val = 3.0;
	Variable* x = new Variable(&x_val);
	Function* f = new Power(x, new Constant(2.0));
	double expected_f = 9.0;
	double computed_f = f->evaluate();
	EXPECT_TRUE(almost_equal(computed_f, expected_f));
	Function* df_dx = f->derivative(x);
	double expected_df_dx = 6.0;
	double computed_df_dx = df_dx->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));

	delete df_dx;
	delete f;
}

TEST(FunctionTest, TestSumOfSquares) {
	double x_val = 3.0;
	double y_val = 4.0;
	Variable* x = new Variable(&x_val);
	Variable* y = new Variable(&y_val);
	Function* x_squared = new Power(x, new Constant(2.0));
	Function* y_squared = new Power(y, new Constant(2.0));
	Function* f = new Addition(x_squared, y_squared);
	double expected_f = 25.0;
	double computed_f = f->evaluate();
	EXPECT_TRUE(almost_equal(computed_f, expected_f));
	Function* df_dx = f->derivative(x);
	double expected_df_dx = 6.0;
	double computed_df_dx = df_dx->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
	Function* df_dy = f->derivative(y);
	double expected_df_dy = 8.0;
	double computed_df_dy = df_dy->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dy, expected_df_dy));

	delete f;
	delete df_dx;
	delete df_dy;
}

TEST(FunctionTest, TestExponentialFunction) {
	double x_val = 2.0;
	double y_val = 3.0;
	Variable* x = new Variable(&x_val);
	Variable* y = new Variable(&y_val);
	Function* xy = new Multiplication(x, y);
	Function* f = new Exp(xy);
	double expected_f = std::exp(6.0);
	double computed_f = f->evaluate();
	EXPECT_TRUE(almost_equal(computed_f, expected_f));
	Function* df_dx = f->derivative(x);
	double expected_df_dx = y_val * expected_f;
	double computed_df_dx = df_dx->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
	Function* df_dy = f->derivative(y);
	double expected_df_dy = x_val * expected_f;
	double computed_df_dy = df_dy->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dy, expected_df_dy));

	delete f;
	delete df_dx;
	delete df_dy;
}
//---------------------------
TEST(FunctionTest, TestDivisionFunction) {
	double x_val = 5.0;
	double y_val = 2.0;
	Variable* x = new Variable(&x_val);
	Variable* y = new Variable(&y_val);
	Function* numerator = new Addition(x, y);
	Function* denominator = new Subtraction(x->clone(), y->clone());
	Function* f = new Division(numerator, denominator);
	double expected_f = 7.0 / 3.0;
	double computed_f = f->evaluate();
	EXPECT_TRUE(almost_equal(computed_f, expected_f));
	Function* df_dx = f->derivative(x);
	double expected_df_dx = (-2.0 * y_val) / std::pow(x_val - y_val, 2);
	double computed_df_dx = df_dx->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));
	Function* df_dy = f->derivative(y);
	double expected_df_dy = (2.0 * x_val) / std::pow(x_val - y_val, 2);
	double computed_df_dy = df_dy->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dy, expected_df_dy));

	delete f;
	delete df_dx;
	delete df_dy;
}

TEST(FunctionTest, TestMultipleVariablesFunction) {
	double x_val = 2.0;
	double y_val = 3.0;
	double z_val = 4.0;
	Variable* x = new Variable(&x_val);
	Variable* y = new Variable(&y_val);
	Variable* z = new Variable(&z_val);
	Function* xy = new Multiplication(x, y);
	Function* z_squared = new Power(z, new Constant(2.0));
	Function* f = new Addition(xy, z_squared);
	double expected_f = 22.0;
	double computed_f = f->evaluate();
	EXPECT_TRUE(almost_equal(computed_f, expected_f));
	Function* df_dx = f->derivative(x);
	double expected_df_dx = y_val;
	double computed_df_dx_val = df_dx->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dx_val, expected_df_dx));
	Function* df_dy = f->derivative(y);
	double expected_df_dy = x_val;
	double computed_df_dy_val = df_dy->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dy_val, expected_df_dy));
	Function* df_dz = f->derivative(z);
	double expected_df_dz = 2.0 * z_val;
	double computed_df_dz_val = df_dz->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dz_val, expected_df_dz));

	delete f;
	delete df_dx;
	delete df_dy;
	delete df_dz;
}

TEST(FunctionTest, TestLnFunction) {
	double x_val = std::exp(1.0);
	Variable* x = new Variable(&x_val);
	Function* f = new Ln(x);
	double expected_f = 1.0;
	double computed_f = f->evaluate();
	EXPECT_TRUE(almost_equal(computed_f, expected_f));
	Function* df_dx = f->derivative(x);
	double expected_df_dx = 1.0 / x_val;
	double computed_df_dx = df_dx->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));

	delete f;
	delete df_dx;
}

TEST(FunctionTest, TestExponentialOfSquareFunction) {
	double x_val = 1.0;
	Variable* x = new Variable(&x_val);
	Function* x_squared = new Power(x, new Constant(2.0));
	Function* f = new Exp(x_squared);
	double expected_f = std::exp(1.0);
	double computed_f = f->evaluate();
	EXPECT_TRUE(almost_equal(computed_f, expected_f));
	Function* df_dx = f->derivative(x);
	double expected_df_dx = 2.0 * x_val * expected_f;
	double computed_df_dx = df_dx->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));

	delete f;
	delete df_dx;
}


TEST(FunctionTest, TestComplexFunction) {
	double x_val = 1.0;
	double y_val = 2.0;
	Variable* x = new Variable(&x_val);
	Variable* y = new Variable(&y_val);
	Function* x_squared = new Power(x, new Constant(2.0));
	Function* y_squared = new Power(y, new Constant(2.0));
	Function* sum = new Addition(x_squared, y_squared);
	Function* xy = new Multiplication(x->clone(), y->clone());
	Function* exponential = new Exp(xy);
	Function* f = new Multiplication(sum, exponential);
	double expected_f = (std::pow(x_val, 2) + std::pow(y_val, 2)) * std::exp(x_val * y_val);
	double computed_f = f->evaluate();
	EXPECT_TRUE(almost_equal(computed_f, expected_f));
	Function* df_dx = f->derivative(x);

	double expected_df_dx = (2.0 * x_val) * std::exp(x_val * y_val) +
							(std::pow(x_val, 2) + std::pow(y_val, 2)) * y_val * std::exp(x_val * y_val);
	double computed_df_dx = df_dx->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dx, expected_df_dx));

	Function* df_dy = f->derivative(y);
	double expected_df_dy = (2.0 * y_val) * std::exp(x_val * y_val) +
							(std::pow(x_val, 2) + std::pow(y_val, 2)) * x_val * std::exp(x_val * y_val);
	double computed_df_dy = df_dy->evaluate();
	EXPECT_TRUE(almost_equal(computed_df_dy, expected_df_dy));

	delete f;
	delete df_dx;
	delete df_dy;
}

TEST(FunctionTest, TestDivisionByZero) {
	double x_val = 1.0;
	double y_val = 1.0;
	Variable* x = new Variable(&x_val);
	Variable* y = new Variable(&y_val);

	Function* numerator = new Constant(1.0);
	Function* denominator = new Subtraction(x, y);
	Function* f = new Division(numerator, denominator);

	EXPECT_THROW({
					 f->evaluate();
				 }, std::runtime_error);

	delete f;
}

TEST(FunctionTest, TestLnOfNonPositive) {
	double x_val = -1.0;
	Variable* x = new Variable(&x_val);

	Function* f = new Ln(x);

	EXPECT_THROW({
        f->evaluate();
        }, std::runtime_error);

	delete f;
}

TEST(FunctionTest, TestLog) {
	// Example 1: Log base 2 of 8 = 3
	double base_val1 = 2.0;
	double arg_val1 = 8.0;
	Variable* base1 = new Variable(&base_val1);
	Variable* arg1 = new Variable(&arg_val1);
	Function* logFunc1 = new Log(base1, arg1);
	double res1 = logFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, 3.0); // Check value Log2(8)

	// Derivative: d/dx Log2(x) = 1 / (x * ln(2))
	Function* logDerivative1 = logFunc1->derivative(arg1);
	double expected_derivative1 = 1.0 / (arg_val1 * std::log(base_val1));
	EXPECT_NEAR(logDerivative1->evaluate(), expected_derivative1, 1e-9); // Check derivative at x=8

	// Example 2: Log base 10 of 1000 = 3
	double base_val2 = 10.0;
	double arg_val2 = 1000.0;
	Variable* base2 = new Variable(&base_val2);
	Variable* arg2 = new Variable(&arg_val2);
	Function* logFunc2 = new Log(base2, arg2);
	double res2 = logFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, 3.0); // Check value Log10(1000)

	// Derivative: d/dx Log10(x) = 1 / (x * ln(10))
	Function* logDerivative2 = logFunc2->derivative(arg2);
	double expected_derivative2 = 1.0 / (arg_val2 * std::log(base_val2));
	EXPECT_NEAR(logDerivative2->evaluate(), expected_derivative2, 1e-9); // Check derivative at x=1000

	// Example 3: Log base e of e = 1
	double base_val3 = std::exp(1.0);
	double arg_val3 = std::exp(1.0);
	Variable* base3 = new Variable(&base_val3);
	Variable* arg3 = new Variable(&arg_val3);
	Function* logFunc3 = new Log(base3, arg3);
	double res3 = logFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, 1.0); // Check value Log_e(e)

	// Derivative: d/dx Log_e(x) = 1 / x
	Function* logDerivative3 = logFunc3->derivative(arg3);
	double expected_derivative3 = 1.0 / arg_val3;
	EXPECT_DOUBLE_EQ(logDerivative3->evaluate(), expected_derivative3); // Check derivative at x=e

	// Example 4: Log base 3 of 1 = 0
	double base_val4 = 3.0;
	double arg_val4 = 1.0;
	Variable* base4 = new Variable(&base_val4);
	Variable* arg4 = new Variable(&arg_val4);
	Function* logFunc4 = new Log(base4, arg4);
	double res4 = logFunc4->evaluate();
	EXPECT_DOUBLE_EQ(res4, 0.0); // Check value Log3(1)

	// Derivative: d/dx Log3(x) = 1 / (ln(3) * x)
	Function* logDerivative4 = logFunc4->derivative(arg4);
	double expected_derivative4 = 1.0 / std::log(base_val4);
	EXPECT_DOUBLE_EQ(logDerivative4->evaluate(), expected_derivative4); // Check derivative at x=1

	// Example 5: Log base 5 of 25 = 2
	double base_val5 = 5.0;
	double arg_val5 = 25.0;
	Variable* base5 = new Variable(&base_val5);
	Variable* arg5 = new Variable(&arg_val5);
	Function* logFunc5 = new Log(base5, arg5);
	double res5 = logFunc5->evaluate();
	EXPECT_DOUBLE_EQ(res5, 2.0); // Check value Log5(25)

	// Derivative: d/dx Log5(x) = 1 / (x * ln(5))
	Function* logDerivative5 = logFunc5->derivative(arg5);
	double expected_derivative5 = 1.0 / (arg_val5 * std::log(base_val5));
	EXPECT_NEAR(logDerivative5->evaluate(), expected_derivative5, 1e-9); // Check derivative at x=25

	delete logFunc1;
	delete logFunc2;
	delete logFunc3;
	delete logFunc4;
	delete logFunc5;

	delete logDerivative1;
	delete logDerivative2;
	delete logDerivative3;
	delete logDerivative4;
	delete logDerivative5;
}

TEST(FunctionTest, TestSqrt) {
	double x_val = 4.0;
	Variable* x = new Variable(&x_val);
	Function* f = new Sqrt(x);
	double res = f->evaluate();
	EXPECT_EQ(res, std::sqrt(x_val)); // Sqrt(x)

	// 2 * Sqrt(5y)
	double y_val = 2.0;
	Variable* y = new Variable(&y_val);
	Function* mu = new Multiplication(new Constant(5.0), y);
	Function* sqrtFunc = new Sqrt(mu);
	Function* mu2 = new Multiplication(new Constant(2.0), sqrtFunc);
	EXPECT_EQ(mu2->evaluate(), 2 * std::sqrt(5 * y_val)); // 2 * Sqrt(5y)

	// d/dy (2 * Sqrt(5y)) = (2 * (5)) / (2 * Sqrt(5y)) = 5 / Sqrt(5y)
	Function* mu2D = mu2->derivative(y);
	double expected_derivative = 5.0 / std::sqrt(5 * y_val);
	EXPECT_EQ(mu2D->evaluate(), expected_derivative);

	delete f;
	delete mu2;
	delete mu2D;
}

TEST(FunctionTest, TestSin) {
	double x_val = 1;
	Variable* x = new Variable(&x_val);
	Function* f = new Sin(x);
	double res = f->evaluate();
	EXPECT_EQ(res, std::sin(x_val));

	// Creating func:  2 * sin(5y)
	double y_val = 7;
	Variable* y = new Variable(&y_val);
	Function* mu = new Multiplication(new Constant(5.0), y);
	Function* sinFunc = new Sin(mu);
	Function* mu2 = new Multiplication(new Constant(2.0), sinFunc);
	EXPECT_EQ(mu2->evaluate(), 2 * std::sin(5 * y_val));
	Function* mu2D = mu2->derivative(y);
	EXPECT_EQ(mu2D->evaluate(), 10 * std::cos(5 * y_val));

	delete f;
	delete mu2;
	delete mu2D;
}

TEST(FunctionTest, TestCos) {
	double x_val = 1;
	Variable* x = new Variable(&x_val);
	Function* f = new Cos(x);
	double res = f->evaluate();
	EXPECT_EQ(res, std::cos(x_val));

	// Creating func:  2 * cos(5y)
	double y_val = 7;
	Variable* y = new Variable(&y_val);
	Function* mu = new Multiplication(new Constant(5.0), y);
	Function* cosFunc = new Cos(mu);
	Function* mu2 = new Multiplication(new Constant(2.0), cosFunc);
	EXPECT_EQ(mu2->evaluate(), 2 * std::cos(5 * y_val));
	Function* mu2D = mu2->derivative(y);
	EXPECT_EQ(mu2D->evaluate(), -10 * std::sin(5 * y_val));

	delete f;
	delete mu2;
	delete mu2D;
}

TEST(FunctionTest, TestAsin) {
	double x_val = 0.5;
	Variable* x = new Variable(&x_val);
	Function* f = new Asin(x);
	double res = f->evaluate();
	EXPECT_EQ(res, std::asin(x_val)); // Asin(x)

	// 2 * Asin(5y)
	double y_val = 0.1;
	Variable* y = new Variable(&y_val);
	Function* mu = new Multiplication(new Constant(5.0), y);
	Function* asinFunc = new Asin(mu);
	Function* mu2 = new Multiplication(new Constant(2.0), asinFunc);
	EXPECT_EQ(mu2->evaluate(), 2 * std::asin(5 * y_val)); // 2 * Asin(5y)

	// d/dy (2 * Asin(5y)) = 10 / sqrt(1 - (5y)^2)
	Function* mu2D = mu2->derivative(y);
	double expected_derivative = 10.0 / std::sqrt(1.0 - std::pow(5 * y_val, 2));
	EXPECT_EQ(mu2D->evaluate(), expected_derivative);

	delete f;
	delete mu2;
	delete mu2D;
}

TEST(FunctionTest, TestAcos) {
	double x_val = 0.5;
	Variable* x = new Variable(&x_val);
	Function* f = new Acos(x);
	double res = f->evaluate();
	EXPECT_EQ(res, std::acos(x_val)); // Acos(x)

	// 2 * Acos(5y)
	double y_val = 0.1;
	Variable* y = new Variable(&y_val);
	Function* mu = new Multiplication(new Constant(5.0), y);
	Function* acosFunc = new Acos(mu);
	Function* mu2 = new Multiplication(new Constant(2.0), acosFunc);
	EXPECT_EQ(mu2->evaluate(), 2 * std::acos(5 * y_val)); // 2 * Acos(5y)

	// d/dy (2 * Acos(5y)) = -10 / sqrt(1 - (5y)^2)
	Function* mu2D = mu2->derivative(y);
	double expected_derivative = -10.0 / std::sqrt(1.0 - std::pow(5 * y_val, 2));
	EXPECT_EQ(mu2D->evaluate(), expected_derivative);

	delete f;
	delete mu2;
	delete mu2D;
}

// -------------------- Negation Test Implementation --------------------
TEST(FunctionTest, TestNegation) {
	double x_val = 3.0;
	Variable* x = new Variable(&x_val);
	Function* negFunc(new Negation(x));
	double res = negFunc->evaluate();
	EXPECT_EQ(res, -x_val); // Check value Negation(x)

	// Check derivative: 2 * Negation(5y)
	double y_val = 2.0;
	Variable* y = new Variable(&y_val);
	Function* mu(new Multiplication(new Constant(5.0), y));
	Function* negMu(new Negation(mu));
	Function* mu2(new Multiplication(new Constant(2.0), negMu));
	EXPECT_EQ(mu2->evaluate(), 2 * (-5.0 * y_val)); // check value 2 * Negation(5y)

	// Check derivative: d/dy (2 * Negation(5y)) = -10
	Function* mu2D(mu2->derivative(y));
	double expected_derivative = -10.0;
	EXPECT_EQ(mu2D->evaluate(), expected_derivative); // check derivative value


	delete negFunc;
	delete mu2;
	delete mu2D;
}

TEST(FunctionTest, TestMod) {
	// Example 1: Mod(10, 3) = 1
	double num_val1 = 10.0;
	double den_val1 = 3.0;
	Variable* num1 = new Variable(&num_val1);
	Variable* den1 = new Variable(&den_val1);
	Function* modFunc1 = new Mod(num1, den1);
	double res1 = modFunc1->evaluate();
	EXPECT_EQ(res1, std::fmod(num_val1, den_val1)); // should be 1.0


	// Derivative Mod(x, y) = 0
	Function* modDerivative1 = modFunc1->derivative(num1);
	double expected_derivative1 = 0.0;
	EXPECT_EQ(modDerivative1->evaluate(), expected_derivative1);

	// Example 2: Mod(7.5, 2.5) = 0
	double num_val2 = 7.5;
	double den_val2 = 2.5;
	Variable* num2 = new Variable(&num_val2);
	Variable* den2 = new Variable(&den_val2);
	Function* modFunc2 = new Mod(num2, den2);
	double res2 = modFunc2->evaluate();
	EXPECT_EQ(res2, std::fmod(num_val2, den_val2)); // Should be 0.0

	// Derivative Mod(x, y) = 0
	Function* modDerivative2 = modFunc2->derivative(num2);
	double expected_derivative2 = 0.0;
	EXPECT_EQ(modDerivative2->evaluate(), expected_derivative2);

	// Example 3: Mod(-5, 3) = 1
	double num_val3 = -5.0;
	double den_val3 = 3.0;
	Variable* num3 = new Variable(&num_val3);
	Variable* den3 = new Variable(&den_val3);
	Function* modFunc3 = new Mod(num3, den3);
	double res3 = modFunc3->evaluate();
	EXPECT_EQ(res3, std::fmod(num_val3, den_val3)); // Should be -2.0 in C++ fmod.

	// Derivative Mod(x, y) = 0
	Function* modDerivative3 = modFunc3->derivative(num3);
	double expected_derivative3 = 0.0;
	EXPECT_EQ(modDerivative3->evaluate(), expected_derivative3);

	delete modFunc1;
	delete modFunc2;
	delete modFunc3;
	delete modDerivative1;
	delete modDerivative2;
	delete modDerivative3;
}

TEST(FunctionTest, TestTan) {
	// Example 1: Tan(0) = 0
	double x_val1 = 0.0;
	Variable* x1 = new Variable(&x_val1);
	Function* tanFunc1 = new Tan(x1);
	double res1 = tanFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, std::tan(x_val1)); // Check value Tan(0)

	// Derivative: sec^2(0) = 1
	Function* tanDerivative1 = tanFunc1->derivative(x1);
	double expected_derivative1 = 1.0 / (std::cos(x_val1) * std::cos(x_val1));
	EXPECT_DOUBLE_EQ(tanDerivative1->evaluate(), expected_derivative1); // Check derivative at 0

	// Example 2: Tan(pi/4) = 1
	double x_val2 = M_PI / 4;
	Variable* x2 = new Variable(&x_val2);
	Function* tanFunc2 = new Tan(x2);
	double res2 = tanFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, std::tan(x_val2)); // Check value Tan(pi/4)

	// Derivative: sec^2(pi/4) = 2
	Function* tanDerivative2 = tanFunc2->derivative(x2);
	double expected_derivative2 = 1.0 / (std::cos(x_val2) * std::cos(x_val2));
	EXPECT_DOUBLE_EQ(tanDerivative2->evaluate(), expected_derivative2); // Check derivative at pi/4

	// Example 3: Tan(pi/3) = sqrt(3)
	double x_val3 = M_PI / 3;
	Variable* x3 = new Variable(&x_val3);
	Function* tanFunc3 = new Tan(x3);
	double res3 = tanFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, std::tan(x_val3)); // Check value Tan(pi/3)

	// Derivative: sec^2(pi/3) = 4
	Function* tanDerivative3 = tanFunc3->derivative(x3);
	double expected_derivative3 = 1.0 / (std::cos(x_val3) * std::cos(x_val3));
	EXPECT_DOUBLE_EQ(tanDerivative3->evaluate(), expected_derivative3); // Check derivative at pi/3

	delete tanFunc1;
	delete tanFunc2;
	delete tanFunc3;
	delete tanDerivative1;
	delete tanDerivative2;
	delete tanDerivative3;
}

TEST(FunctionTest, TestAtan) {
	// Example 1: Atan(0) = 0
	double x_val1 = 0.0;
	Variable* x1 = new Variable(&x_val1);
	Function* atanFunc1 = new Atan(x1);
	double res1 = atanFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, std::atan(x_val1)); // Check value Atan(0)

	// Derivative: 1 / (1 + 0^2) = 1
	Function* atanDerivative1 = atanFunc1->derivative(x1);
	double expected_derivative1 = 1.0 / (1.0 + x_val1 * x_val1);
	EXPECT_DOUBLE_EQ(atanDerivative1->evaluate(), expected_derivative1); // Check derivative at 0

	// Example 2: Atan(1) = pi/4
	double x_val2 = 1.0;
	Variable* x2 = new Variable(&x_val2);
	Function* atanFunc2 = new Atan(x2);
	double res2 = atanFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, std::atan(x_val2)); // Check value Atan(1)

	// Derivative: 1 / (1 + 1^2) = 0.5
	Function* atanDerivative2 = atanFunc2->derivative(x2);
	double expected_derivative2 = 1.0 / (1.0 + x_val2 * x_val2);
	EXPECT_DOUBLE_EQ(atanDerivative2->evaluate(), expected_derivative2); // Check derivative at 1

	// Example 3: Atan(sqrt(3)) = pi/3
	double x_val3 = std::sqrt(3.0);
	Variable* x3 = new Variable(&x_val3);
	Function* atanFunc3 = new Atan(x3);
	double res3 = atanFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, std::atan(x_val3)); // Check value Atan(sqrt(3))

	// Derivative: 1 / (1 + (sqrt(3))^2) = 0.25
	Function* atanDerivative3 = atanFunc3->derivative(x3);
	double expected_derivative3 = 1.0 / (1.0 + x_val3 * x_val3);
	EXPECT_DOUBLE_EQ(atanDerivative3->evaluate(), expected_derivative3); // Check derivative at sqrt(3))

	delete atanFunc1;
	delete atanFunc2;
	delete atanFunc3;
	delete atanDerivative1;
	delete atanDerivative2;
	delete atanDerivative3;
}

TEST(FunctionTest, TestCot) {
	// Example 1: Cot(pi/4) = 1
	double x_val1 = M_PI / 4;
	Variable* x1 = new Variable(&x_val1);
	Function* cotFunc1 = new Cot(x1);
	double res1 = cotFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, 1.0); // Check value Cot(pi/4)

	// Derivative: -1 / sin^2(pi/4) = -2
	Function* cotDerivative1 = cotFunc1->derivative(x1);
	double expected_derivative1 = -1.0 / (std::sin(x_val1) * std::sin(x_val1));
	EXPECT_DOUBLE_EQ(cotDerivative1->evaluate(), expected_derivative1); // Check derivative at pi/4

	// Example 2: Cot(pi/3) = 1/sqrt(3)
	double x_val2 = M_PI / 3;
	Variable* x2 = new Variable(&x_val2);
	Function* cotFunc2 = new Cot(x2);
	double res2 = cotFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, 1.0 / std::sqrt(3.0)); // Check value Cot(pi/3)

	// Derivative: -1 / sin^2(pi/3) = -4/3
	Function* cotDerivative2 = cotFunc2->derivative(x2);
	double expected_derivative2 = -1.0 / (std::sin(x_val2) * std::sin(x_val2));
	EXPECT_DOUBLE_EQ(cotDerivative2->evaluate(), expected_derivative2); // Check derivative at pi/3

	// Example 3: Cot(pi/6) = sqrt(3)
	double x_val3 = M_PI / 6;
	Variable* x3 = new Variable(&x_val3);
	Function* cotFunc3 = new Cot(x3);
	double res3 = cotFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, std::sqrt(3.0)); // Check value Cot(pi/6)

	// Derivative: -1 / sin^2(pi/6) = -4
	Function* cotDerivative3 = cotFunc3->derivative(x3);
	double expected_derivative3 = -1.0 / (std::sin(x_val3) * std::sin(x_val3));
	EXPECT_DOUBLE_EQ(cotDerivative3->evaluate(), expected_derivative3); // Check derivative at pi/6

	// Clean up
	delete cotFunc1;
	delete cotFunc2;
	delete cotFunc3;
	delete cotDerivative1;
	delete cotDerivative2;
	delete cotDerivative3;
}

TEST(FunctionTest, TestAcot) {
	// Example 1: Acot(1) = pi/4
	double x_val1 = 1.0;
	Variable* x1 = new Variable(&x_val1);
	Function* acotFunc1 = new Acot(x1);
	double res1 = acotFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, M_PI / 4.0); // Check value Acot(1)

	// Derivative: -1 / (1 + 1^2) = -0.5
	Function* acotDerivative1 = acotFunc1->derivative(x1);
	double expected_derivative1 = -1.0 / (1.0 + x_val1 * x_val1);
	EXPECT_DOUBLE_EQ(acotDerivative1->evaluate(), expected_derivative1); // Check derivative at 1

	// Example 2: Acot(sqrt(3)) = pi/6
	double x_val2 = std::sqrt(3.0);
	Variable* x2 = new Variable(&x_val2);
	Function* acotFunc2 = new Acot(x2);
	double res2 = acotFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, M_PI / 6.0); // Check value Acot(sqrt(3))

	// Derivative: -1 / (1 + (sqrt(3))^2) = -0.25
	Function* acotDerivative2 = acotFunc2->derivative(x2);
	double expected_derivative2 = -1.0 / (1.0 + x_val2 * x_val2);
	EXPECT_DOUBLE_EQ(acotDerivative2->evaluate(), expected_derivative2); // Check derivative at sqrt(3)

	// Example 3: Acot(0) = pi/2
	double x_val3 = 0.0;
	Variable* x3 = new Variable(&x_val3);
	Function* acotFunc3 = new Acot(x3);
	double res3 = acotFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, M_PI / 2.0); // Check value Acot(0)

	// Derivative: -1 / (1 + 0^2) = -1
	Function* acotDerivative3 = acotFunc3->derivative(x3);
	double expected_derivative3 = -1.0 / (1.0 + x_val3 * x_val3);
	EXPECT_DOUBLE_EQ(acotDerivative3->evaluate(), expected_derivative3); // Check derivative at 0

	// Clean up
	delete acotFunc1;
	delete acotFunc2;
	delete acotFunc3;
	delete acotDerivative1;
	delete acotDerivative2;
	delete acotDerivative3;
}

// -------------------- Abs Test Implementation --------------------
TEST(FunctionTest, TestAbs) {
	// Example 1: Abs(x) = 5
	double x_val1 = 5.0;
	Variable* x1 = new Variable(&x_val1);
	Function* absFunc1 = new Abs(x1);
	double res1 = absFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, std::abs(x_val1)); // Check value Abs(5)

	// Derivative: d/dx Abs(x) = 1
	Function* absDerivative1 = absFunc1->derivative(x1);
	double expected_derivative1 = 1.0;
	EXPECT_DOUBLE_EQ(absDerivative1->evaluate(), expected_derivative1); // Check derivative at 5

	// Example 2: Abs(-3) = 3
	double x_val2 = -3.0;
	Variable* x2 = new Variable(&x_val2);
	Function* absFunc2 = new Abs(x2);
	double res2 = absFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, std::abs(x_val2)); // Check value Abs(-3)

	// Derivative: d/dx Abs(x) = sign(x) => sign(-3) = -1
	Function* absDerivative2 = absFunc2->derivative(x2);
	double expected_derivative2 = -1.0;
	EXPECT_DOUBLE_EQ(absDerivative2->evaluate(), expected_derivative2); // Check derivative at -3

	// Example 3: Abs(0) = 0
	double x_val3 = 0.0;
	Variable* x3 = new Variable(&x_val3);
	Function* absFunc3 = new Abs(x3);
	double res3 = absFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, std::abs(x_val3)); // Check value Abs(0)

	// Derivative: d/dx Abs(x) = sign(x) => sign(0) = 0 (by convention)
	Function* absDerivative3 = absFunc3->derivative(x3);
	double expected_derivative3 = 0.0;
	EXPECT_DOUBLE_EQ(absDerivative3->evaluate(), expected_derivative3); // Check derivative at 0


	delete absFunc1;
	delete absFunc2;
	delete absFunc3;
	delete absDerivative1;
	delete absDerivative2;
	delete absDerivative3;
}

TEST(FunctionTest, TestSign) {
	// Example 1: Sign(5) = 1
	double x_val1 = 5.0;
	Variable* x1 = new Variable(&x_val1);
	Function* signFunc1 = new Sign(x1);
	double res1 = signFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, 1.0); // Check value Sign(5)

	// Derivative: d/dx Sign(5) = 0
	Function* signDerivative1 = signFunc1->derivative(x1);
	double expected_derivative1 = 0.0;
	EXPECT_DOUBLE_EQ(signDerivative1->evaluate(), expected_derivative1); // Check derivative at 5

	// Example 2: Sign(-3) = -1
	double x_val2 = -3.0;
	Variable* x2 = new Variable(&x_val2);
	Function* signFunc2 = new Sign(x2);
	double res2 = signFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, -1.0); // Check value Sign(-3)

	// Derivative: d/dx Sign(-3) = 0
	Function* signDerivative2 = signFunc2->derivative(x2);
	double expected_derivative2 = 0.0;
	EXPECT_DOUBLE_EQ(signDerivative2->evaluate(), expected_derivative2); // Check derivative at -3

	// Example 3: Sign(0) = 0
	double x_val3 = 0.0;
	Variable* x3 = new Variable(&x_val3);
	Function* signFunc3 = new Sign(x3);
	double res3 = signFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, 0.0); // Check value Sign(0)

	// Derivative: d/dx Sign(0) = 0 (by convention)
	Function* signDerivative3 = signFunc3->derivative(x3);
	double expected_derivative3 = 0.0;
	EXPECT_DOUBLE_EQ(signDerivative3->evaluate(), expected_derivative3); // Check derivative at 0

	// Example 4: Sign(2.718) ≈ 1
	double x_val4 = 2.718;
	Variable* x4 = new Variable(&x_val4);
	Function* signFunc4 = new Sign(x4);
	double res4 = signFunc4->evaluate();
	EXPECT_DOUBLE_EQ(res4, 1.0); // Check value Sign(2.718)

	// Derivative: d/dx Sign(2.718) = 0
	Function* signDerivative4 = signFunc4->derivative(x4);
	double expected_derivative4 = 0.0;
	EXPECT_DOUBLE_EQ(signDerivative4->evaluate(), expected_derivative4); // Check derivative at 2.718

	// Example 5: Sign(-0.001) = -1
	double x_val5 = -0.001;
	Variable* x5 = new Variable(&x_val5);
	Function* signFunc5 = new Sign(x5);
	double res5 = signFunc5->evaluate();
	EXPECT_DOUBLE_EQ(res5, -1.0); // Check value Sign(-0.001)

	// Derivative: d/dx Sign(-0.001) = 0
	Function* signDerivative5 = signFunc5->derivative(x5);
	double expected_derivative5 = 0.0;
	EXPECT_DOUBLE_EQ(signDerivative5->evaluate(), expected_derivative5); // Check derivative at -0.001

	delete signFunc1;
	delete signFunc2;
	delete signFunc3;
	delete signFunc4;
	delete signFunc5;
	delete signDerivative1;
	delete signDerivative2;
	delete signDerivative3;
	delete signDerivative4;
	delete signDerivative5;
}

TEST(FunctionTest, TestMax) {
	// Example 1: Max(3, 5) = 5
	double a_val1 = 3.0;
	double b_val1 = 5.0;
	Variable* a1 = new Variable(&a_val1);
	Variable* b1 = new Variable(&b_val1);
	Function* maxFunc1 = new Max(a1, b1);
	double res1 = maxFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, 5.0); // Check value Max(3, 5)

	// Derivative: d/dx Max(x, 5) = 1 if x > 5, else 0
	Function* maxDerivative1 = maxFunc1->derivative(a1);
	double expected_derivative1 = 0.0; // Since a_val1 = 3 < 5
	EXPECT_DOUBLE_EQ(maxDerivative1->evaluate(), expected_derivative1); // Check derivative at x=3

	// Example 2: Max(7, 2) = 7
	double a_val2 = 7.0;
	double b_val2 = 2.0;
	Variable* a2 = new Variable(&a_val2);
	Variable* b2 = new Variable(&b_val2);
	Function* maxFunc2 = new Max(a2, b2);
	double res2 = maxFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, 7.0); // Check value Max(7, 2)

	// Derivative: d/dx Max(x, 2) = 1 if x > 2, else 0
	Function* maxDerivative2 = maxFunc2->derivative(a2);
	double expected_derivative2 = 1.0; // Since a_val2 = 7 > 2
	EXPECT_DOUBLE_EQ(maxDerivative2->evaluate(), expected_derivative2); // Check derivative at x=7

	// Example 3: Max(x, x) = x
	double a_val3 = 4.0;
	double b_val3 = 4.0;
	Variable* a3 = new Variable(&a_val3);
	Variable* b3 = new Variable(&b_val3);
	Function* maxFunc3 = new Max(a3, b3);
	double res3 = maxFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, 4.0); // Check value Max(4, 4)

	// Derivative: d/dx Max(x, x) = 0 by convention
	Function* maxDerivative3 = maxFunc3->derivative(a3);
	double expected_derivative3 = 0.0;
	EXPECT_DOUBLE_EQ(maxDerivative3->evaluate(), expected_derivative3); // Check derivative at x=4

	// Example 4: Max(x, 10) where x = 15
	double a_val4 = 15.0;
	double b_val4 = 10.0;
	Variable* a4 = new Variable(&a_val4);
	Variable* b4 = new Variable(&b_val4);
	Function* maxFunc4 = new Max(a4, b4);
	double res4 = maxFunc4->evaluate();
	EXPECT_DOUBLE_EQ(res4, 15.0); // Check value Max(15, 10)

	// Derivative: d/dx Max(x, 10) = 1
	Function* maxDerivative4 = maxFunc4->derivative(a4);
	double expected_derivative4 = 1.0;
	EXPECT_DOUBLE_EQ(maxDerivative4->evaluate(), expected_derivative4); // Check derivative at x=15

	// Example 5: Max(x, -5) where x = -10
	double a_val5 = -10.0;
	double b_val5 = -5.0;
	Variable* a5 = new Variable(&a_val5);
	Variable* b5 = new Variable(&b_val5);
	Function* maxFunc5 = new Max(a5, b5);
	double res5 = maxFunc5->evaluate();
	EXPECT_DOUBLE_EQ(res5, -5.0); // Check value Max(-10, -5)

	// Derivative: d/dx Max(x, -5) = 0
	Function* maxDerivative5 = maxFunc5->derivative(a5);
	double expected_derivative5 = 0.0;
	EXPECT_DOUBLE_EQ(maxDerivative5->evaluate(), expected_derivative5); // Check derivative at x=-10

	delete maxFunc1;
	delete maxFunc2;
	delete maxFunc3;
	delete maxFunc4;
	delete maxFunc5;
	delete maxDerivative1;
	delete maxDerivative2;
	delete maxDerivative3;
	delete maxDerivative4;
	delete maxDerivative5;
}

TEST(FunctionTest, TestMin) {
	// Example 1: Min(3, 5) = 3
	double a_val1 = 3.0;
	double b_val1 = 5.0;
	Variable* a1 = new Variable(&a_val1);
	Variable* b1 = new Variable(&b_val1);
	Function* minFunc1 = new Min(a1, b1);
	double res1 = minFunc1->evaluate();
	EXPECT_DOUBLE_EQ(res1, 3.0); // Check value Min(3, 5)

	// Derivative: d/dx Min(x, 5) = 1 if x < 5, else 0
	Function* minDerivative1 = minFunc1->derivative(a1);
	double expected_derivative1 = 1.0; // Since a_val1 = 3 < 5
	EXPECT_DOUBLE_EQ(minDerivative1->evaluate(), expected_derivative1); // Check derivative at x=3

	// Example 2: Min(7, 2) = 2
	double a_val2 = 7.0;
	double b_val2 = 2.0;
	Variable* a2 = new Variable(&a_val2);
	Variable* b2 = new Variable(&b_val2);
	Function* minFunc2 = new Min(a2, b2);
	double res2 = minFunc2->evaluate();
	EXPECT_DOUBLE_EQ(res2, 2.0); // Check value Min(7, 2)

	// Derivative: d/dx Min(x, 2) = 0 since x > 2
	Function* minDerivative2 = minFunc2->derivative(a2);
	double expected_derivative2 = 0.0; // Since a_val2 = 7 > 2
	EXPECT_DOUBLE_EQ(minDerivative2->evaluate(), expected_derivative2); // Check derivative at x=7

	// Example 3: Min(x, x) = x
	double a_val3 = 4.0;
	double b_val3 = 4.0;
	Variable* a3 = new Variable(&a_val3);
	Variable* b3 = new Variable(&b_val3);
	Function* minFunc3 = new Min(a3, b3);
	double res3 = minFunc3->evaluate();
	EXPECT_DOUBLE_EQ(res3, 4.0); // Check value Min(4, 4)

	// Derivative: d/dx Min(x, x) = 0 by convention
	Function* minDerivative3 = minFunc3->derivative(a3);
	double expected_derivative3 = 0.0;
	EXPECT_DOUBLE_EQ(minDerivative3->evaluate(), expected_derivative3); // Check derivative at x=4

	// Example 4: Min(x, 10) where x = 15
	double a_val4 = 15.0;
	double b_val4 = 10.0;
	Variable* a4 = new Variable(&a_val4);
	Variable* b4 = new Variable(&b_val4);
	Function* minFunc4 = new Min(a4, b4);
	double res4 = minFunc4->evaluate();
	EXPECT_DOUBLE_EQ(res4, 10.0); // Check value Min(15, 10)

	// Derivative: d/dx Min(x, 10) = 0
	Function* minDerivative4 = minFunc4->derivative(a4);
	double expected_derivative4 = 0.0;
	EXPECT_DOUBLE_EQ(minDerivative4->evaluate(), expected_derivative4); // Check derivative at x=15

	// Example 5: Min(x, -5) where x = -10
	double a_val5 = -10.0;
	double b_val5 = -5.0;
	Variable* a5 = new Variable(&a_val5);
	Variable* b5 = new Variable(&b_val5);
	Function* minFunc5 = new Min(a5, b5);
	double res5 = minFunc5->evaluate();
	EXPECT_DOUBLE_EQ(res5, -10.0); // Check value Min(-10, -5)

	// Derivative: d/dx Min(x, -5) = 1
	Function* minDerivative5 = minFunc5->derivative(a5);
	double expected_derivative5 = 1.0;
	EXPECT_DOUBLE_EQ(minDerivative5->evaluate(), expected_derivative5); // Check derivative at x=-10

	delete minFunc1;
	delete minFunc2;
	delete minFunc3;
	delete minFunc4;
	delete minFunc5;
	delete minDerivative1;
	delete minDerivative2;
	delete minDerivative3;
	delete minDerivative4;
	delete minDerivative5;
}
TEST(FunctionTest, TestSimplifyConstantExpressions) {
    // Test 1: 2 + 3 simplifies to 5
    Function* f1 = new Addition(new Constant(2.0), new Constant(3.0));
    Function* simplified_f1 = f1->simplify();
    EXPECT_TRUE(dynamic_cast<Constant*>(simplified_f1));
    EXPECT_TRUE(almost_equal(simplified_f1->evaluate(), 5.0));
    EXPECT_EQ(simplified_f1->to_string(), "5.000000");

    // Test 2: 4 * 2 simplifies to 8
    Function* f2 = new Multiplication(new Constant(4.0), new Constant(2.0));
    Function* simplified_f2 = f2->simplify();
    EXPECT_TRUE(dynamic_cast<Constant*>(simplified_f2));
    EXPECT_TRUE(almost_equal(simplified_f2->evaluate(), 8.0));
    EXPECT_EQ(simplified_f2->to_string(), "8.000000");

    // Test 3: 0 * x simplifies to 0
    double x_val = 5.0;
    Variable* x = new Variable(&x_val);
    Function* f3 = new Multiplication(new Constant(0.0), x);
    Function* simplified_f3 = f3->simplify();
    EXPECT_TRUE(dynamic_cast<Constant*>(simplified_f3));
    EXPECT_TRUE(almost_equal(simplified_f3->evaluate(), 0.0));
    EXPECT_EQ(simplified_f3->to_string(), "0.000000");

    // Test 4: x / x simplifies to 1
    Function* f4 = new Division(x->clone(), x->clone());
    Function* simplified_f4 = f4->simplify();
    EXPECT_TRUE(almost_equal(simplified_f4->evaluate(), 1.0));
    EXPECT_EQ(simplified_f4->to_string(), "(VAR(" + std::to_string(x_val) + ") / VAR(" + std::to_string(x_val) + "))");

    delete f1;
    delete simplified_f1;
    delete f2;
    delete simplified_f2;
    delete f3;
    delete simplified_f3;
    delete f4;
    delete simplified_f4;
}

TEST(FunctionTest, TestSimplifyComplexExpressions) {
    double x_val = 2.0;
    Variable* x = new Variable(&x_val);

    // Test 1: x + 0 simplifies to x
    Function* f1 = new Addition(x, new Constant(0.0));
    Function* simplified_f1 = f1->simplify();
    EXPECT_TRUE(dynamic_cast<Variable*>(simplified_f1));
    EXPECT_TRUE(almost_equal(simplified_f1->evaluate(), x_val));
    EXPECT_EQ(simplified_f1->to_string(), "VAR(" + std::to_string(x_val) + ")");

    // Test 2: 1 * (x^2) simplifies to x^2
    Function* x_squared = new Power(x->clone(), new Constant(2.0));
    Function* f2 = new Multiplication(new Constant(1.0), x_squared);
    Function* simplified_f2 = f2->simplify();
    EXPECT_TRUE(dynamic_cast<Power*>(simplified_f2));
    EXPECT_TRUE(almost_equal(simplified_f2->evaluate(), 4.0));
    EXPECT_EQ(simplified_f2->to_string(), "(VAR(" + std::to_string(x_val) + ") ^ 2.000000)");

    // Test 3: sin(0) simplifies to 0
    Function* f3 = new Sin(new Constant(0.0));
    Function* simplified_f3 = f3->simplify();
    EXPECT_FALSE(dynamic_cast<Constant*>(simplified_f3));
    EXPECT_TRUE(almost_equal(simplified_f3->evaluate(), 0.0));
    EXPECT_EQ(simplified_f3->to_string(), "sin(0.000000)");

    delete f1;
    delete simplified_f1;
    delete f2;
    delete simplified_f2;
    delete f3;
    delete simplified_f3;
}

TEST(FunctionTest, TestToStringBasicFunctions) {
    double x_val = 3.0;
    Variable* x = new Variable(&x_val);

    // Test 1: Constant
    Function* f1 = new Constant(5.0);
    EXPECT_EQ(f1->to_string(), "5.000000");

    // Test 2: Variable
    EXPECT_EQ(x->to_string(), "VAR(" + std::to_string(x_val) + ")");

    // Test 3: Power
    Function* f3 = new Power(x, new Constant(2.0));
    EXPECT_EQ(f3->to_string(), "(VAR(" + std::to_string(x_val) + ") ^ 2.000000)");

    // Test 4: Addition
    Function* f4 = new Addition(x->clone(), new Constant(1.0));
    EXPECT_EQ(f4->to_string(), "(VAR(" + std::to_string(x_val) + ") + 1.000000)");

    // Test 5: Multiplication
    Function* f5 = new Multiplication(new Constant(2.0), x->clone());
    EXPECT_EQ(f5->to_string(), "(2.000000 * VAR(" + std::to_string(x_val) + "))");

    delete f1;
    delete f3;
    delete f4;
    delete f5;
}

TEST(FunctionTest, TestToStringUnaryFunctions) {
    double x_val = 1.0;
    Variable* x = new Variable(&x_val);

    // Test 1: Sin
    Function* f1 = new Sin(x);
    EXPECT_EQ(f1->to_string(), "sin(VAR(" + std::to_string(x_val) + "))");

    // Test 2: Exp
    Function* f2 = new Exp(x->clone());
    EXPECT_EQ(f2->to_string(), "exp(VAR(" + std::to_string(x_val) + "))");

    // Test 3: Ln
    Function* f3 = new Ln(x->clone());
    EXPECT_EQ(f3->to_string(), "ln(VAR(" + std::to_string(x_val) + "))");

    // Test 4: Abs
    Function* f4 = new Abs(x->clone());
    EXPECT_EQ(f4->to_string(), "abs(VAR(" + std::to_string(x_val) + "))");

    // Test 5: Negation
    Function* f5 = new Negation(x->clone());
    EXPECT_EQ(f5->to_string(), "(-VAR(" + std::to_string(x_val) + "))");

    delete f1;
    delete f2;
    delete f3;
    delete f4;
    delete f5;
}

TEST(FunctionTest, TestToStringComplexFunctions) {
    double x_val = 2.0;
    double y_val = 3.0;
    Variable* x = new Variable(&x_val);
    Variable* y = new Variable(&y_val);

    // Test 1: (x^2 + y^2)
    Function* x_squared = new Power(x, new Constant(2.0));
    Function* y_squared = new Power(y, new Constant(2.0));
    Function* f1 = new Addition(x_squared, y_squared);
    EXPECT_EQ(f1->to_string(), "((VAR(" + std::to_string(x_val) + ") ^ 2.000000) + (VAR(" + std::to_string(y_val) + ") ^ 2.000000))");

    // Test 2: exp(x * y)
    Function* xy = new Multiplication(x->clone(), y->clone());
    Function* f2 = new Exp(xy);
    EXPECT_EQ(f2->to_string(), "exp((VAR(" + std::to_string(x_val) + ") * VAR(" + std::to_string(y_val) + ")))");

    // Test 3: (x + y) / (x - y)
    Function* numerator = new Addition(x->clone(), y->clone());
    Function* denominator = new Subtraction(x->clone(), y->clone());
    Function* f3 = new Division(numerator, denominator);
    EXPECT_EQ(f3->to_string(), "((VAR(" + std::to_string(x_val) + ") + VAR(" + std::to_string(y_val) + ")) / (VAR(" + std::to_string(x_val) + ") - VAR(" + std::to_string(y_val) + ")))");

    // Test 4: log_2(x)
    Function* f4 = new Log(new Constant(2.0), x->clone());
    EXPECT_EQ(f4->to_string(), "log(2.000000, VAR(" + std::to_string(x_val) + "))");

    delete f1;
    delete f2;
    delete f3;
    delete f4;
}