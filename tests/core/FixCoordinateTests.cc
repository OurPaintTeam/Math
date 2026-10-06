#include "ErrorFunction.h"
#include <gtest/gtest.h>
using VAR = double*;
#define EQ(a,b) EXPECT_NEAR((a),(b),1e-9)
TEST(FixCoordinateFunctionTest, EvaluateAtTarget) {
    double x = 5.0;
    std::vector<VAR> vars = {&x};
    FixCoordinateError f(vars, 5.0);
    EQ(f.evaluate(), 0.0);
}

TEST(FixCoordinateFunctionTest, EvaluateDeviated) {
    double x = 7.0;
    std::vector<VAR> vars = {&x};
    FixCoordinateError f(vars, 5.0);
    EQ(f.evaluate(), 2.0);
}

TEST(FixCoordinateFunctionTest, EvaluateNegativeDeviation) {
    double x = 3.0;
    std::vector<VAR> vars = {&x};
    FixCoordinateError f(vars, 5.0);
    EQ(f.evaluate(), -2.0);
}

TEST(FixCoordinateFunctionTest, GradientIsOne) {
    double x = 5.0;
    std::vector<VAR> vars = {&x};
    FixCoordinateError f(vars, 5.0);

    auto grad = f.gradient();
    EXPECT_EQ(grad.size(), 1u);
    EQ(grad[&x], 1.0);
}

TEST(FixCoordinateFunctionTest, VarCount) {
    double x = 0.0;
    std::vector<VAR> vars = {&x};
    FixCoordinateError f(vars, 0.0);
    EXPECT_EQ(f.variables().size(), 1u);
}

TEST(FixCoordinateFunctionTest, InvalidVarCountThrows) {
    double x = 0.0, y = 0.0;
    std::vector<VAR> vars = {&x, &y};
    EXPECT_THROW(
        FixCoordinateError(vars, 0.0),
        std::invalid_argument
    );
}


TEST(FixCoordinateFunctionTest, TracksVariableChanges) {
    double x = 5.0;
    std::vector<VAR> vars = {&x};
    FixCoordinateError f(vars, 5.0);
    EQ(f.evaluate(), 0.0);

    x = 10.0;
    EQ(f.evaluate(), 5.0);
}

