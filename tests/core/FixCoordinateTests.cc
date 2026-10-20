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

TEST(FixCoordinateFunctionTest, ArcFixUsesSixIndependentCapturedAssignments) {
    double coordinates[] = {0, 0, 10, 0, 5, 5};
    std::vector<std::unique_ptr<FixCoordinateError>> rows;
    for (auto& coordinate : coordinates)
        rows.push_back(std::make_unique<FixCoordinateError>(std::vector<VAR>{&coordinate}, coordinate));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        double* variable = nullptr;
        double target = 0;
        ASSERT_TRUE(rows[i]->assignment(variable, target));
        EXPECT_EQ(variable, &coordinates[i]);
        coordinates[i] += 3;
        EXPECT_DOUBLE_EQ(rows[i]->evaluate(), 3);
        *variable = target;
        EXPECT_TRUE(rows[i]->satisfied(0));
        for (auto& other : coordinates)
            EXPECT_DOUBLE_EQ(rows[i]->gradient().contains(&other) ? rows[i]->gradient().at(&other) : 0,
                             &other == variable ? 1 : 0);
    }
}

