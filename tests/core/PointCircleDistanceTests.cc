#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <limits>

TEST(PointCircleDistance, SignedTargetsLiveCoordinatesAndWeights) {
    double px = 3, py = 4, cx = 0, cy = 0, radius = 2;
    const std::vector<double*> vars{&px, &py, &cx, &cy, &radius};
    PointCircleDistanceError f(vars, 3);
    EXPECT_TRUE(f.satisfied(0));
    radius = 8;
    f.setTarget(-3);
    EXPECT_TRUE(f.satisfied(0));
    f.setTarget(-4);
    f.setWeight(2);
    EXPECT_DOUBLE_EQ(f.evaluate(), 1);
    EXPECT_DOUBLE_EQ(f.weightedValue(), 2);
    EXPECT_DOUBLE_EQ(f.weightedGradient().at(&radius), -2);
    std::unique_ptr<PointCircleDistanceError> copy(f.clone());
    radius = 9;
    EXPECT_TRUE(copy->satisfied(0));
    Variable x(&px), y(&py), centerX(&cx), centerY(&cy), r(&radius);
    PointCircleDistanceError wrapped(std::vector<Variable*>{&x, &y, &centerX, &centerY, &r}, -4);
    EXPECT_TRUE(wrapped.satisfied(0));
}

TEST(PointCircleDistance, GradientAndHessianMatchFiniteDifferencesInsideAndOutside) {
    double px = 3, py = 4, cx = 1, cy = -2, radius = 2;
    const std::vector<double*> vars{&px, &py, &cx, &cy, &radius};
    for (double r : {2.0, 8.0}) {
        radius = r;
        PointCircleDistanceError f(vars, -1);
        const auto gradient = f.gradient();
        EXPECT_DOUBLE_EQ(gradient.at(&radius), -1);
        constexpr double h = 1e-5;
        for (auto* variable : vars) {
            const double original = *variable;
            *variable = original + h;
            const double plus = f.evaluate();
            const auto plusGradient = f.gradient();
            *variable = original - h;
            const double minus = f.evaluate();
            const auto minusGradient = f.gradient();
            *variable = original;
            EXPECT_NEAR(gradient.at(variable), (plus - minus) / (2 * h), 1e-8);
            for (auto* other : vars) {
                EXPECT_NEAR(f.secondDerivative(variable, other),
                    (plusGradient.at(other) - minusGradient.at(other)) / (2 * h), 1e-7);
            }
        }
    }
}

TEST(PointCircleDistance, SharedCenterZeroGapAndLargeCoordinates) {
    double x = 2, y = 3, radius = 5;
    PointCircleDistanceError shared({&x, &y, &x, &y, &radius}, -5);
    EXPECT_TRUE(shared.satisfied(0));
    EXPECT_DOUBLE_EQ(shared.gradient().at(&x), 0);
    EXPECT_DOUBLE_EQ(shared.gradient().at(&y), 0);
    EXPECT_DOUBLE_EQ(shared.gradient().at(&radius), -1);
    double px = 7;
    PointCircleDistanceError on({&px, &y, &x, &y, &radius}, 0);
    PointOnCircleError membership({&px, &y, &x, &y, &radius});
    EXPECT_EQ(on.evaluate(), membership.evaluate());
    EXPECT_EQ(on.gradient(), membership.gradient());
    x = 1e308;
    px = 9e307;
    radius = 2e307;
    PointCircleDistanceError large({&px, &y, &x, &y, &radius}, -1e307);
    EXPECT_NEAR(large.evaluate() / 1e307, 0, 1e-14);
}

TEST(PointCircleDistance, RejectsInvalidRadiiAndNonfiniteTargets) {
    double px = 3, py = 4, cx = 0, cy = 0, radius = 5;
    const std::vector<double*> vars{&px, &py, &cx, &cy, &radius};
    PointCircleDistanceError f(vars, -1);
    for (double bad : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        radius = bad;
        EXPECT_FALSE(f.satisfied(100));
        EXPECT_THROW(PointCircleDistanceError(vars, -1), std::invalid_argument);
    }
    radius = 5;
    for (double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(f.setTarget(bad), std::invalid_argument);
        EXPECT_THROW(PointCircleDistanceError(vars, bad), std::invalid_argument);
    }
    f.setWeight(0);
    EXPECT_TRUE(f.satisfied(0));
}
