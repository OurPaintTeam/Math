#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <limits>

TEST(PointArcDistance, SignedValuesMatchCircleWithoutRestrictingAngularSector) {
    double px = 0, py = -8, ax = 3, ay = 4, cx = 0, cy = 0, radius = 5;
    for (double distance : {0.0, 2.0, 5.0, 7.0}) {
        py = -distance;
        for (double target : {-3.0, 0.0, 2.0}) {
            PointArcDistanceError arc({&px, &py, &ax, &ay, &cx, &cy}, target);
            PointCircleDistanceError circle({&px, &py, &cx, &cy, &radius}, target);
            EXPECT_DOUBLE_EQ(arc.evaluate(), circle.evaluate());
            EXPECT_DOUBLE_EQ(arc.gradient().at(&px), circle.gradient().at(&px));
            EXPECT_DOUBLE_EQ(arc.gradient().at(&py), circle.gradient().at(&py));
        }
    }
    PointArcDistanceError zero({&px, &py, &ax, &ay, &cx, &cy}, 0);
    PointOnArcError membership({&px, &py, &ax, &ay, &cx, &cy});
    EXPECT_EQ(zero.evaluate(), membership.evaluate());
    EXPECT_EQ(zero.gradient(), membership.gradient());
}

TEST(PointArcDistance, GradientAndHessianIncludeTheDerivedRadius) {
    double px = 2, py = 7, ax = 4, ay = 1, cx = 1, cy = -2;
    const std::vector<double*> vars{&px, &py, &ax, &ay, &cx, &cy};
    for (double y : {7.0, -1.0}) {
        py = y;
        PointArcDistanceError f(vars, -2);
        const auto gradient = f.gradient();
        for (auto* p : vars) {
            const double old = *p, h = 1e-5;
            *p = old+h; const double plus = f.evaluate(); const auto gp = f.gradient();
            *p = old-h; const double minus = f.evaluate(); const auto gm = f.gradient();
            *p = old;
            EXPECT_NEAR(gradient.at(p), (plus-minus)/(2*h), 1e-8);
            for (auto* q : vars)
                EXPECT_NEAR(f.secondDerivative(p, q), (gp.at(q)-gm.at(q))/(2*h), 1e-7);
        }
    }
}

TEST(PointArcDistance, EndpointAliasesCancelAndCenterAliasesKeepRadiusDerivatives) {
    double ax = 3, ay = 4, cx = 0, cy = 0;
    PointArcDistanceError endpoint({&ax, &ay, &ax, &ay, &cx, &cy}, 0);
    EXPECT_TRUE(endpoint.satisfied(0));
    for (auto* p : {&ax, &ay, &cx, &cy}) {
        EXPECT_DOUBLE_EQ(endpoint.gradient().at(p), 0);
        for (auto* q : {&ax, &ay, &cx, &cy}) EXPECT_DOUBLE_EQ(endpoint.secondDerivative(p, q), 0);
    }
    endpoint.setTarget(2);
    EXPECT_DOUBLE_EQ(endpoint.evaluate(), -2);
    PointArcDistanceError center({&cx, &cy, &ax, &ay, &cx, &cy}, -5);
    EXPECT_TRUE(center.satisfied(0));
    EXPECT_NEAR(center.gradient().at(&ax), -0.6, 1e-12);
    EXPECT_NEAR(center.gradient().at(&cx), 0.6, 1e-12);
}

TEST(PointArcDistance, CoordinatesTargetsWeightsClonesAndVariablesRemainLive) {
    double px = 0, py = -8, ax = 3, ay = 4, cx = 0, cy = 0;
    Variable x(&px), y(&py), a(&ax), b(&ay), c(&cx), d(&cy);
    PointArcDistanceError f(std::vector<Variable*>{&x, &y, &a, &b, &c, &d}, 3);
    EXPECT_TRUE(f.satisfied(0));
    std::unique_ptr<PointArcDistanceError> copy(f.clone());
    ax = 6; ay = 8;
    f.setTarget(-2);
    EXPECT_TRUE(copy->satisfied(0));
    f.setTarget(-3);
    f.setWeight(2);
    EXPECT_DOUBLE_EQ(copy->weightedValue(), 2);
    EXPECT_NEAR(copy->weightedGradient().at(&ax), -1.2, 1e-12);
}

TEST(PointArcDistance, NonfiniteTargetsZeroRadiusAndLargeCoordinates) {
    double px = 9e307, py = 0, ax = 1e308, ay = 0, cx = 8e307, cy = 0;
    const std::vector<double*> vars{&px, &py, &ax, &ay, &cx, &cy};
    PointArcDistanceError f(vars, -1e307);
    EXPECT_NEAR(f.evaluate()/1e307, 0, 1e-14);
    for (double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(PointArcDistanceError(vars, bad), std::invalid_argument);
        EXPECT_THROW(f.setTarget(bad), std::invalid_argument);
    }
    ax = cx; ay = cy;
    EXPECT_FALSE(f.satisfied(1e308));
    f.setWeight(0);
    EXPECT_TRUE(f.satisfied(0));
}
