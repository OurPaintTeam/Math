#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <cmath>

TEST(ArcArcMidpoint, DifferentSweepsAndRadiiHaveTheSameMidpoint) {
    const double s = std::sqrt(0.5);
    double ax = 1, ay = 0, bx = -1, by = 0, cx = 0, cy = 0;
    double dx = 2*s+2, dy = 1-2*s, ex = 2*s, ey = -1-2*s, fx = 2*s, fy = 1-2*s;
    const std::vector<double*> vars{&ax, &ay, &bx, &by, &cx, &cy, &dx, &dy, &ex, &ey, &fx, &fy};
    ArcArcMidpointXError x(vars); ArcArcMidpointYError y(vars);
    EXPECT_TRUE(x.satisfied(1e-12));
    EXPECT_TRUE(y.satisfied(1e-12));
    dx += 3; ex += 3; fx += 3;
    EXPECT_NEAR(x.evaluate(), -3, 1e-12);
    EXPECT_TRUE(y.satisfied(1e-12));
    x.setWeight(2);
    EXPECT_NEAR(x.weightedValue(), -6, 1e-12);
    EXPECT_THROW(x.setTarget(1), std::invalid_argument);
}

TEST(ArcArcMidpoint, AllTwelveCoordinateDerivativesAndHessiansMatchFiniteDifferences) {
    double coordinates[] = {4, 5, -2, 6, 1, 1, 6, -2, -2, -3, 0, 0};
    std::vector<double*> vars;
    for (auto& value : coordinates) vars.push_back(&value);
    ArcArcMidpointXError x(vars); ArcArcMidpointYError y(vars);
    for (ErrorFunction* f : {static_cast<ErrorFunction*>(&x), static_cast<ErrorFunction*>(&y)}) {
        for (auto* p : vars) {
            const double old = *p, h = 1e-5;
            const auto gradient = f->gradient();
            *p = old+h; const double plus = f->evaluate(); const auto gp = f->gradient();
            *p = old-h; const double minus = f->evaluate(); const auto gm = f->gradient();
            *p = old;
            EXPECT_NEAR(gradient.at(p), (plus-minus)/(2*h), 1e-8);
            for (auto* q : vars)
                EXPECT_NEAR(f->secondDerivative(p, q), (gp.at(q)-gm.at(q))/(2*h), 1e-7);
        }
    }
}

TEST(ArcArcMidpoint, SharedCoordinatesCancelAndCollapsedArcsFail) {
    double ax = 1, ay = 0, bx = 0, by = 1, cx = 0, cy = 0;
    ArcArcMidpointXError same({&ax, &ay, &bx, &by, &cx, &cy, &ax, &ay, &bx, &by, &cx, &cy});
    EXPECT_TRUE(same.satisfied(0));
    for (const auto& [coordinate, value] : same.gradient()) EXPECT_DOUBLE_EQ(value, 0);
    bx = ax; by = ay;
    EXPECT_FALSE(same.satisfied(1));
    same.setWeight(0);
    EXPECT_TRUE(same.satisfied(0));
}
