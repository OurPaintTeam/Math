#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <cmath>

TEST(LineArcMidpoint, MajorArcsAndCollapsedLinesHaveIndependentCoordinateErrors) {
    const double s = std::sqrt(0.5);
    double ax = -s-2, ay = s-3, bx = -s+2, by = s+3;
    double px = 1, py = 0, qx = 0, qy = -1, cx = 0, cy = 0;
    const std::vector<double*> vars{&ax, &ay, &bx, &by, &px, &py, &qx, &qy, &cx, &cy};
    LineArcMidpointXError x(vars); LineArcMidpointYError y(vars);
    EXPECT_TRUE(x.satisfied(1e-12));
    EXPECT_TRUE(y.satisfied(1e-12));
    ax = bx = -s; ay = by = s;
    EXPECT_TRUE(x.satisfied(1e-12));
    EXPECT_TRUE(y.satisfied(1e-12));
    ax += 3; bx += 3; ay -= 3; by -= 3;
    EXPECT_NEAR(x.evaluate(), 3, 1e-12);
    EXPECT_NEAR(y.evaluate(), -3, 1e-12);
    EXPECT_FALSE(x.satisfied(1e-6));
    EXPECT_FALSE(y.satisfied(1e-6));
    EXPECT_THROW(x.setTarget(1), std::invalid_argument);
    qx = px; qy = py;
    EXPECT_FALSE(x.satisfied(1));
    x.setWeight(0);
    EXPECT_TRUE(x.satisfied(0));
}

TEST(LineArcMidpoint, TenCoordinateAndSharedEndpointDerivativesMatchFiniteDifferences) {
    double coordinates[] = {2, -3, 3, 4, 4, 5, -2, 6, 1, 1};
    std::vector<double*> vars;
    for (auto& value : coordinates) vars.push_back(&value);
    for (bool shared : {false, true}) {
        if (shared) { vars[0] = vars[4]; vars[1] = vars[5]; }
        LineArcMidpointXError x(vars); LineArcMidpointYError y(vars);
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
}
