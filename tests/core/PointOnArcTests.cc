#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <cmath>

TEST(PointOnArc, SupportingCircleAndLiveRadius) {
    double px = 0, py = -5, ax = 3, ay = 4, cx = 0, cy = 0, radius = 5;
    PointOnArcError arc({&px, &py, &ax, &ay, &cx, &cy});
    PointOnCircleError circle({&px, &py, &cx, &cy, &radius});
    EXPECT_TRUE(arc.satisfied(0));
    EXPECT_DOUBLE_EQ(arc.evaluate(), circle.evaluate());
    ax = 6; ay = 8; radius = 10;
    EXPECT_DOUBLE_EQ(arc.evaluate(), circle.evaluate());
    ax = cx; ay = cy;
    EXPECT_FALSE(arc.satisfied(1));
    arc.setWeight(0);
    EXPECT_TRUE(arc.satisfied(0));
}

TEST(PointOnArc, DerivativesAndSharedEndpoints) {
    double coordinates[] = {2, -3, 4, 5, 1, 1};
    std::vector<double*> variables;
    for (auto& value : coordinates) variables.push_back(&value);
    PointOnArcError f(variables);
    for (auto* p : variables) {
        const double old = *p, h = 1e-5;
        const auto gradient = f.gradient();
        *p = old + h; const double plus = f.evaluate(); const auto gp = f.gradient();
        *p = old - h; const double minus = f.evaluate(); const auto gm = f.gradient();
        *p = old;
        EXPECT_NEAR(gradient.at(p), (plus - minus) / (2 * h), 1e-8);
        for (auto* q : variables)
            EXPECT_NEAR(f.secondDerivative(p, q), (gp.at(q) - gm.at(q)) / (2 * h), 1e-8);
    }
    PointOnArcError shared({variables[0], variables[1], variables[0], variables[1], variables[4], variables[5]});
    EXPECT_TRUE(shared.satisfied(0));
    for (const auto& [coordinate, value] : shared.gradient()) EXPECT_DOUBLE_EQ(value, 0);
    EXPECT_THROW(f.setTarget(1), std::invalid_argument);
}
