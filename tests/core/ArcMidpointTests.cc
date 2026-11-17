#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

TEST(ArcMidpoint, CounterclockwiseMinorMajorSemicircleAndWrappedAngles) {
    for (double start : {0.0, 45.0, 170.0, 350.0}) {
        for (double sweep : {20.0, 90.0, 180.0, 270.0, 359.9}) {
            const double a = start * std::numbers::pi / 180;
            const double b = (start + sweep) * std::numbers::pi / 180;
            const double mid = (a + b) / 2;
            double cx = 2, cy = 3, ax = cx + 5 * std::cos(a), ay = cy + 5 * std::sin(a);
            double bx = cx + 5 * std::cos(b), by = cy + 5 * std::sin(b);
            double px = cx + 5 * std::cos(mid), py = cy + 5 * std::sin(mid);
            const std::vector<double*> vars{&px, &py, &ax, &ay, &bx, &by, &cx, &cy};
            ArcMidpointXError x(vars); ArcMidpointYError y(vars);
            EXPECT_TRUE(x.satisfied(1e-12));
            EXPECT_TRUE(y.satisfied(1e-12));
        }
    }
}

TEST(ArcMidpoint, DerivativesAreFiniteAcrossSemicirclesAndSharedCoordinates) {
    for (const auto endpoint : {std::pair{-2.0, 6.0}, std::pair{6.0, -2.0}, std::pair{-2.0, -3.0}}) {
        double px = 2, py = -3, ax = 4, ay = 5, bx = endpoint.first, by = endpoint.second, cx = 1, cy = 1;
        for (bool shared : {false, true}) {
            const std::vector<double*> vars{shared ? &cx : &px, shared ? &cy : &py, &ax, &ay, &bx, &by, &cx, &cy};
            ArcMidpointXError x(vars); ArcMidpointYError y(vars);
            for (ErrorFunction* f : {static_cast<ErrorFunction*>(&x), static_cast<ErrorFunction*>(&y)}) {
                for (auto* p : vars) {
                    const double old = *p, h = 1e-5;
                    const auto gradient = f->gradient();
                    *p = old + h; const double plus = f->evaluate(); const auto gp = f->gradient();
                    *p = old - h; const double minus = f->evaluate(); const auto gm = f->gradient();
                    *p = old;
                    EXPECT_NEAR(gradient.at(p), (plus - minus) / (2 * h), 1e-8);
                    for (auto* q : vars)
                        EXPECT_NEAR(f->secondDerivative(p, q), (gp.at(q) - gm.at(q)) / (2 * h), 1e-7);
                }
            }
        }
    }
}

TEST(ArcMidpoint, DegenerateArcIsUndefinedAndDisabledRowsAreIgnored) {
    double x = 1, y = 0, cx = 0, cy = 0;
    ArcMidpointXError sameEnds({&cx, &cy, &x, &y, &x, &y, &cx, &cy});
    EXPECT_FALSE(sameEnds.satisfied(1));
    sameEnds.setWeight(0);
    EXPECT_TRUE(sameEnds.satisfied(0));
    ArcMidpointYError zeroRadius({&x, &y, &cx, &cy, &x, &y, &cx, &cy});
    EXPECT_FALSE(zeroRadius.satisfied(1));
    EXPECT_THROW(zeroRadius.setTarget(1), std::invalid_argument);
}
