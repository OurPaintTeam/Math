#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace { constexpr double pi = std::numbers::pi; }

TEST(ArcAngle, CounterclockwiseMinorMajorSemicircleAndWrappedEndpoints) {
    for (double start : {0.0, 45.0, 170.0, 350.0}) {
        for (double sweep : {0.1, 20.0, 90.0, 180.0, 270.0, 359.9}) {
            const double a = start * pi / 180, b = (start + sweep) * pi / 180;
            double cx = 2, cy = 3, ax = cx + 5 * std::cos(a), ay = cy + 5 * std::sin(a);
            double bx = cx + 7 * std::cos(b), by = cy + 7 * std::sin(b);
            ArcAngleError f({&ax, &ay, &bx, &by, &cx, &cy}, sweep * pi / 180);
            EXPECT_TRUE(f.satisfied(1e-12));
        }
    }
}

TEST(ArcAngle, ResidualUsesShortestCorrectionAndDistinguishesOppositeSweeps) {
    double ax = 1, ay = 0, bx = std::cos(0.2), by = std::sin(0.2), cx = 0, cy = 0;
    ArcAngleError f({&ax, &ay, &bx, &by, &cx, &cy}, 2*pi - 0.2);
    EXPECT_NEAR(f.evaluate(), 0.4, 1e-12);
    by = -by;
    f.setTarget(0.2);
    EXPECT_NEAR(f.evaluate(), -0.4, 1e-12);
    bx = 0; by = -1;
    f.setTarget(pi / 2);
    EXPECT_NEAR(std::abs(f.evaluate()), pi, 1e-12);
    f.setTarget(3*pi / 2);
    EXPECT_TRUE(f.satisfied(1e-12));
    ArcAngleError reversed({&bx, &by, &ax, &ay, &cx, &cy}, pi / 2);
    EXPECT_TRUE(reversed.satisfied(1e-12));
}

TEST(ArcAngle, GradientsAndHessiansMatchFiniteDifferencesIncludingSemicircles) {
    for (auto coordinates : {std::array<double, 6>{4, 5, -2, 6, 1, 1},
                             std::array<double, 6>{-2, 6, 4, 5, 1, 1},
                             std::array<double, 6>{3, 4, -3, -4, 0, 0}}) {
        std::vector<double*> vars;
        for (auto& coordinate : coordinates) vars.push_back(&coordinate);
        ArcAngleError f(vars, 2.5);
        const auto gradient = f.gradient();
        for (auto* p : vars) {
            const double old = *p, h = 1e-5;
            *p = old + h; const double plus = f.evaluate(); const auto gp = f.gradient();
            *p = old - h; const double minus = f.evaluate(); const auto gm = f.gradient();
            *p = old;
            EXPECT_NEAR(gradient.at(p), (plus - minus) / (2*h), 1e-8);
            for (auto* q : vars)
                EXPECT_NEAR(f.secondDerivative(p, q), (gp.at(q) - gm.at(q)) / (2*h), 1e-7);
        }
    }
}

TEST(ArcAngle, SharedCoordinatesCancelAndExtremeScalesStayFinite) {
    double a = 1, b = 2, zero = 0;
    ArcAngleError shared({&a, &zero, &zero, &b, &zero, &zero}, pi / 2);
    EXPECT_TRUE(shared.satisfied(1e-12));
    for (auto* p : {&a, &b, &zero}) {
        EXPECT_NEAR(shared.gradient().at(p), 0, 1e-14);
        for (auto* q : {&a, &b, &zero}) EXPECT_NEAR(shared.secondDerivative(p, q), 0, 1e-14);
    }
    for (double scale : {1e-150, 1.0, 1e150}) {
        double ax = scale, ay = 0, bx = 0, by = scale, cx = 0, cy = 0;
        const std::vector<double*> vars{&ax, &ay, &bx, &by, &cx, &cy};
        ArcAngleError f(vars, pi / 2);
        EXPECT_TRUE(f.satisfied(1e-12));
        for (auto* p : vars) {
            EXPECT_TRUE(std::isfinite(f.gradient().at(p)));
            for (auto* q : vars) EXPECT_TRUE(std::isfinite(f.secondDerivative(p, q)));
        }
    }
}

TEST(ArcAngle, ClonesVariablesTargetsAndWeightsRemainLive) {
    double ax = 1, ay = 0, bx = 0, by = 1, cx = 0, cy = 0;
    Variable a(&ax), b(&ay), c(&bx), d(&by), e(&cx), f(&cy);
    ArcAngleError angle(std::vector<Variable*>{&a, &b, &c, &d, &e, &f}, pi / 2);
    std::unique_ptr<ArcAngleError> copy(angle.clone());
    angle.setTarget(pi);
    angle.setWeight(2);
    EXPECT_NEAR(copy->weightedValue(), -pi, 1e-12);
    bx = -1; by = 0;
    EXPECT_TRUE(copy->satisfied(1e-12));
    EXPECT_NEAR(copy->weightedGradient().at(&by), -2, 1e-12);
}

TEST(ArcAngle, InvalidTargetsAndDegenerateArcsAreRejected) {
    double ax = 1, ay = 0, bx = 0, by = 1, cx = 0, cy = 0;
    const std::vector<double*> vars{&ax, &ay, &bx, &by, &cx, &cy};
    ArcAngleError f(vars, pi / 2);
    for (double bad : {0.0, -1.0, 2*pi, 2*pi + 1, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(ArcAngleError(vars, bad), std::invalid_argument);
        EXPECT_THROW(f.setTarget(bad), std::invalid_argument);
    }
    for (int variant = 0; variant < 3; ++variant) {
        ax = variant == 0 ? 0 : 1; ay = 0;
        bx = variant == 2 ? 1 : 0; by = variant == 0 ? 1 : 0;
        EXPECT_FALSE(f.satisfied(100));
    }
    f.setWeight(0);
    EXPECT_TRUE(f.satisfied(0));
}
