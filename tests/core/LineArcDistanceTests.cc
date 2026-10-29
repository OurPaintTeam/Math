#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>

TEST(LineArcDistance, MatchesCircleForInteriorEndpointsAndCollapsedSegments) {
    double ax = 3, ay = 4, cx = 0, cy = 0, radius = 5;
    for (auto line : {std::array<double, 4>{-8, -8, 8, -8}, {6, -8, 12, -16},
                      {12, -16, 6, -8}, {0, -7, 0, -7}, {-8, 0, 8, 0}}) {
        for (double target : {0.0, 2.0, 5.0}) {
            SectionArcDistanceError arc({&line[0], &line[1], &line[2], &line[3], &ax, &ay, &cx, &cy}, target);
            SectionCircleDistanceError circle({&line[0], &line[1], &line[2], &line[3], &cx, &cy, &radius}, target);
            EXPECT_DOUBLE_EQ(arc.evaluate(), circle.evaluate());
            for (auto& coordinate : line)
                EXPECT_DOUBLE_EQ(arc.gradient().at(&coordinate), circle.gradient().at(&coordinate));
        }
    }
}

TEST(LineArcDistance, GradientAndHessianIncludeRadiusAndAllProjectionBranches) {
    double ax = 3, ay = 4, cx = 0.3, cy = -0.2;
    for (auto line : {std::array<double, 4>{-8, -8, 8, -8}, {4, 3, 10, 6}, {-10, -6, -4, -3}}) {
        const std::vector<double*> vars{&line[0], &line[1], &line[2], &line[3], &ax, &ay, &cx, &cy};
        SectionArcDistanceError f(vars, 2);
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

TEST(LineArcDistance, SharedEndpointsCancelAndCenterOnSegmentKeepsRadiusDerivatives) {
    double ax = 3, ay = 4, cx = 0, cy = 0, x1 = -8, y1 = 0, x2 = 8, y2 = 0;
    SectionArcDistanceError endpoint({&ax, &ay, &ax, &ay, &ax, &ay, &cx, &cy});
    EXPECT_TRUE(endpoint.satisfied(0));
    for (auto* p : {&ax, &ay, &cx, &cy}) {
        EXPECT_DOUBLE_EQ(endpoint.gradient().at(p), 0);
        for (auto* q : {&ax, &ay, &cx, &cy}) EXPECT_DOUBLE_EQ(endpoint.secondDerivative(p, q), 0);
    }
    SectionArcDistanceError center({&x1, &y1, &x2, &y2, &ax, &ay, &cx, &cy});
    EXPECT_DOUBLE_EQ(center.evaluate(), -5);
    EXPECT_NEAR(center.gradient().at(&ax), -0.6, 1e-12);
    EXPECT_NEAR(center.gradient().at(&cx), 0.6, 1e-12);
}

TEST(LineArcDistance, LiveRadiusTargetsWeightsAndInvalidInputs) {
    double x1 = -8, y1 = -8, x2 = 8, y2 = -8, ax = 3, ay = 4, cx = 0, cy = 0;
    const std::vector<double*> vars{&x1, &y1, &x2, &y2, &ax, &ay, &cx, &cy};
    Variable a(&x1), b(&y1), c(&x2), d(&y2), e(&ax), f(&ay), g(&cx), h(&cy);
    SectionArcDistanceError error(std::vector<Variable*>{&a, &b, &c, &d, &e, &f, &g, &h}, 3);
    EXPECT_TRUE(error.satisfied(0));
    std::unique_ptr<SectionArcDistanceError> copy(error.clone());
    error.setTarget(1);
    error.setWeight(2);
    EXPECT_DOUBLE_EQ(copy->weightedValue(), 4);
    ax = 6; ay = 8;
    EXPECT_DOUBLE_EQ(copy->weightedValue(), -6);
    for (double bad : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(SectionArcDistanceError(vars, bad), std::invalid_argument);
        EXPECT_THROW(error.setTarget(bad), std::invalid_argument);
    }
    ax = cx; ay = cy;
    EXPECT_FALSE(error.satisfied(1e308));
    error.setWeight(0);
    EXPECT_TRUE(error.satisfied(0));
}
