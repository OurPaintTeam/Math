#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <limits>

TEST(PointPointHorizontalDistance, BothDirectionsDerivativesAndLiveTargets) {
    double x1 = 2, y1 = 8, x2 = 7, y2 = -9;
    PointPointHorizontalDistanceError f({&x1, &y1, &x2, &y2}, 4);
    EXPECT_DOUBLE_EQ(f.evaluate(), 1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&x1), -1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&x2), 1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&y1), 0);
    EXPECT_DOUBLE_EQ(f.gradient().at(&y2), 0);
    EXPECT_DOUBLE_EQ(f.secondDerivative(&x1, &x2), 0);
    x2 = -3;
    EXPECT_DOUBLE_EQ(f.evaluate(), 1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&x1), 1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&x2), -1);
    f.setTarget(5);
    EXPECT_TRUE(f.satisfied(0));
    EXPECT_THROW(f.setTarget(-1), std::invalid_argument);
    EXPECT_THROW(f.setTarget(std::numeric_limits<double>::infinity()), std::invalid_argument);
    x2 = x1;
    EXPECT_DOUBLE_EQ(f.evaluate(), -5);
    EXPECT_TRUE(std::isfinite(f.gradient().at(&x1)));
    f.setWeight(0);
    EXPECT_TRUE(f.satisfied(0));
}

TEST(PointPointHorizontalDistance, SharedCoordinatesAndLargeOffsets) {
    double x = 1e308, y = 1;
    PointPointHorizontalDistanceError shared({&x, &y, &x, &y}, 0);
    EXPECT_TRUE(shared.satisfied(0));
    EXPECT_DOUBLE_EQ(shared.gradient().at(&x), 0);
    double other = 9e307;
    PointPointHorizontalDistanceError large({&x, &y, &other, &y}, 1e307);
    EXPECT_NEAR(large.evaluate() / 1e307, 0, 1e-14);
}
