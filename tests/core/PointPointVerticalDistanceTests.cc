#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <limits>

TEST(PointPointVerticalDistance, BothDirectionsDerivativesAndLiveTargets) {
    double y1 = 2, x1 = 8, y2 = 7, x2 = -9;
    PointPointVerticalDistanceError f({&x1, &y1, &x2, &y2}, 4);
    EXPECT_DOUBLE_EQ(f.evaluate(), 1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&y1), -1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&y2), 1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&x1), 0);
    EXPECT_DOUBLE_EQ(f.gradient().at(&x2), 0);
    EXPECT_DOUBLE_EQ(f.secondDerivative(&y1, &y2), 0);
    y2 = -3;
    EXPECT_DOUBLE_EQ(f.evaluate(), 1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&y1), 1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&y2), -1);
    f.setTarget(5);
    EXPECT_TRUE(f.satisfied(0));
    EXPECT_THROW(f.setTarget(-1), std::invalid_argument);
    EXPECT_THROW(f.setTarget(std::numeric_limits<double>::infinity()), std::invalid_argument);
    y2 = y1;
    EXPECT_DOUBLE_EQ(f.evaluate(), -5);
    EXPECT_TRUE(std::isfinite(f.gradient().at(&y1)));
    f.setWeight(0);
    EXPECT_TRUE(f.satisfied(0));
}

TEST(PointPointVerticalDistance, SharedCoordinatesAndLargeOffsets) {
    double y = 1e308, x = 1;
    PointPointVerticalDistanceError shared({&x, &y, &x, &y}, 0);
    EXPECT_TRUE(shared.satisfied(0));
    EXPECT_DOUBLE_EQ(shared.gradient().at(&y), 0);
    double other = 9e307;
    PointPointVerticalDistanceError large({&x, &y, &x, &other}, 1e307);
    EXPECT_NEAR(large.evaluate() / 1e307, 0, 1e-14);
}
