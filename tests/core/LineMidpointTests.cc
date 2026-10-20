#include "ErrorFunction.h"
#include <gtest/gtest.h>

TEST(LineMidpoint, IndependentLinearCoordinatesAndSharedEndpoints) {
    double a = 1, b = 7, c = 2, d = 8;
    MidpointDifferenceError f({&a, &b, &c, &d});
    EXPECT_DOUBLE_EQ(f.evaluate(), -1);
    EXPECT_DOUBLE_EQ(f.gradient().at(&a), 0.5);
    EXPECT_DOUBLE_EQ(f.gradient().at(&b), 0.5);
    EXPECT_DOUBLE_EQ(f.gradient().at(&c), -0.5);
    EXPECT_DOUBLE_EQ(f.gradient().at(&d), -0.5);
    EXPECT_DOUBLE_EQ(f.secondDerivative(&a, &b), 0);
    MidpointDifferenceError shared({&a, &b, &a, &d});
    EXPECT_DOUBLE_EQ(shared.gradient().at(&a), 0);
    MidpointDifferenceError same({&a, &b, &a, &b});
    EXPECT_TRUE(same.satisfied(0));
    EXPECT_DOUBLE_EQ(same.gradient().at(&b), 0);
    d = 6;
    EXPECT_TRUE(f.satisfied(0));
    f.setWeight(2);
    b = 9;
    EXPECT_DOUBLE_EQ(f.weightedValue(), 2);
    EXPECT_THROW(f.setTarget(1), std::invalid_argument);
}

TEST(LineMidpoint, LargeOffsetsAndCollapsedLines) {
    double a = 1e308, b = 1e308;
    MidpointDifferenceError large({&a, &a, &b, &b});
    EXPECT_TRUE(large.satisfied(0));
    b = 9e307;
    EXPECT_NEAR(large.evaluate() / 1e307, 1, 1e-14);
}
