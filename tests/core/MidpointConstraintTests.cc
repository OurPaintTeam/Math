#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <limits>

TEST(MidpointConstraint, LinearDerivativesAndSharedCoordinates) {
    double p = 3, a = 1, b = 7;
    MidpointCoordinateError midpoint({&p, &a, &b});
    EXPECT_DOUBLE_EQ(midpoint.evaluate(), -1);
    EXPECT_DOUBLE_EQ(midpoint.gradient().at(&p), 1);
    EXPECT_DOUBLE_EQ(midpoint.gradient().at(&a), -0.5);
    EXPECT_DOUBLE_EQ(midpoint.gradient().at(&b), -0.5);
    EXPECT_DOUBLE_EQ(midpoint.secondDerivative(&p, &b), 0);
    MidpointCoordinateError shared({&p, &p, &b});
    EXPECT_DOUBLE_EQ(shared.gradient().at(&p), 0.5);
    EXPECT_DOUBLE_EQ(shared.gradient().at(&b), -0.5);
    MidpointCoordinateError same({&p, &p, &p});
    EXPECT_DOUBLE_EQ(same.evaluate(), 0);
    EXPECT_DOUBLE_EQ(same.gradient().at(&p), 0);
}

TEST(MidpointConstraint, LargeCoordinatesWeightsAndLiveValues) {
    double p = 1e308, a = 1e308, b = 1e308;
    MidpointCoordinateError midpoint({&p, &a, &b});
    EXPECT_DOUBLE_EQ(midpoint.evaluate(), 0);
    a = 9e307; b = 1.1e308;
    EXPECT_NEAR(midpoint.evaluate() / 1e308, 0, 1e-15);
    p = 4; a = 0; b = 2;
    midpoint.setWeight(2);
    EXPECT_DOUBLE_EQ(midpoint.weightedValue(), 6);
    EXPECT_DOUBLE_EQ(midpoint.weightedGradient().at(&p), 2);
    EXPECT_THROW(midpoint.setTarget(1), std::invalid_argument);
    p = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(midpoint.satisfied(1));
    midpoint.setWeight(0);
    EXPECT_TRUE(midpoint.satisfied(0));
}
