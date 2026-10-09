#include "ErrorFunction.h"
#include "SparseLSMTask.h"
#include "sparse/SparseLevenbergMarquardtSolver.h"
#include <gtest/gtest.h>
#include <limits>
#include <memory>

TEST(SizeConstraints, PointOnCircleHasSignedDistanceAndSumsAliasedCoordinates) {
    double x = 3, y = 4, cx = 0, cy = 0, radius = 2;
    PointOnCircleError point({&x, &y, &cx, &cy, &radius});
    EXPECT_DOUBLE_EQ(point.evaluate(), 3);
    EXPECT_NEAR(point.gradient().at(&x), 0.6, 1e-12);
    EXPECT_NEAR(point.gradient().at(&y), 0.8, 1e-12);
    EXPECT_NEAR(point.gradient().at(&cx), -0.6, 1e-12);
    EXPECT_NEAR(point.gradient().at(&cy), -0.8, 1e-12);
    EXPECT_DOUBLE_EQ(point.gradient().at(&radius), -1);
    PointOnCircleError shared({&x, &y, &x, &cy, &radius});
    EXPECT_DOUBLE_EQ(shared.evaluate(), 2);
    EXPECT_DOUBLE_EQ(shared.gradient().at(&x), 0);
    radius = 6;
    EXPECT_DOUBLE_EQ(point.evaluate(), -1);
}

TEST(SizeConstraints, PointOnCircleRecoversFromTheCenterWithoutMovingFixedCircle) {
    double x = 0, y = 0, cx = 0, cy = 0, radius = 5;
    Variable vx(&x), vy(&y);
    PointOnCircleError point({&x, &y, &cx, &cy, &radius});
    SparseLSMTask task({point.weightedFunction()}, {&vx, &vy});
    SparseLMSolver optimizer(200, 1e-3, 1e-10, 1e-10, 1e-12);
    optimizer.setTask(&task);
    ASSERT_NO_THROW(optimizer.optimize());
    ASSERT_TRUE(optimizer.isConverged());
    EXPECT_NEAR(std::hypot(x, y), 5, 1e-6);
    EXPECT_DOUBLE_EQ(cx, 0);
    EXPECT_DOUBLE_EQ(cy, 0);
    EXPECT_DOUBLE_EQ(radius, 5);
}
