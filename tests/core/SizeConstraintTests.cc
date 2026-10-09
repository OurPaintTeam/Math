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

TEST(SizeConstraints, RadiusIsAWeightedResidualWithoutDirectAssignment) {
    double radius = 3;
    Variable variable(&radius);
    CircleRadiusError f({&radius}, 5);
    EXPECT_DOUBLE_EQ(f.evaluate(), -2);
    EXPECT_DOUBLE_EQ(f.gradient().at(&radius), 1);
    double* coordinate = nullptr;
    double target = 0;
    EXPECT_FALSE(f.assignment(coordinate, target));
    auto clone = std::unique_ptr<CircleRadiusError>(f.clone());
    auto weighted = std::unique_ptr<Function>(f.weightedFunction());
    auto derivative = std::unique_ptr<Function>(weighted->derivative(&variable));
    f.setWeight(2);
    f.setTarget(4);
    EXPECT_DOUBLE_EQ(clone->evaluate(), -1);
    EXPECT_DOUBLE_EQ(weighted->evaluate(), -2);
    EXPECT_DOUBLE_EQ(derivative->evaluate(), 2);
    radius = -1;
    EXPECT_FALSE(std::isfinite(f.evaluate()));
    f.setWeight(0);
    EXPECT_DOUBLE_EQ(weighted->evaluate(), 0);
    EXPECT_DOUBLE_EQ(derivative->evaluate(), 0);
}

TEST(SizeConstraints, DiameterUsesRadiusUnitsForEquivalentAndConflictingSizes) {
    for (double diameter : {20.0, 30.0}) {
        double radius = 3;
        Variable variable(&radius);
        CircleRadiusError prescribedRadius({&radius}, 10);
        CircleRadiusError prescribedDiameter({&radius}, diameter / 2);
        EXPECT_DOUBLE_EQ(prescribedDiameter.gradient().at(&radius), 1);
        if (diameter == 20) {
            EXPECT_DOUBLE_EQ(prescribedRadius.evaluate(), prescribedDiameter.evaluate());
            EXPECT_EQ(prescribedRadius.weightedGradient(), prescribedDiameter.weightedGradient());
        }
        SparseLSMTask task({prescribedRadius.weightedFunction(), prescribedDiameter.weightedFunction()}, {&variable});
        SparseLMSolver optimizer;
        optimizer.setTask(&task);
        ASSERT_NO_THROW(optimizer.optimize());
        EXPECT_EQ(optimizer.isConverged(), diameter == 20);
        EXPECT_NEAR(radius, diameter == 20 ? 10 : 12.5, 1e-6);
        EXPECT_NEAR(task.getError(), diameter == 20 ? 0 : 12.5, 1e-8);
    }
}

TEST(SizeConstraints, RadiusTargetsMustBePositiveAndFinite) {
    double first = 2, second = 3;
    const double inf = std::numeric_limits<double>::infinity();
    for (double invalid : {0.0, -1.0, inf, std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(CircleRadiusError({&first}, invalid), std::invalid_argument);
        CircleRadiusError f({&first}, 4);
        EXPECT_THROW(f.setTarget(invalid), std::invalid_argument);
        EXPECT_DOUBLE_EQ(f.evaluate(), -2);

    }
}

TEST(SizeConstraints, EqualLengthsSumSharedCoordinatesAndAllowZeroLengths) {
    double ax = 0, ay = 0, bx = 3, by = 4, cx = 0, cy = 2;
    EqualLengthError equal({&ax, &ay, &bx, &by, &ax, &ay, &cx, &cy});
    EXPECT_DOUBLE_EQ(equal.evaluate(), 3);
    EXPECT_NEAR(equal.gradient().at(&ax), -0.6, 1e-12);
    EXPECT_NEAR(equal.gradient().at(&ay), 0.2, 1e-12);
    for (auto* coordinate : {&ax, &ay}) {
        const double original = *coordinate, step = 1e-5;
        *coordinate = original + step;
        const double plus = equal.evaluate();
        const double gradientPlus = equal.gradient().at(coordinate);
        *coordinate = original - step;
        const double minus = equal.evaluate();
        const double gradientMinus = equal.gradient().at(coordinate);
        *coordinate = original;
        EXPECT_NEAR(equal.gradient().at(coordinate), (plus - minus) / (2 * step), 1e-8);
        EXPECT_NEAR(equal.secondDerivative(coordinate, coordinate), (gradientPlus - gradientMinus) / (2 * step), 1e-8);
    }
    EqualLengthError self({&ax, &ay, &bx, &by, &ax, &ay, &bx, &by});
    EXPECT_DOUBLE_EQ(self.evaluate(), 0);
    for (const auto& [coordinate, derivative] : self.gradient()) EXPECT_DOUBLE_EQ(derivative, 0);
    bx = ax; by = ay; cx = ax; cy = ay;
    EXPECT_DOUBLE_EQ(equal.evaluate(), 0);
    for (const auto& [coordinate, derivative] : equal.gradient()) EXPECT_DOUBLE_EQ(derivative, 0);
    EXPECT_TRUE(equal.satisfied(0));
    EXPECT_THROW(equal.setTarget(1), std::invalid_argument);
}

TEST(SizeConstraints, EqualLengthsRemainStableAcrossScales) {
    for (double scale : {1e-300, 1e-200, 1.0, 1e200, 1e300}) {
        double zero = 0, x = 3 * scale, y = 4 * scale, other = 2 * scale;
        EqualLengthError equal({&zero, &zero, &x, &y, &zero, &zero, &other, &zero});
        EXPECT_NEAR(equal.evaluate() / scale, 3, 1e-12);
        EXPECT_NEAR(equal.gradient().at(&x), 0.6, 1e-12);
    }
}

TEST(SizeConstraints, AnInvalidRadiusTrialRestoresTheAcceptedCoordinatesAndCaches) {
    double value = 1;
    Variable radius(&value);
    CircleRadiusError positive({&value}, 1);
    SparseLSMTask task({positive.weightedFunction(), new Multiplication(new Constant(10),
        new Addition(radius.clone(), new Constant(5)))}, {&radius});
    SparseLMSolver optimizer(1);
    optimizer.setTask(&task);
    ASSERT_NO_THROW(optimizer.optimize());
    EXPECT_FALSE(optimizer.isConverged());
    EXPECT_GT(positive.invalidEvaluations(), 0u);
    EXPECT_DOUBLE_EQ(value, 1);
    EXPECT_EQ(task.getValues(), optimizer.getResult());
    EXPECT_DOUBLE_EQ(task.getError(), optimizer.getCurrentError());
    EXPECT_DOUBLE_EQ(task.normalGradient()(0, 0), 600);
}
