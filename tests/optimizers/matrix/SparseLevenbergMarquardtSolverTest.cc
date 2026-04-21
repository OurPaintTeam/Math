#include "gtest/gtest.h"

#include "SparseLSMTask.h"
#include "sparse/SparseLevenbergMarquardtSolver.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

TEST(SparseLSMTaskTest, JacobianStoresOnlyNonZeroEntries) {
    double xValue = 3.0;
    double yValue = -1.0;
    double zValue = 5.0;

    Variable x(&xValue);
    Variable y(&yValue);
    Variable z(&zValue);

    Function* f1 = new Subtraction(x.clone(), new Constant(1.0));
    Function* f2 = new Addition(y.clone(), new Constant(2.0));
    Function* f3 = new Subtraction(new Addition(x.clone(), z.clone()), new Constant(4.0));

    SparseLSMTask task({f1, f2, f3}, {&x, &y, &z});

    auto [residuals, jacobian] = task.linearizeFunction();

    EXPECT_EQ(jacobian.nonZeros(), 4u);
    EXPECT_DOUBLE_EQ(residuals(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(residuals(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(residuals(2, 0), 4.0);
    EXPECT_DOUBLE_EQ(jacobian(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(0, 2), 0.0);
}

TEST(SparseLSMTaskTest, JacobianKeepsStructuralEntriesAtNumericalZero) {
    double xValue = 0.0;
    double yValue = 0.0;

    Variable x(&xValue);
    Variable y(&yValue);

    Function* f = new Multiplication(x.clone(), y.clone());

    SparseLSMTask task({f}, {&x, &y});

    auto [initialResiduals, initialJacobian] = task.linearizeFunction();

    EXPECT_EQ(initialJacobian.nonZeros(), 2u);
    EXPECT_DOUBLE_EQ(initialResiduals(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(initialJacobian(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(initialJacobian(0, 1), 0.0);

    task.setError({2.0, 3.0});
    auto [updatedResiduals, updatedJacobian] = task.linearizeFunction();

    EXPECT_EQ(updatedJacobian.nonZeros(), 2u);
    EXPECT_DOUBLE_EQ(updatedResiduals(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(updatedJacobian(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(updatedJacobian(0, 1), 2.0);
}

TEST(SparseLMSolverTest, ConvergesOnSparseLeastSquaresSystem) {
    double xValue = 0.0;
    double yValue = 0.0;
    double zValue = 0.0;

    Variable x(&xValue);
    Variable y(&yValue);
    Variable z(&zValue);

    Function* f1 = new Subtraction(x.clone(), new Constant(1.0));
    Function* f2 = new Addition(y.clone(), new Constant(2.0));
    Function* f3 = new Subtraction(z.clone(), new Constant(3.0));
    Function* f4 = new Subtraction(new Addition(x.clone(), z.clone()), new Constant(4.0));

    SparseLSMTask task({f1, f2, f3, f4}, {&x, &y, &z});
    SparseLMSolver optimizer(200, 1e-3, 1e-10, 1e-10);

    optimizer.setTask(&task);
    optimizer.optimize();

    const std::vector<double> result = optimizer.getResult();

    EXPECT_TRUE(optimizer.isConverged());
    EXPECT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0], 1.0, 1e-6);
    EXPECT_NEAR(result[1], -2.0, 1e-6);
    EXPECT_NEAR(result[2], 3.0, 1e-6);
    EXPECT_NEAR(optimizer.getCurrentError(), 0.0, 1e-8);
}

namespace {

struct TriangleScenario {
    std::string name;
    int triangleCount;
    double scale;
    double perturbation;
    bool addRightAngleConstraint;
    bool addCrossTriangleDistanceConstraint;
};

struct PointVar {
    size_t xIndex;
    size_t yIndex;
};

struct GeometryChainProblem {
    std::vector<double> values;
    std::vector<double> targetValues;
    std::vector<std::unique_ptr<Variable>> variablesStorage;
    std::vector<Variable*> variableRefs;
    std::vector<Function*> residuals;
    std::unique_ptr<SparseLSMTask> task;
    std::vector<PointVar> anchors;
    std::vector<PointVar> rightVertices;
    std::vector<double> legX;
    std::vector<double> legY;
};

Function* makeDistanceResidual(const PointVar& p,
                               const PointVar& q,
                               const std::vector<Variable*>& vars,
                               double targetDistance) {
    const double targetDistanceSquared = targetDistance * targetDistance;
    Function* dxSquared = new Multiplication(
        new Subtraction(vars[p.xIndex]->clone(), vars[q.xIndex]->clone()),
        new Subtraction(vars[p.xIndex]->clone(), vars[q.xIndex]->clone()));
    Function* dySquared = new Multiplication(
        new Subtraction(vars[p.yIndex]->clone(), vars[q.yIndex]->clone()),
        new Subtraction(vars[p.yIndex]->clone(), vars[q.yIndex]->clone()));
    return new Subtraction(
        new Addition(dxSquared, dySquared),
        new Constant(targetDistanceSquared));
}

Function* makeRightAngleResidual(const PointVar& a,
                                 const PointVar& b,
                                 const PointVar& c,
                                 const std::vector<Variable*>& vars) {
    Function* xDot = new Multiplication(
        new Subtraction(vars[b.xIndex]->clone(), vars[a.xIndex]->clone()),
        new Subtraction(vars[c.xIndex]->clone(), vars[a.xIndex]->clone()));
    Function* yDot = new Multiplication(
        new Subtraction(vars[b.yIndex]->clone(), vars[a.yIndex]->clone()),
        new Subtraction(vars[c.yIndex]->clone(), vars[a.yIndex]->clone()));
    return new Addition(xDot, yDot);
}

Function* makeAnchorResidual(size_t variableIndex,
                             const std::vector<Variable*>& vars,
                             double targetValue) {
    return new Subtraction(vars[variableIndex]->clone(), new Constant(targetValue));
}

Function* makeWeightedAnchorResidual(size_t variableIndex,
                                     const std::vector<Variable*>& vars,
                                     double targetValue,
                                     double weight) {
    Function* residual = makeAnchorResidual(variableIndex, vars, targetValue);
    if (std::abs(weight - 1.0) < 1e-12) {
        return residual;
    }
    return new Multiplication(new Constant(weight), residual);
}

std::vector<Function*> buildTriangleChainResiduals(const GeometryChainProblem& problem,
                                                   const TriangleScenario& scenario,
                                                   bool addMovingAnchorRequirement = false,
                                                   double movingAnchorTargetX = 0.0,
                                                   double movingAnchorTargetY = 0.0,
                                                   double movingAnchorWeight = 1.0) {
    std::vector<Function*> residuals;
    residuals.reserve(problem.residuals.size() + 2);

    for (int i = 0; i < scenario.triangleCount; ++i) {
        const PointVar& a = problem.anchors[i];
        const PointVar& c = problem.anchors[i + 1];
        const PointVar& b = problem.rightVertices[i];

        const double lx = problem.legX[i];
        const double ly = problem.legY[i];
        const double hypotenuse = std::sqrt(lx * lx + ly * ly);

        residuals.push_back(makeDistanceResidual(a, c, problem.variableRefs, lx));
        residuals.push_back(makeDistanceResidual(a, b, problem.variableRefs, ly));
        residuals.push_back(makeDistanceResidual(b, c, problem.variableRefs, hypotenuse));

        if (scenario.addRightAngleConstraint) {
            residuals.push_back(makeRightAngleResidual(a, b, c, problem.variableRefs));
        }
    }

    if (scenario.addCrossTriangleDistanceConstraint) {
        for (int i = 0; i + 1 < scenario.triangleCount; ++i) {
            const PointVar& bCurrent = problem.rightVertices[i];
            const PointVar& bNext = problem.rightVertices[i + 1];
            const double dx = problem.targetValues[bCurrent.xIndex] - problem.targetValues[bNext.xIndex];
            const double dy = problem.targetValues[bCurrent.yIndex] - problem.targetValues[bNext.yIndex];
            residuals.push_back(makeDistanceResidual(
                bCurrent,
                bNext,
                problem.variableRefs,
                std::sqrt(dx * dx + dy * dy)));
        }
    }

    residuals.push_back(makeAnchorResidual(problem.anchors.front().xIndex, problem.variableRefs, 0.0));
    residuals.push_back(makeAnchorResidual(problem.anchors.front().yIndex, problem.variableRefs, 0.0));
    residuals.push_back(makeAnchorResidual(problem.anchors[1].yIndex, problem.variableRefs, 0.0));

    if (addMovingAnchorRequirement) {
        const PointVar& movingAnchor = problem.anchors.back();
        residuals.push_back(makeWeightedAnchorResidual(
            movingAnchor.xIndex,
            problem.variableRefs,
            movingAnchorTargetX,
            movingAnchorWeight));
        residuals.push_back(makeWeightedAnchorResidual(
            movingAnchor.yIndex,
            problem.variableRefs,
            movingAnchorTargetY,
            movingAnchorWeight));
    }

    return residuals;
}

std::unique_ptr<SparseLSMTask> buildTriangleChainMotionTask(const GeometryChainProblem& problem,
                                                            const TriangleScenario& scenario,
                                                            double movingAnchorTargetX,
                                                            double movingAnchorTargetY) {
    constexpr double movingAnchorWeight = 50.0;
    std::vector<Function*> residuals = buildTriangleChainResiduals(
        problem,
        scenario,
        true,
        movingAnchorTargetX,
        movingAnchorTargetY,
        movingAnchorWeight);

    return std::make_unique<SparseLSMTask>(residuals, problem.variableRefs);
}

void setProblemValues(GeometryChainProblem& problem, const std::vector<double>& values) {
    ASSERT_EQ(problem.values.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        problem.values[i] = values[i];
    }
}

GeometryChainProblem buildTriangleChainProblem(const TriangleScenario& scenario) {
    GeometryChainProblem problem;

    const int anchorCount = scenario.triangleCount + 1;
    const int rightVertexCount = scenario.triangleCount;
    const int pointCount = anchorCount + rightVertexCount;
    const int variableCount = pointCount * 2;

    problem.values.reserve(variableCount);
    problem.targetValues.reserve(variableCount);
    problem.variablesStorage.reserve(variableCount);
    problem.variableRefs.reserve(variableCount);
    problem.anchors.reserve(anchorCount);
    problem.rightVertices.reserve(rightVertexCount);
    problem.legX.reserve(scenario.triangleCount);
    problem.legY.reserve(scenario.triangleCount);

    auto appendPoint = [&](double targetX, double targetY) -> PointVar {
        const size_t xIndex = problem.values.size();
        const double initialX = targetX + scenario.perturbation * (0.35 + 0.04 * static_cast<double>(xIndex % 7));
        problem.values.push_back(initialX);
        problem.targetValues.push_back(targetX);
        problem.variablesStorage.emplace_back(std::make_unique<Variable>(&problem.values[xIndex]));
        problem.variableRefs.push_back(problem.variablesStorage.back().get());

        const size_t yIndex = problem.values.size();
        const double initialY = targetY - scenario.perturbation * (0.28 + 0.05 * static_cast<double>(yIndex % 5));
        problem.values.push_back(initialY);
        problem.targetValues.push_back(targetY);
        problem.variablesStorage.emplace_back(std::make_unique<Variable>(&problem.values[yIndex]));
        problem.variableRefs.push_back(problem.variablesStorage.back().get());

        return {xIndex, yIndex};
    };

    double currentX = 0.0;
    for (int i = 0; i < anchorCount; ++i) {
        problem.anchors.push_back(appendPoint(currentX, 0.0));
        if (i < scenario.triangleCount) {
            const double lx = scenario.scale * (1.0 + 0.08 * static_cast<double>(i % 4));
            problem.legX.push_back(lx);
            currentX += lx;
        }
    }

    for (int i = 0; i < rightVertexCount; ++i) {
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        const double ly = scenario.scale * (0.7 + 0.06 * static_cast<double>(i % 5));
        problem.legY.push_back(ly);
        problem.rightVertices.push_back(appendPoint(problem.targetValues[problem.anchors[i].xIndex],
                                                    sign * ly));
    }

    problem.residuals = buildTriangleChainResiduals(problem, scenario);
    problem.task = std::make_unique<SparseLSMTask>(problem.residuals, problem.variableRefs);
    return problem;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double maxDiff = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        maxDiff = std::max(maxDiff, std::abs(a[i] - b[i]));
    }
    return maxDiff;
}

double squaredDistanceAt(const std::vector<double>& values, const PointVar& p, const PointVar& q) {
    const double dx = values[p.xIndex] - values[q.xIndex];
    const double dy = values[p.yIndex] - values[q.yIndex];
    return dx * dx + dy * dy;
}

} // namespace

TEST(SparseLMSolverGeometryTest, ConvergesOnConnectedTwentyRightTriangles) {
    const TriangleScenario scenario{
        "twenty_connected_triangles",
        20,
        1.2,
        0.45,
        true,
        true
    };

    GeometryChainProblem problem = buildTriangleChainProblem(scenario);
    SparseLMSolver optimizer(600, 1e-2, 1e-10, 1e-10);
    optimizer.setTask(problem.task.get());

    const double initialError = problem.task->getError();
    optimizer.optimize();
    const std::vector<double> result = optimizer.getResult();

    EXPECT_TRUE(optimizer.isConverged());
    EXPECT_LT(optimizer.getCurrentError(), 1e-8);
    EXPECT_LT(optimizer.getCurrentError(), initialError);
    EXPECT_EQ(result.size(), problem.targetValues.size());
    EXPECT_LT(maxAbsDiff(result, problem.targetValues), 3e-3);
}

TEST(SparseLMSolverGeometryTest, TracksIncrementalAnchorMotion) {
    const TriangleScenario scenario{
        "incremental_last_anchor_motion",
        12,
        1.5,
        0.38,
        true,
        true
    };

    GeometryChainProblem problem = buildTriangleChainProblem(scenario);
    const PointVar movingAnchor = problem.anchors.back();
    const double initialTargetX = problem.targetValues[movingAnchor.xIndex];
    const double initialTargetY = problem.targetValues[movingAnchor.yIndex];
    const double stepDelta = 0.01;
    const int stepCount = 20;
    const double tolerance = 1e-2;

    std::vector<double> currentValues = problem.values;

    for (int step = 1; step <= stepCount; ++step) {
        const double targetX = initialTargetX + stepDelta * static_cast<double>(step);
        const double targetY = initialTargetY + stepDelta * static_cast<double>(step);

        setProblemValues(problem, currentValues);
        std::unique_ptr<SparseLSMTask> motionTask =
            buildTriangleChainMotionTask(problem, scenario, targetX, targetY);

        SparseLMSolver optimizer(600, 1e-2, 1e-10, 1e-10);
        optimizer.setTask(motionTask.get());

        const double initialError = motionTask->getError();
        const double initialXDistance = std::abs(currentValues[movingAnchor.xIndex] - targetX);
        const double initialYDistance = std::abs(currentValues[movingAnchor.yIndex] - targetY);
        optimizer.optimize();

        const std::vector<double> result = optimizer.getResult();
        ASSERT_EQ(result.size(), currentValues.size()) << "step: " << step;
        EXPECT_TRUE(optimizer.isConverged()) << "step: " << step;
        EXPECT_LT(optimizer.getCurrentError(), initialError) << "step: " << step;
        EXPECT_LT(std::abs(result[movingAnchor.xIndex] - targetX), initialXDistance) << "step: " << step;
        EXPECT_LT(std::abs(result[movingAnchor.yIndex] - targetY), initialYDistance) << "step: " << step;
        EXPECT_NEAR(result[movingAnchor.xIndex], targetX, tolerance) << "step: " << step;
        EXPECT_NEAR(result[movingAnchor.yIndex], targetY, tolerance) << "step: " << step;

        currentValues = result;
    }

    const double totalDelta = stepDelta * static_cast<double>(stepCount);
    EXPECT_NEAR(currentValues[movingAnchor.xIndex], initialTargetX + totalDelta, tolerance);
    EXPECT_NEAR(currentValues[movingAnchor.yIndex], initialTargetY + totalDelta, tolerance);
}

class SparseLMSolverGeometryScenarioTest : public ::testing::TestWithParam<TriangleScenario> {};

TEST_P(SparseLMSolverGeometryScenarioTest, HandlesDifferentGeometricConstraintsAndSizes) {
    const TriangleScenario scenario = GetParam();
    GeometryChainProblem problem = buildTriangleChainProblem(scenario);

    SparseLMSolver optimizer(500, 5e-3, 1e-9, 1e-9);
    optimizer.setTask(problem.task.get());

    const double initialError = problem.task->getError();
    optimizer.optimize();

    const std::vector<double> result = optimizer.getResult();
    EXPECT_TRUE(optimizer.isConverged()) << "Scenario: " << scenario.name;
    EXPECT_LT(optimizer.getCurrentError(), std::max(1e-8, initialError * 1e-10))
        << "Scenario: " << scenario.name;

    const double tolerance = 7e-3;
    for (int i = 0; i < scenario.triangleCount; ++i) {
        const PointVar& a = problem.anchors[i];
        const PointVar& c = problem.anchors[i + 1];
        const PointVar& b = problem.rightVertices[i];

        const double lx = problem.legX[i];
        const double ly = problem.legY[i];
        const double hypotenuse = std::sqrt(lx * lx + ly * ly);

        EXPECT_NEAR(squaredDistanceAt(result, a, c), lx * lx, tolerance)
            << "Scenario: " << scenario.name << " triangle: " << i;
        EXPECT_NEAR(squaredDistanceAt(result, a, b), ly * ly, tolerance)
            << "Scenario: " << scenario.name << " triangle: " << i;
        EXPECT_NEAR(squaredDistanceAt(result, b, c), hypotenuse * hypotenuse, tolerance)
            << "Scenario: " << scenario.name << " triangle: " << i;

        if (scenario.addRightAngleConstraint) {
            const double bax = result[b.xIndex] - result[a.xIndex];
            const double bay = result[b.yIndex] - result[a.yIndex];
            const double cax = result[c.xIndex] - result[a.xIndex];
            const double cay = result[c.yIndex] - result[a.yIndex];
            EXPECT_NEAR(bax * cax + bay * cay, 0.0, tolerance)
                << "Scenario: " << scenario.name << " triangle: " << i;
        }
    }

    if (scenario.addCrossTriangleDistanceConstraint) {
        for (int i = 0; i + 1 < scenario.triangleCount; ++i) {
            const PointVar& bCurrent = problem.rightVertices[i];
            const PointVar& bNext = problem.rightVertices[i + 1];
            const double targetDx =
                problem.targetValues[bCurrent.xIndex] - problem.targetValues[bNext.xIndex];
            const double targetDy =
                problem.targetValues[bCurrent.yIndex] - problem.targetValues[bNext.yIndex];
            EXPECT_NEAR(
                squaredDistanceAt(result, bCurrent, bNext),
                targetDx * targetDx + targetDy * targetDy,
                tolerance)
                << "Scenario: " << scenario.name << " cross-link: " << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    GeometryVariations,
    SparseLMSolverGeometryScenarioTest,
    ::testing::Values(
        TriangleScenario{"triangles_04_small", 4, 0.8, 0.18, false, false},
        TriangleScenario{"triangles_05_small_with_links", 5, 0.9, 0.22, false, true},
        TriangleScenario{"triangles_06_small_with_right_angles", 6, 0.85, 0.20, true, false},
        TriangleScenario{"triangles_07_mid_plain", 7, 1.1, 0.25, false, false},
        TriangleScenario{"triangles_08_mid_with_links", 8, 1.2, 0.30, false, true},
        TriangleScenario{"triangles_09_mid_with_right_angles", 9, 1.05, 0.27, true, false},
        TriangleScenario{"triangles_10_mid_full_constraints", 10, 1.15, 0.33, true, true},
        TriangleScenario{"triangles_12_large_plain", 12, 1.5, 0.38, false, false},
        TriangleScenario{"triangles_12_large_with_links", 12, 1.7, 0.40, false, true},
        TriangleScenario{"triangles_14_large_with_right_angles", 14, 1.6, 0.42, true, false},
        TriangleScenario{"triangles_15_large_full_constraints", 15, 1.8, 0.48, true, true},
        TriangleScenario{"triangles_20_full_constraints", 20, 1.3, 0.46, true, true}),
    [](const testing::TestParamInfo<TriangleScenario>& info) {
        return info.param.name;
    });
