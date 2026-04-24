#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "ErrorFunction.h"
#include "LSMFORLMTask.h"
#include "LMWithSparse.h"
#include "SparseLSMTask.h"
#include "sparse/SparseLevenbergMarquardtSolver.h"

namespace {

struct VariableSet {
    std::vector<double> values;
    std::vector<std::unique_ptr<Variable>> owned;
    std::vector<Variable*> raw;
};

struct PointVar {
    size_t xIndex;
    size_t yIndex;
};

struct TriangleScenario {
    const char* name;
    int triangleCount;
    double scale;
    double perturbation;
    bool addRightAngleConstraint;
    bool addCrossTriangleDistanceConstraint;
};

struct GeometryChainProblem {
    std::vector<double> values;
    std::vector<double> targetValues;
    std::vector<std::unique_ptr<Variable>> variablesStorage;
    std::vector<Variable*> variableRefs;
    std::vector<PointVar> anchors;
    std::vector<PointVar> rightVertices;
    std::vector<double> legX;
    std::vector<double> legY;
};

struct MotionResidualBundle {
    std::vector<double> fixedValues;
    std::vector<Function*> residuals;
};

struct RunStats {
    double taskInitMs = 0.0;
    double optimizeMs = 0.0;
    double finalError = 0.0;
    double averageIterations = 0.0;
};

enum class ResidualPattern {
    Chain2,
    Stencil3,
    Block4
};

struct BenchmarkScenario {
    const char* name;
    size_t variableCount;
    int repeats;
    int maxIterations;
    ResidualPattern pattern;
};

constexpr double kChainCoupling = 0.35;
constexpr double kStencilCoupling1 = -0.45;
constexpr double kStencilCoupling2 = 0.2;
constexpr double kBlockCoupling1 = 0.25;
constexpr double kBlockCoupling2 = -0.4;
constexpr double kBlockCoupling3 = 0.3;
constexpr double kBlockCoupling4 = -0.2;
constexpr double kBlockCoupling5 = 0.15;
constexpr double kMovingAnchorWeight = 50.0;

VariableSet MakeVariableSet(const std::vector<double>& initialValues) {
    VariableSet result;
    result.values = initialValues;
    result.owned.reserve(initialValues.size());
    result.raw.reserve(initialValues.size());

    for (size_t i = 0; i < result.values.size(); ++i) {
        result.owned.push_back(std::make_unique<Variable>(&result.values[i]));
        result.raw.push_back(result.owned.back().get());
    }
    return result;
}

GeometryChainProblem BuildTriangleChainProblem(const TriangleScenario& scenario) {
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
        problem.rightVertices.push_back(appendPoint(problem.targetValues[problem.anchors[i].xIndex], sign * ly));
    }

    return problem;
}

MotionResidualBundle BuildTriangleChainErrorResiduals(const GeometryChainProblem& problem,
                                                      const TriangleScenario& scenario,
                                                      bool addMovingAnchorRequirement = false,
                                                      double movingAnchorTargetX = 0.0,
                                                      double movingAnchorTargetY = 0.0) {
    MotionResidualBundle bundle;
    bundle.fixedValues.reserve(addMovingAnchorRequirement ? 4 : 2);
    bundle.residuals.reserve(4 * static_cast<size_t>(scenario.triangleCount) + 16);

    auto appendFixedValue = [&](double value) -> double* {
        bundle.fixedValues.push_back(value);
        return &bundle.fixedValues.back();
    };

    auto cloneProblemVariable = [&](size_t index) -> Variable* {
        return problem.variableRefs[index]->clone();
    };

    for (int i = 0; i < scenario.triangleCount; ++i) {
        const PointVar& a = problem.anchors[i];
        const PointVar& c = problem.anchors[i + 1];
        const PointVar& b = problem.rightVertices[i];

        const double lx = problem.legX[i];
        const double ly = problem.legY[i];
        const double hypotenuse = std::sqrt(lx * lx + ly * ly);

        bundle.residuals.push_back(new PointPointDistanceError(
            {cloneProblemVariable(a.xIndex), cloneProblemVariable(a.yIndex),
             cloneProblemVariable(c.xIndex), cloneProblemVariable(c.yIndex)},
            lx));
        bundle.residuals.push_back(new PointPointDistanceError(
            {cloneProblemVariable(a.xIndex), cloneProblemVariable(a.yIndex),
             cloneProblemVariable(b.xIndex), cloneProblemVariable(b.yIndex)},
            ly));
        bundle.residuals.push_back(new PointPointDistanceError(
            {cloneProblemVariable(b.xIndex), cloneProblemVariable(b.yIndex),
             cloneProblemVariable(c.xIndex), cloneProblemVariable(c.yIndex)},
            hypotenuse));

        if (scenario.addRightAngleConstraint) {
            bundle.residuals.push_back(new SectionSectionPerpendicularError(
                {cloneProblemVariable(a.xIndex), cloneProblemVariable(a.yIndex),
                 cloneProblemVariable(b.xIndex), cloneProblemVariable(b.yIndex),
                 cloneProblemVariable(a.xIndex), cloneProblemVariable(a.yIndex),
                 cloneProblemVariable(c.xIndex), cloneProblemVariable(c.yIndex)}));
        }
    }

    if (scenario.addCrossTriangleDistanceConstraint) {
        for (int i = 0; i + 1 < scenario.triangleCount; ++i) {
            const PointVar& bCurrent = problem.rightVertices[i];
            const PointVar& bNext = problem.rightVertices[i + 1];
            const double dx = problem.targetValues[bCurrent.xIndex] - problem.targetValues[bNext.xIndex];
            const double dy = problem.targetValues[bCurrent.yIndex] - problem.targetValues[bNext.yIndex];
            bundle.residuals.push_back(new PointPointDistanceError(
                {cloneProblemVariable(bCurrent.xIndex), cloneProblemVariable(bCurrent.yIndex),
                 cloneProblemVariable(bNext.xIndex), cloneProblemVariable(bNext.yIndex)},
                std::sqrt(dx * dx + dy * dy)));
        }
    }

    double* originX = appendFixedValue(0.0);
    double* originY = appendFixedValue(0.0);
    bundle.residuals.push_back(new PointOnPointError(
        {cloneProblemVariable(problem.anchors.front().xIndex), cloneProblemVariable(problem.anchors.front().yIndex),
         new Variable(originX), new Variable(originY)}));

    if (addMovingAnchorRequirement) {
        const PointVar& movingAnchor = problem.anchors.back();
        double* movingTargetX = appendFixedValue(movingAnchorTargetX);
        double* movingTargetY = appendFixedValue(movingAnchorTargetY);
        const int duplicateRequirements = static_cast<int>(kMovingAnchorWeight);
        for (int i = 0; i < duplicateRequirements; ++i) {
            bundle.residuals.push_back(new PointOnPointError(
                {cloneProblemVariable(movingAnchor.xIndex), cloneProblemVariable(movingAnchor.yIndex),
                 new Variable(movingTargetX), new Variable(movingTargetY)}));
        }
    }

    return bundle;
}

void SetGeometryProblemValues(GeometryChainProblem& problem, const std::vector<double>& values) {
    ASSERT_EQ(problem.values.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        problem.values[i] = values[i];
    }
}

std::vector<double> BuildReferenceSolution(size_t variableCount) {
    std::vector<double> solution(variableCount, 0.0);
    for (size_t i = 0; i < variableCount; ++i) {
        solution[i] = std::sin(static_cast<double>(i) * 0.15) + 0.01 * static_cast<double>(i);
    }
    return solution;
}

std::vector<double> BuildInitialValues(size_t variableCount) {
    std::vector<double> initialValues(variableCount, 0.0);
    for (size_t i = 0; i < variableCount; ++i) {
        initialValues[i] =
            0.3 * std::cos(static_cast<double>(i) * 0.11) - 0.015 * static_cast<double>(i);
    }
    return initialValues;
}

size_t CountResiduals(size_t variableCount, ResidualPattern pattern) {
    switch (pattern) {
        case ResidualPattern::Chain2:
            return variableCount + (variableCount > 0 ? variableCount - 1 : 0);
        case ResidualPattern::Stencil3:
            return variableCount + (variableCount > 2 ? variableCount - 2 : 0);
        case ResidualPattern::Block4:
            return variableCount + 4 * (variableCount / 4);
    }
    return variableCount;
}

std::vector<Function*> BuildResidualFunctions(
    const std::vector<Variable*>& variables,
    const std::vector<double>& solution,
    ResidualPattern pattern)
{
    std::vector<Function*> residuals;
    residuals.reserve(CountResiduals(variables.size(), pattern));

    for (size_t i = 0; i < variables.size(); ++i) {
        residuals.push_back(new Subtraction(
            variables[i]->clone(),
            new Constant(solution[i])));
    }

    switch (pattern) {
        case ResidualPattern::Chain2:
            for (size_t i = 0; i + 1 < variables.size(); ++i) {
                const double target = solution[i] + kChainCoupling * solution[i + 1];
                residuals.push_back(new Subtraction(
                    new Addition(
                        variables[i]->clone(),
                        new Multiplication(new Constant(kChainCoupling), variables[i + 1]->clone())),
                    new Constant(target)));
            }
            break;
        case ResidualPattern::Stencil3:
            for (size_t i = 0; i + 2 < variables.size(); ++i) {
                const double target =
                    solution[i]
                    + kStencilCoupling1 * solution[i + 1]
                    + kStencilCoupling2 * solution[i + 2];
                residuals.push_back(new Subtraction(
                    new Addition(
                        new Addition(
                            variables[i]->clone(),
                            new Multiplication(new Constant(kStencilCoupling1), variables[i + 1]->clone())),
                        new Multiplication(new Constant(kStencilCoupling2), variables[i + 2]->clone())),
                    new Constant(target)));
            }
            break;
        case ResidualPattern::Block4:
            for (size_t base = 0; base + 3 < variables.size(); base += 4) {
                const double target1 = solution[base] + kBlockCoupling1 * solution[base + 1];
                residuals.push_back(new Subtraction(
                    new Addition(
                        variables[base]->clone(),
                        new Multiplication(new Constant(kBlockCoupling1), variables[base + 1]->clone())),
                    new Constant(target1)));

                const double target2 = solution[base + 1] + kBlockCoupling2 * solution[base + 2];
                residuals.push_back(new Subtraction(
                    new Addition(
                        variables[base + 1]->clone(),
                        new Multiplication(new Constant(kBlockCoupling2), variables[base + 2]->clone())),
                    new Constant(target2)));

                const double target3 = solution[base + 2] + kBlockCoupling3 * solution[base + 3];
                residuals.push_back(new Subtraction(
                    new Addition(
                        variables[base + 2]->clone(),
                        new Multiplication(new Constant(kBlockCoupling3), variables[base + 3]->clone())),
                    new Constant(target3)));

                const double target4 =
                    solution[base]
                    + kBlockCoupling4 * solution[base + 2]
                    + kBlockCoupling5 * solution[base + 3];
                residuals.push_back(new Subtraction(
                    new Addition(
                        new Addition(
                            variables[base]->clone(),
                            new Multiplication(new Constant(kBlockCoupling4), variables[base + 2]->clone())),
                        new Multiplication(new Constant(kBlockCoupling5), variables[base + 3]->clone())),
                    new Constant(target4)));
            }
            break;
    }

    return residuals;
}

RunStats RunSparseLmBenchmark(
    const BenchmarkScenario& scenario,
    const std::vector<double>& initialValues,
    const std::vector<double>& solution)
{
    RunStats stats;

    for (int repeat = 0; repeat < scenario.repeats; ++repeat) {
        VariableSet variables = MakeVariableSet(initialValues);
        std::vector<Function*> residuals =
            BuildResidualFunctions(variables.raw, solution, scenario.pattern);

        const auto taskBegin = std::chrono::steady_clock::now();
        SparseLSMTask task(std::move(residuals), variables.raw);
        const auto taskEnd = std::chrono::steady_clock::now();

        SparseLMSolver solver(scenario.maxIterations, 1e-3, 1e-10, 1e-10);
        solver.setTask(&task);

        const auto optimizeBegin = std::chrono::steady_clock::now();
        solver.optimize();
        const auto optimizeEnd = std::chrono::steady_clock::now();

        EXPECT_TRUE(solver.isConverged());
        EXPECT_NEAR(solver.getCurrentError(), 0.0, 1e-8);

        stats.taskInitMs += std::chrono::duration<double, std::milli>(taskEnd - taskBegin).count();
        stats.optimizeMs += std::chrono::duration<double, std::milli>(optimizeEnd - optimizeBegin).count();
        stats.finalError += solver.getCurrentError();
        stats.averageIterations += static_cast<double>(solver.getIterationCount());
    }

    stats.taskInitMs /= static_cast<double>(scenario.repeats);
    stats.optimizeMs /= static_cast<double>(scenario.repeats);
    stats.finalError /= static_cast<double>(scenario.repeats);
    stats.averageIterations /= static_cast<double>(scenario.repeats);
    return stats;
}

RunStats RunEigenSparseLmBenchmark(
    const BenchmarkScenario& scenario,
    const std::vector<double>& initialValues,
    const std::vector<double>& solution)
{
    RunStats stats;

    for (int repeat = 0; repeat < scenario.repeats; ++repeat) {
        VariableSet variables = MakeVariableSet(initialValues);
        std::vector<Function*> residuals =
            BuildResidualFunctions(variables.raw, solution, scenario.pattern);

        double taskInitMs = 0.0;
        double optimizeMs = 0.0;
        double finalError = 0.0;
        int iterations = 0;
        bool converged = false;

        {
            const auto taskBegin = std::chrono::steady_clock::now();
            LSMFORLMTask task(residuals, variables.raw);
            const auto taskEnd = std::chrono::steady_clock::now();

            LMSparse solver(scenario.maxIterations, 1e-3, 1e-10, 1e-10);
            solver.setTask(&task);

            const auto optimizeBegin = std::chrono::steady_clock::now();
            solver.optimize();
            const auto optimizeEnd = std::chrono::steady_clock::now();

            taskInitMs = std::chrono::duration<double, std::milli>(taskEnd - taskBegin).count();
            optimizeMs = std::chrono::duration<double, std::milli>(optimizeEnd - optimizeBegin).count();
            finalError = solver.getCurrentError();
            iterations = solver.getIterationCount();
            converged = solver.isConverged();
        }

        for (Function* function : residuals) {
            delete function;
        }

        EXPECT_TRUE(converged);
        EXPECT_NEAR(finalError, 0.0, 1e-8);

        stats.taskInitMs += taskInitMs;
        stats.optimizeMs += optimizeMs;
        stats.finalError += finalError;
        stats.averageIterations += static_cast<double>(iterations);
    }

    stats.taskInitMs /= static_cast<double>(scenario.repeats);
    stats.optimizeMs /= static_cast<double>(scenario.repeats);
    stats.finalError /= static_cast<double>(scenario.repeats);
    stats.averageIterations /= static_cast<double>(scenario.repeats);
    return stats;
}

} // namespace

TEST(SparseLMComparisonPerformanceTests, CompareSparseAndEigenSparseAcrossScenarios) {
    const std::vector<BenchmarkScenario> scenarios = {
        {"chain_96", 96, 5, 250, ResidualPattern::Chain2},
        {"stencil3_128", 128, 4, 250, ResidualPattern::Stencil3},
        {"block4_160", 160, 4, 250, ResidualPattern::Block4},
        {"chain_224", 224, 3, 300, ResidualPattern::Chain2}
    };

    for (const BenchmarkScenario& scenario : scenarios) {
        SCOPED_TRACE(scenario.name);

        const std::vector<double> referenceSolution =
            BuildReferenceSolution(scenario.variableCount);
        const std::vector<double> initialValues =
            BuildInitialValues(scenario.variableCount);

        const RunStats sparseStats =
            RunSparseLmBenchmark(scenario, initialValues, referenceSolution);
        const RunStats eigenSparseStats =
            RunEigenSparseLmBenchmark(scenario, initialValues, referenceSolution);

        std::cout
            << "[SparseLMComparison] scenario=" << scenario.name
            << " variables=" << scenario.variableCount
            << " residuals=" << CountResiduals(scenario.variableCount, scenario.pattern)
            << " repeats=" << scenario.repeats
            << " sparse_task_init_ms_avg=" << sparseStats.taskInitMs
            << " sparse_optimize_ms_avg=" << sparseStats.optimizeMs
            << " sparse_iterations_avg=" << sparseStats.averageIterations
            << " eigen_sparse_qr_task_init_ms_avg=" << eigenSparseStats.taskInitMs
            << " eigen_sparse_qr_optimize_ms_avg=" << eigenSparseStats.optimizeMs
            << " eigen_sparse_qr_iterations_avg=" << eigenSparseStats.averageIterations
            << " optimize_ratio_sparse_to_eigen_sparse_qr="
            << (eigenSparseStats.optimizeMs == 0.0 ? 0.0 : sparseStats.optimizeMs / eigenSparseStats.optimizeMs)
            << std::endl;

        EXPECT_GT(sparseStats.optimizeMs, 0.0);
        EXPECT_GT(eigenSparseStats.optimizeMs, 0.0);
        EXPECT_GE(sparseStats.averageIterations, 1.0);
        EXPECT_GE(eigenSparseStats.averageIterations, 1.0);
    }
}

TEST(SparseLMComparisonPerformanceTests, CompareSparseAndEigenSparseOnIncrementalAnchorMotion) {
    const TriangleScenario scenario{
        "incremental_last_anchor_motion",
        12,
        1.5,
        0.38,
        true,
        true
    };

    GeometryChainProblem problem = BuildTriangleChainProblem(scenario);
    const PointVar movingAnchor = problem.anchors.back();
    const double initialTargetX = problem.targetValues[movingAnchor.xIndex];
    const double initialTargetY = problem.targetValues[movingAnchor.yIndex];
    const double stepDelta = 0.001;
    const int stepCount = 20;
    const double tolerance = 1e-2;
    const double solverAgreementTolerance = 1e-2;

    std::vector<double> currentSparseValues = problem.values;
    std::vector<double> currentEigenValues = problem.values;
    double sparseOptimizeMsTotal = 0.0;
    double eigenOptimizeMsTotal = 0.0;
    double sparseIterationsTotal = 0.0;
    double eigenIterationsTotal = 0.0;

    for (int step = 1; step <= stepCount; ++step) {
        const double targetX = initialTargetX + stepDelta * static_cast<double>(step);
        const double targetY = initialTargetY + stepDelta * static_cast<double>(step);

        SetGeometryProblemValues(problem, currentSparseValues);
        SetGeometryProblemValues(problem, currentEigenValues);
        MotionResidualBundle eigenBundle =
            BuildTriangleChainErrorResiduals(problem, scenario, true, targetX, targetY);
        LSMFORLMTask eigenTask(eigenBundle.residuals, problem.variableRefs);
        LMSparse eigenSolver(600, 1e-2, 1e-6, 1e-6);
        eigenSolver.setTask(&eigenTask);

        const auto eigenOptimizeBegin = std::chrono::steady_clock::now();
        eigenSolver.optimize();
        const auto eigenOptimizeEnd = std::chrono::steady_clock::now();
        eigenOptimizeMsTotal +=
            std::chrono::duration<double, std::milli>(eigenOptimizeEnd - eigenOptimizeBegin).count();
        eigenIterationsTotal += static_cast<double>(eigenSolver.getIterationCount());

        const std::vector<double> eigenResult = eigenSolver.getResult();
        ASSERT_EQ(eigenResult.size(), currentEigenValues.size()) << "step: " << step;
        EXPECT_TRUE(eigenSolver.isConverged()) << "step: " << step;
        EXPECT_NEAR(eigenResult[movingAnchor.xIndex], targetX, tolerance) << "step: " << step;
        EXPECT_NEAR(eigenResult[movingAnchor.yIndex], targetY, tolerance) << "step: " << step;

        SetGeometryProblemValues(problem, currentSparseValues);
        MotionResidualBundle sparseBundle =
            BuildTriangleChainErrorResiduals(problem, scenario, true, targetX, targetY);
        SparseLSMTask sparseTask(std::move(sparseBundle.residuals), problem.variableRefs);
        SparseLMSolver sparseSolver(600, 1e-2, 1e-6, 1e-6);
        sparseSolver.setTask(&sparseTask);

        const auto sparseOptimizeBegin = std::chrono::steady_clock::now();
        sparseSolver.optimize();
        const auto sparseOptimizeEnd = std::chrono::steady_clock::now();
        sparseOptimizeMsTotal +=
            std::chrono::duration<double, std::milli>(sparseOptimizeEnd - sparseOptimizeBegin).count();
        sparseIterationsTotal += static_cast<double>(sparseSolver.getIterationCount());

        const std::vector<double> sparseResult = sparseSolver.getResult();
        ASSERT_EQ(sparseResult.size(), currentSparseValues.size()) << "step: " << step;
        EXPECT_TRUE(sparseSolver.isConverged()) << "step: " << step;
        EXPECT_NEAR(sparseResult[movingAnchor.xIndex], targetX, tolerance) << "step: " << step;
        EXPECT_NEAR(sparseResult[movingAnchor.yIndex], targetY, tolerance) << "step: " << step;

        EXPECT_NEAR(sparseResult[movingAnchor.xIndex], eigenResult[movingAnchor.xIndex], solverAgreementTolerance)
            << "step: " << step;
        EXPECT_NEAR(sparseResult[movingAnchor.yIndex], eigenResult[movingAnchor.yIndex], solverAgreementTolerance)
            << "step: " << step;

        currentSparseValues = sparseResult;
        currentEigenValues = eigenResult;
    }

    const double totalDelta = stepDelta * static_cast<double>(stepCount);
    EXPECT_NEAR(currentSparseValues[movingAnchor.xIndex], initialTargetX + totalDelta, tolerance);
    EXPECT_NEAR(currentSparseValues[movingAnchor.yIndex], initialTargetY + totalDelta, tolerance);
    EXPECT_NEAR(currentEigenValues[movingAnchor.xIndex], initialTargetX + totalDelta, tolerance);
    EXPECT_NEAR(currentEigenValues[movingAnchor.yIndex], initialTargetY + totalDelta, tolerance);
    EXPECT_GT(sparseOptimizeMsTotal, 0.0);
    EXPECT_GT(eigenOptimizeMsTotal, 0.0);
    EXPECT_GE(sparseIterationsTotal, static_cast<double>(stepCount));
    EXPECT_GE(eigenIterationsTotal, static_cast<double>(stepCount));

    std::cout
        << "[SparseLMMotionComparison] scenario=" << scenario.name
        << " steps=" << stepCount
        << " sparse_optimize_ms_total=" << sparseOptimizeMsTotal
        << " sparse_iterations_avg=" << (sparseIterationsTotal / static_cast<double>(stepCount))
        << " eigen_sparse_qr_optimize_ms_total=" << eigenOptimizeMsTotal
        << " eigen_sparse_qr_iterations_avg=" << (eigenIterationsTotal / static_cast<double>(stepCount))
        << " optimize_ratio_sparse_to_eigen_sparse_qr="
        << (eigenOptimizeMsTotal == 0.0 ? 0.0 : sparseOptimizeMsTotal / eigenOptimizeMsTotal)
        << std::endl;
}
