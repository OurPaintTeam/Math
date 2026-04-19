#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

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
            << " eigen_sparse_task_init_ms_avg=" << eigenSparseStats.taskInitMs
            << " eigen_sparse_optimize_ms_avg=" << eigenSparseStats.optimizeMs
            << " eigen_sparse_iterations_avg=" << eigenSparseStats.averageIterations
            << " optimize_ratio_sparse_to_eigen_sparse="
            << (eigenSparseStats.optimizeMs == 0.0 ? 0.0 : sparseStats.optimizeMs / eigenSparseStats.optimizeMs)
            << std::endl;

        EXPECT_GT(sparseStats.optimizeMs, 0.0);
        EXPECT_GT(eigenSparseStats.optimizeMs, 0.0);
        EXPECT_GE(sparseStats.averageIterations, 1.0);
        EXPECT_GE(eigenSparseStats.averageIterations, 1.0);
    }
}
