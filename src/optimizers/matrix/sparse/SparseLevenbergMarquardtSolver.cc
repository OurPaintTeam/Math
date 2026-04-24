#include "sparse/SparseLevenbergMarquardtSolver.h"

#include "SparseQR.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

enum class StopReason {
    ResidualTolerance,
    StationaryPoint,
    StepTooSmall,
    MaxIterations
};

double computeGainDenominator(const Matrix<>& step, const Matrix<>& gradient, double lambda) {
    double gain = 0.0;
    for (size_t i = 0; i < step.rows_size(); ++i) {
        gain += step(i, 0) * (lambda * step(i, 0) - gradient(i, 0));
    }
    return 0.5 * gain;
}

const char* describeStopReason(StopReason reason) {
    switch (reason) {
        case StopReason::ResidualTolerance:
            return "residual tolerance reached";
        case StopReason::StationaryPoint:
            return "stationary point with non-zero residual";
        case StopReason::StepTooSmall:
            return "step too small with non-zero residual";
        case StopReason::MaxIterations:
            return "maximum iterations reached";
    }

    return "unknown";
}

} // namespace

SparseLMSolver::SparseLMSolver(int maxIterations,
                               double initLambda,
                               double epsilon1,
                               double epsilon2,
                               double errorTolerance)
    : c_task(nullptr),
      converged(false),
      currentError(0.0),
      initialLambda(initLambda),
      lambda(initLambda),
      nu(2.0),
      epsilon1(epsilon1),
      epsilon2(epsilon2),
      errorTolerance(errorTolerance),
      maxIterations(maxIterations),
      performedIterations(0) {}

void SparseLMSolver::setTask(TaskMatrix* task) {
    if (task == nullptr) {
        throw std::runtime_error("Task is null");
    }

    c_task = dynamic_cast<SparseLSMTask*>(task);
    if (c_task == nullptr) {
        throw std::runtime_error("Task is not SparseLSMTask");
    }

    m_result = c_task->getValues();
    currentError = c_task->getError();
    converged = false;
    lambda = initialLambda;
    nu = 2.0;
    performedIterations = 0;
}

void SparseLMSolver::optimize() {
    if (c_task == nullptr) {
        throw std::runtime_error("Task is not set");
    }

    converged = false;
    performedIterations = 0;
    currentError = c_task->setError(m_result);

    int iteration = 0;
    StopReason stopReason = StopReason::MaxIterations;
    while (iteration < maxIterations) {
<<<<<<< HEAD
        if (currentError <= errorTolerance) {
            converged = true;
            stopReason = StopReason::ResidualTolerance;
            break;
        }

        c_task->linearizationView();
=======
>>>>>>> 1e12c1b1650384183c68999802d011d4e05bc24f
        const Matrix<>& gradient = c_task->normalGradient();
        const double gradientNorm = gradient.norm();
        if (gradientNorm < epsilon1) {
            stopReason = StopReason::StationaryPoint;
            break;
        }

        SparseMatrix<> dampedNormalMatrix = c_task->dampedNormalMatrix(lambda);

        SparseQR solver(dampedNormalMatrix);
        solver.qr();

        Matrix<> step;
        Matrix<> negativeGradient = gradient * (-1.0);
        step = solver.solve(negativeGradient, 0.0);

        const double stepNorm = step.norm();
        if (stepNorm < epsilon2) {
            stopReason = StopReason::StepTooSmall;
            break;
        }

        std::vector<double> candidate = m_result;
        for (size_t i = 0; i < candidate.size(); ++i) {
            candidate[i] += step(i, 0);
        }

        const double gainDenominator = computeGainDenominator(step, gradient, lambda);
        const double candidateError = c_task->setError(candidate);
        const double gainNumerator = currentError - candidateError;
        const double rho = gainNumerator
                         / (gainDenominator + 1e-20);

        if (std::isfinite(candidateError) && std::isfinite(rho) && rho > 0.0) {
            m_result = candidate;
            currentError = candidateError;
            lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
            nu = 2.0;

            if (currentError <= errorTolerance) {
                converged = true;
                stopReason = StopReason::ResidualTolerance;
                ++iteration;
                break;
            }
        } else {
            c_task->setError(m_result);
            lambda *= nu;
            nu *= 2.0;
        }

        ++iteration;
    }

    performedIterations = iteration;
    currentError = c_task->setError(m_result);
    std::cout << "[Sparse LM] Finished after " << iteration
              << " iterations. Final error: " << currentError
              << (converged ? " [converged]" : " [not converged]")
              << " Reason: " << describeStopReason(stopReason) << std::endl;
}

std::vector<double> SparseLMSolver::getResult() const {
    return m_result;
}

double SparseLMSolver::getCurrentError() const {
    return currentError;
}

bool SparseLMSolver::isConverged() const {
    return converged;
}

int SparseLMSolver::getIterationCount() const {
    return performedIterations;
}
