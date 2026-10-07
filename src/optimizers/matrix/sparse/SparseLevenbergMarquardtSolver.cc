#include "sparse/SparseLevenbergMarquardtSolver.h"

#include "SparseQR.h"

#include <algorithm>
#include <cmath>
#include <limits>
#ifdef DEBUG
#include <iostream>
#endif
#include <stdexcept>

namespace {

enum class StopReason {
    ResidualTolerance,
    StationaryPoint,
    StepTooSmall,
    MaxIterations
};

#ifdef DEBUG
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
#endif

double computeGainDenominator(const Matrix<>& step, const Matrix<>& gradient, double lambda) {
    double gain = 0.0;
    for (size_t i = 0; i < step.rows_size(); ++i) {
        gain += step(i, 0) * (lambda * step(i, 0) - gradient(i, 0));
    }
    return 0.5 * gain;
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

bool SparseLMSolver::tryEscapeStationaryPoint() {
    if (!std::isfinite(currentError) || m_result.empty()) return false;

    // Gauss-Newton loses the residual's curvature when J is zero. At a smooth
    // maximum, the objective Hessian supplies a scale for a small escape step.
    // At a norm cusp (coincident points), use the residual norm instead.
    std::vector<double> curvature(m_result.size(), 0.0);
    std::vector<bool> zeroCoordinateGradient(m_result.size());
    const auto& gradient = c_task->normalGradient();
    for (size_t i = 0; i < m_result.size(); ++i) zeroCoordinateGradient[i] = gradient(i, 0) == 0;
    try {
        const auto hessian = c_task->objectiveHessian();
        for (size_t i = 0; i < curvature.size(); ++i) curvature[i] = hessian(i, i);
    } catch (const std::logic_error&) {
        // First-order-only residuals can still use the bounded probe search.
    }

    const double residualNorm = std::sqrt(currentError);
    const double improvementFloor = 32 * std::numeric_limits<double>::epsilon() * currentError;
    std::vector<double> candidate = m_result;
    for (size_t i = 0; i < m_result.size(); ++i) {
        const bool negativeCurvature = std::isfinite(curvature[i]) && curvature[i] < 0;
        if (!negativeCurvature && !zeroCoordinateGradient[i]) continue;
        const double scale = negativeCurvature
            ? residualNorm / std::sqrt(-curvature[i]) : residualNorm;
        if (!std::isfinite(scale) || scale <= 0) continue;
        const double spacing = std::abs(std::nextafter(m_result[i],
            m_result[i] == 0 ? 1.0 : 0.0) - m_result[i]);
        for (const double fraction : {0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1.0}) {
            const double step = std::max(fraction * scale, 8 * spacing);
            for (const double sign : {1.0, -1.0}) {
                candidate[i] = m_result[i] + sign * step;
                if (!std::isfinite(candidate[i]) || candidate[i] == m_result[i]) continue;
                const double candidateError = c_task->setError(candidate);
                if (std::isfinite(candidateError) && currentError - candidateError > improvementFloor) {
                    m_result = candidate;
                    currentError = candidateError;
                    lambda = initialLambda;
                    nu = 2.0;
                    return true;
                }
                c_task->setError(m_result);
            }
        }
        candidate[i] = m_result[i];
    }
    return false;
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
        if (currentError <= errorTolerance) {
            converged = true;
            stopReason = StopReason::ResidualTolerance;
            break;
        }

        c_task->linearizationView();
        const Matrix<>& gradient = c_task->normalGradient();
        const double gradientNorm = gradient.norm();
        if (gradientNorm <= epsilon1) {
            if (tryEscapeStationaryPoint()) {
                ++iteration;
                if (currentError <= errorTolerance) {
                    converged = true;
                    stopReason = StopReason::ResidualTolerance;
                    break;
                }
                continue;
            }
            if (gradientNorm == 0) {
                stopReason = StopReason::StationaryPoint;
                break;
            }
            // A small nonzero gradient is not proof of an unsatisfied minimum:
            // cosine angle residuals flatten near 0 and pi. Let LM finish them.
        }

        Matrix<> step;
        Matrix<> negativeGradient = gradient * (-1.0);
        bool solved = false;
        for (int attempt = 0; attempt < 8; ++attempt) {
            c_task->fillDampedNormalMatrix(lambda, m_dampedNormalMatrix);
            if (m_linearSolver == nullptr) {
                m_linearSolver = std::make_unique<SparseQR>(m_dampedNormalMatrix);
                m_linearSolver->qr();
            } else {
                m_linearSolver->factorize(m_dampedNormalMatrix);
            }

            try {
                step = m_linearSolver->solve(negativeGradient, 0.0);
                solved = true;
                break;
            } catch (const std::runtime_error&) {
                lambda *= nu;
                nu *= 2.0;
            }
        }

        if (!solved) {
            stopReason = StopReason::StepTooSmall;
            break;
        }

        const double stepNorm = step.norm();
        if (stepNorm < epsilon2) {
            if (currentError <= errorTolerance) {
                converged = true;
                stopReason = StopReason::ResidualTolerance;
            } else {
                stopReason = StopReason::StepTooSmall;
            }
            break;
        }

        std::vector<double> candidate = m_result;
        for (size_t i = 0; i < candidate.size(); ++i) {
            candidate[i] += step(i, 0);
        }

        const double candidateError = c_task->setError(candidate);
        // Task error is ||r||^2 (no 1/2 factor); match numerator to denominator scaling.
        const double gainNumerator = currentError - candidateError;
        const double rho = gainNumerator
                         / (computeGainDenominator(step, gradient, lambda) + 1e-20);

        if (rho > 0.0 && std::isfinite(candidateError)) {
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
#ifdef DEBUG
    std::cout << "[Sparse LM] Finished after " << iteration
              << " iterations. Final error: " << currentError
              << (converged ? " [converged]" : " [not converged]")
              << " Reason: " << describeStopReason(stopReason) << std::endl;
#else
    (void)stopReason;
#endif
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
