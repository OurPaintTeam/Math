#include "sparse/SparseDogLegSolver.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

SparseDogLegSolver::SparseDogLegSolver(int maxIterations,
                                       double initialTrustRegionRadius,
                                       double maxTrustRegionRadius,
                                       double eta,
                                       double epsilon1,
                                       double epsilon2,
                                       double errorTolerance,
                                       double linearSolverDamping)
    : c_task(nullptr),
      converged(false),
      currentError(0.0),
      initialTrustRegionRadius(initialTrustRegionRadius),
      trustRegionRadius(initialTrustRegionRadius),
      maxTrustRegionRadius(maxTrustRegionRadius),
      eta(eta),
      epsilon1(epsilon1),
      epsilon2(epsilon2),
      errorTolerance(errorTolerance),
      linearSolverDamping(linearSolverDamping),
      maxIterations(maxIterations),
      performedIterations(0) {}

void SparseDogLegSolver::setTask(TaskMatrix* task) {
    if (task == nullptr) {
        throw std::runtime_error("Task is null");
    }

    c_task = dynamic_cast<SparseLSMTask*>(task);
    if (c_task == nullptr) {
        throw std::runtime_error("Task is not SparseLSMTask");
    }

    m_result = c_task->getValues();
    currentError = c_task->getError();
    trustRegionRadius = initialTrustRegionRadius;
    converged = false;
    performedIterations = 0;
}

Matrix<> SparseDogLegSolver::buildDogLegStep(
    const Matrix<>& gradient,
    const Matrix<>& gaussNewtonStep) const
{
    const double gaussNewtonNorm = gaussNewtonStep.norm();
    if (gaussNewtonNorm <= trustRegionRadius) {
        return gaussNewtonStep;
    }

    Matrix<> hGradient = m_normalMatrix * gradient;
    double gradientSquaredNorm = 0.0;
    double curvature = 0.0;
    for (size_t i = 0; i < gradient.rows_size(); ++i) {
        for (size_t j = 0; j < gradient.cols_size(); ++j) {
            gradientSquaredNorm += gradient(i, j) * gradient(i, j);
            curvature += gradient(i, j) * hGradient(i, j);
        }
    }

    const double alpha = curvature > 0.0 ? gradientSquaredNorm / curvature : 1.0;
    Matrix<> steepestDescentStep = gradient * (-alpha);
    const double steepestDescentNorm = steepestDescentStep.norm();

    if (steepestDescentNorm >= trustRegionRadius) {
        return steepestDescentStep * (trustRegionRadius / steepestDescentNorm);
    }

    Matrix<> dogLegDirection = gaussNewtonStep - steepestDescentStep;
    double a = 0.0;
    double b = 0.0;
    double c = -trustRegionRadius * trustRegionRadius;
    for (size_t i = 0; i < dogLegDirection.rows_size(); ++i) {
        for (size_t j = 0; j < dogLegDirection.cols_size(); ++j) {
            a += dogLegDirection(i, j) * dogLegDirection(i, j);
            b += 2.0 * steepestDescentStep(i, j) * dogLegDirection(i, j);
            c += steepestDescentStep(i, j) * steepestDescentStep(i, j);
        }
    }

    const double discriminant = std::max(0.0, b * b - 4.0 * a * c);
    const double tau = (-b + std::sqrt(discriminant)) / (2.0 * a);
    return steepestDescentStep + dogLegDirection * tau;
}

void SparseDogLegSolver::optimize() {
    if (c_task == nullptr) {
        throw std::runtime_error("Task is not set");
    }

    converged = false;
    performedIterations = 0;
    currentError = c_task->setError(m_result);

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        if (currentError <= errorTolerance) {
            converged = true;
            break;
        }

        c_task->linearizationView();
        const Matrix<>& gradient = c_task->normalGradient();
        if (gradient.norm() < epsilon1) {
            converged = currentError <= errorTolerance;
            break;
        }

        c_task->fillApproximateHessian(m_normalMatrix);
        if (m_linearSolver == nullptr) {
            m_linearSolver = std::make_unique<SparseQR>(m_normalMatrix);
            m_linearSolver->qr();
        } else {
            m_linearSolver->factorize(m_normalMatrix);
        }

        const Matrix<> rhs = gradient * (-1.0);
        Matrix<> gaussNewtonStep;
        try {
            gaussNewtonStep = m_linearSolver->solve(rhs, 0.0);
        } catch (const std::runtime_error&) {
            gaussNewtonStep = m_linearSolver->pseudoInverse(linearSolverDamping) * rhs;
        }

        Matrix<> step = buildDogLegStep(gradient, gaussNewtonStep);

        if (step.norm() < epsilon2) {
            converged = currentError <= errorTolerance;
            break;
        }

        Matrix<> hStep = m_normalMatrix * step;
        double gradientStep = 0.0;
        double stepHStep = 0.0;
        for (size_t i = 0; i < step.rows_size(); ++i) {
            for (size_t j = 0; j < step.cols_size(); ++j) {
                gradientStep += gradient(i, j) * step(i, j);
                stepHStep += step(i, j) * hStep(i, j);
            }
        }

        const double predictedReduction = -(2.0 * gradientStep + stepHStep);
        if (predictedReduction <= 0.0) {
            trustRegionRadius *= 0.25;
            if (trustRegionRadius < epsilon2) {
                break;
            }
            continue;
        }

        std::vector<double> candidate = m_result;
        for (size_t i = 0; i < candidate.size(); ++i) {
            candidate[i] += step(i, 0);
        }

        const double candidateError = c_task->setError(candidate);
        const double actualReduction = currentError - candidateError;
        const double rho = actualReduction / predictedReduction;
        const bool nearBoundary =
            std::abs(step.norm() - trustRegionRadius)
            <= 1e-8 * std::max(1.0, trustRegionRadius);

        if (rho < 0.25) {
            trustRegionRadius *= 0.25;
        } else if (rho > 0.75 && nearBoundary) {
            trustRegionRadius = std::min(2.0 * trustRegionRadius, maxTrustRegionRadius);
        }

        if (rho > eta && std::isfinite(candidateError) && actualReduction > 0.0) {
            m_result = candidate;
            currentError = candidateError;
            ++performedIterations;

            if (currentError <= errorTolerance) {
                converged = true;
                break;
            }
        } else {
            currentError = c_task->setError(m_result);
            if (trustRegionRadius < epsilon2) {
                break;
            }
        }
    }

    currentError = c_task->setError(m_result);
}

std::vector<double> SparseDogLegSolver::getResult() const {
    return m_result;
}

double SparseDogLegSolver::getCurrentError() const {
    return currentError;
}

bool SparseDogLegSolver::isConverged() const {
    return converged;
}

int SparseDogLegSolver::getIterationCount() const {
    return performedIterations;
}
