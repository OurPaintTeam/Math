#include "sparse/SparseNewtonGaussSolver.h"

#include <cmath>
#include <memory>
#include <stdexcept>

SparseNewtonGaussSolver::SparseNewtonGaussSolver(int maxIterations,
                                                 double epsilon1,
                                                 double epsilon2,
                                                 double errorTolerance,
                                                 double linearSolverDamping)
    : c_task(nullptr),
      converged(false),
      currentError(0.0),
      epsilon1(epsilon1),
      epsilon2(epsilon2),
      errorTolerance(errorTolerance),
      linearSolverDamping(linearSolverDamping),
      maxIterations(maxIterations),
      performedIterations(0) {}

void SparseNewtonGaussSolver::setTask(TaskMatrix* task) {
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
    performedIterations = 0;
}

void SparseNewtonGaussSolver::optimize() {
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
        Matrix<> negativeGradient = gradient * (-1.0);
        if (m_linearSolver == nullptr) {
            m_linearSolver = std::make_unique<SparseQR>(m_normalMatrix);
            m_linearSolver->qr();
        } else {
            m_linearSolver->factorize(m_normalMatrix);
        }

        Matrix<> step;
        try {
            step = m_linearSolver->solve(negativeGradient, 0.0);
        } catch (const std::runtime_error&) {
            step = m_linearSolver->pseudoInverse(linearSolverDamping) * negativeGradient;
        }

        if (step.norm() < epsilon2) {
            converged = currentError <= errorTolerance;
            break;
        }

        bool accepted = false;
        double stepScale = 1.0;
        std::vector<double> candidate;
        double candidateError = currentError;
        while (stepScale > 1e-8) {
            candidate = m_result;
            for (size_t i = 0; i < candidate.size(); ++i) {
                candidate[i] += stepScale * step(i, 0);
            }
            candidateError = c_task->setError(candidate);
            if (std::isfinite(candidateError) && candidateError < currentError) {
                accepted = true;
                break;
            }
            stepScale *= 0.5;
        }

        if (!accepted) {
            currentError = c_task->setError(m_result);
            break;
        }

        m_result = candidate;
        currentError = candidateError;
        ++performedIterations;

        if (currentError <= errorTolerance || step.norm() * stepScale < epsilon2) {
            converged = currentError <= errorTolerance;
            break;
        }
    }

    currentError = c_task->setError(m_result);
}

std::vector<double> SparseNewtonGaussSolver::getResult() const {
    return m_result;
}

double SparseNewtonGaussSolver::getCurrentError() const {
    return currentError;
}

bool SparseNewtonGaussSolver::isConverged() const {
    return converged;
}

int SparseNewtonGaussSolver::getIterationCount() const {
    return performedIterations;
}
