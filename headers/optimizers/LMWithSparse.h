#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_LMWITHSPARSE_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_LMWITHSPARSE_H_

#include "Optimizer.h"
#include "LSMFORLMTask.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>

class LMSparse {
private:
    LSMFORLMTask* task;
    Eigen::VectorXd result;
    bool converged = false;
    double currentError = 0.0;
    double lambda = 1e-3;
    double nu = 2.0;

    double epsilon1 = 1e-8;
    double epsilon2 = 1e-8;
    int maxIterations = 100;

    Eigen::SparseMatrix<double> jacobian;
    bool structureInitialized = false;

public:
    LMSparse(int maxIterations = 100,
                             double initLambda = 1e-3,
                             double epsilon1 = 1e-8,
                             double epsilon2 = 1e-8)
            : maxIterations(maxIterations),
              lambda(initLambda),
              epsilon1(epsilon1),
              epsilon2(epsilon2) {}

    void setTask(LSMFORLMTask* task) {
        this->task = dynamic_cast<LSMFORLMTask*>(task);
        if (!this->task) {
            throw std::invalid_argument("Task must be of type LSMFORLMTask.");
        }
        result = Eigen::Map<Eigen::VectorXd>(task->getValues().data(), task->getValues().size());
        currentError = task->getError();
    }

    void optimize() {
        int iteration = 0;
        Eigen::VectorXd residuals;
        Eigen::MatrixXd denseJacobian;

        while (iteration < maxIterations) {
            std::tie(residuals, denseJacobian) = task->linearizeFunction();

            if (!structureInitialized) {
                jacobian = denseJacobian.sparseView();
                jacobian.makeCompressed();
                structureInitialized = true;
            } else {
                for (int k = 0; k < jacobian.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(jacobian, k); it; ++it) {
                        it.valueRef() = denseJacobian.coeff(it.row(), it.col());
                    }
                }
            }

            Eigen::VectorXd gradient = jacobian.transpose() * residuals;
            double gradientNorm = gradient.norm();
            if (gradientNorm < epsilon1) {
                converged = true;
                break;
            }

            Eigen::SparseMatrix<double> hessian = jacobian.transpose() * jacobian;
            Eigen::SparseMatrix<double> identity(hessian.rows(), hessian.cols());
            identity.setIdentity();
            Eigen::SparseMatrix<double> dampedHessian = hessian + lambda * identity;

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(dampedHessian);
            if (solver.info() != Eigen::Success) {
                throw std::runtime_error("Failed to decompose damped Hessian.");
            }

            Eigen::VectorXd delta = solver.solve(-gradient);
            if (delta.norm() < epsilon2) {
                converged = true;
                break;
            }

            Eigen::VectorXd candidate = result + delta;
            std::vector<double> candidateVec(candidate.data(), candidate.data() + candidate.size());
            double candidateError = task->setError(candidateVec);

            double gainNumerator = currentError - candidateError;
            double gainDenominator = 0.5 * delta.dot(lambda * delta - gradient);
            double rho = gainNumerator / (gainDenominator + 1e-20);

            if (rho > 0) {
                result = candidate;
                currentError = candidateError;
                lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2 * rho - 1, 3));
                nu = 2.0;
            } else {
                lambda *= nu;
                nu *= 2.0;
            }

            ++iteration;
        }

        std::cout << "[LM] Finished after " << iteration << " iterations. "
                  << "Final error: " << currentError
                  << (converged ? " [converged]" : " [not converged]") << std::endl;
    }

    std::vector<double> getResult() const {
        return std::vector<double>(result.data(), result.data() + result.size());
    }

    bool isConverged() const {
        return converged;
    }

    double getCurrentError() const {
        return currentError;
    }
};

#endif // MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_LMWITHSPARSE_H_
