//
// Created by Eugene Bychkov on 12.11.2024.
//

#ifndef MINIMIZEROPTIMIZER_LMFORTEST_H
#define MINIMIZEROPTIMIZER_LMFORTEST_H

#include "Optimizer.h"
#include "LSMFORLMTask.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>

class LevenbergMarquardtSolver {
private:
    LSMFORLMTask* task;
    Eigen::VectorXd result;
    bool converged;
    double currentError;
    double lambda;
    double b_increase;
    double b_decrease;
    double epsilon1;
    double epsilon2;
    int maxIterations;

public:
    LevenbergMarquardtSolver(int maxIterations = 100, double initLambda = 0.1, double b_increase = 2.0, double b_decrease = 2.0,
                             double epsilon1 = 1e-6, double epsilon2 = 1e-6)
            : lambda(initLambda), b_increase(b_increase), b_decrease(b_decrease), epsilon1(epsilon1), epsilon2(epsilon2),
              maxIterations(maxIterations), converged(false) {}

    void setTask(LSMFORLMTask* task) {
        this->task = dynamic_cast<LSMFORLMTask*>(task);
        if (!this->task) {
            throw std::invalid_argument("Task is not of type LSMFORLMTask.");
        }
        result = Eigen::Map<Eigen::VectorXd>(task->getValues().data(), task->getValues().size());;
        currentError = task->getError();
    }

    void optimize() {
        int iteration = 0;
        Eigen::VectorXd gradient;
        Eigen::MatrixXd hessian;
        Eigen::MatrixXd jacobian;
        Eigen::VectorXd residuals;

        while (iteration < maxIterations) {
            auto [residuals, jacobian] = task->linearizeFunction();

            gradient = jacobian.transpose() * residuals;
            double gradientNorm = gradient.norm();
            if (gradientNorm < epsilon1) {
                converged = true;
                break;
            }
            hessian = jacobian.transpose() * jacobian;

            Eigen::MatrixXd dampedHessian = hessian + lambda * Eigen::MatrixXd::Identity(hessian.rows(), hessian.cols());
            Eigen::VectorXd delta = dampedHessian.fullPivHouseholderQr().solve(gradient);
            if (delta.norm() < epsilon1){
                converged = true;
                break;
            }
            Eigen::VectorXd newResult = result - delta;
            std::vector<double> newParams(newResult.data(), newResult.data() + newResult.size());
            double newError = task->setError(newParams);
            if (newError < currentError) {
                result = newResult;
                currentError = newError;
                lambda /= b_decrease;
            } else {
                lambda *= b_increase;
            }
            if (gradient.norm() < epsilon1 && (newResult - result).norm() < epsilon2) {
                converged = true;
                break;
            }

            ++iteration;
        }
        std::cout << "Levenberg-Marquardt converged after " << iteration << " iterations." << std::endl;
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

#endif //MINIMIZEROPTIMIZER_LMFORTEST_H
