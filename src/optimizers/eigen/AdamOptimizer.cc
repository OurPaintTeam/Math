#include "AdamOptimizer.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

AdamOptimizer::AdamOptimizer(double lr, double beta1, double beta2,
                             double eps, double convEps, int maxIter)
    : task(nullptr), converged(false), learningRate(lr), beta1(beta1), beta2(beta2),
      epsilon(eps), convergenceEps(convEps), maxIterations(maxIter) {}

void AdamOptimizer::setTask(TaskEigen* newTask) {
    this->task = dynamic_cast<LSMFORLMTask*>(newTask);
    if (!this->task) {
        throw std::invalid_argument("Task must be of type LSMFORLMTask.");
    }
    result = Eigen::Map<Eigen::VectorXd>(task->getValues().data(),
                                         task->getValues().size());
}

void AdamOptimizer::optimize() {
    if (!task) return;

    int n = static_cast<int>(result.size());
    Eigen::VectorXd m = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n);

    for (int iter = 1; iter <= maxIterations; ++iter) {
        Eigen::VectorXd grad = task->gradient();

        if (grad.norm() < convergenceEps) {
            converged = true;
            break;
        }

        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);

        Eigen::VectorXd mHat = m / (1.0 - std::pow(beta1, iter));
        Eigen::VectorXd vHat = v / (1.0 - std::pow(beta2, iter));

        result -= learningRate * mHat.cwiseQuotient(
            vHat.cwiseSqrt().array().operator+(epsilon).matrix());

        std::vector<double> params(result.data(), result.data() + result.size());
        task->setError(params);
    }
}

std::vector<double> AdamOptimizer::getResult() const {
    return std::vector<double>(result.data(), result.data() + result.size());
}

bool AdamOptimizer::isConverged() const {
    return converged;
}

double AdamOptimizer::getCurrentError() const {
    return task ? task->getError() : 0.0;
}
