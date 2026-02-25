#include "StochasticGradientOptimizer.h"
#include <stdexcept>

StochasticGradientOptimizer::StochasticGradientOptimizer(double lr, int maxIter,
                                                         double eps, unsigned seed)
    : task(nullptr), converged(false), learningRate(lr), epsilon(eps),
      maxIterations(maxIter), rng(seed) {}

void StochasticGradientOptimizer::setTask(TaskEigen* newTask) {
    this->task = dynamic_cast<LSMFORLMTask*>(newTask);
    if (!this->task) {
        throw std::invalid_argument("Task must be of type LSMFORLMTask.");
    }
    result = Eigen::Map<Eigen::VectorXd>(task->getValues().data(),
                                         task->getValues().size());
}

void StochasticGradientOptimizer::optimize() {
    if (!task) return;

    int n = static_cast<int>(result.size());
    std::uniform_int_distribution<int> dist(0, n - 1);

    for (int iter = 0; iter < maxIterations; ++iter) {
        Eigen::VectorXd grad = task->gradient();

        if (grad.norm() < epsilon) {
            converged = true;
            break;
        }

        int idx = dist(rng);

        Eigen::VectorXd stochasticGrad = Eigen::VectorXd::Zero(n);
        stochasticGrad(idx) = grad(idx) * n;

        result -= learningRate * stochasticGrad;

        std::vector<double> params(result.data(), result.data() + result.size());
        task->setError(params);
    }
}

std::vector<double> StochasticGradientOptimizer::getResult() const {
    return std::vector<double>(result.data(), result.data() + result.size());
}

bool StochasticGradientOptimizer::isConverged() const {
    return converged;
}

double StochasticGradientOptimizer::getCurrentError() const {
    return task ? task->getError() : 0.0;
}
