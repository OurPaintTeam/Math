#include "GradientOptimizer.h"

GradientOptimizer::GradientOptimizer(double lr, int maxIter)
        : task(nullptr), converged(false), learningRate(lr), maxIterations(maxIter) {}

void GradientOptimizer::setTask(Task* tsk) {
    this->task = tsk;
    if (task) {
        result = task->getValues();
    }
}

void GradientOptimizer::optimize() {
    if (!task) return;

    for (int iter = 0; iter < maxIterations; ++iter) {
        Matrix<> grad = task->gradient();

        for (size_t i = 0; i < grad.row_size(); ++i) {
            result[i] -= learningRate * grad(i, 0);
        }
        double update = task->setError(result);

        if (update < 1e-6) {
            converged = true;
            break;
        }
    }
}

std::vector<double> GradientOptimizer::getResult() const {
    return result;
}

bool GradientOptimizer::isConverged() const {
    return converged;
}

double GradientOptimizer::getCurrentError() const {
    return task ? task->getError() : 0.0;
}