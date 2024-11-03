//
// Created by Eugene Bychkov on 03.11.2024.
//

#include "NewtonGaussSolver.h"
#include <stdexcept>
#include <cmath>

NewtonGaussSolver::NewtonGaussSolver(int maxItr)
        : task(nullptr), converged(false), maxIterations(maxItr) {}

void NewtonGaussSolver::setTask(Task *task) {
    this->task = dynamic_cast<LSMTask*>(task);
    if (!this->task) {
        throw std::invalid_argument("Task must be of type LSMTask");
    }
    result = this->task->getValues();
}

void NewtonGaussSolver::optimize() {
    if (!task) {
        throw std::runtime_error("Task is not set");
    }

    const double epsilon = 1e-6;
    int iteration = 0;
    converged = false;
    while (iteration < maxIterations) {
        result = task->getValues();
        Matrix<> J = task->jacobian();
        Matrix<> JT = J.transpose();
        Matrix<> H = JT * J;
        Matrix<> g = task->gradient();
        Matrix<> delta = H.solveQR(g);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] -= delta(i, 0);
        }
        double delta_norm = 0.0;
        for (size_t i = 0; i < delta.row_size(); ++i) {
            delta_norm += delta(i, 0) * delta(i, 0);
        }
        delta_norm = std::sqrt(delta_norm);
        if (delta_norm < epsilon) {
            converged = true;
            break;
        }
        iteration++;
    }
}

std::vector<double> NewtonGaussSolver::getResult() const {
    return result;
}

bool NewtonGaussSolver::isConverged() const {
    return converged;
}

double NewtonGaussSolver::getCurrentError() const {
    if (!task) {
        throw std::runtime_error("Task is not set");
    }
    return task->getError();
}