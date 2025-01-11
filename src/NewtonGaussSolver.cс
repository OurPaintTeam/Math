#include "NewtonGaussSolver.h"

NewtonGaussSolver::NewtonGaussSolver(int maxItr)
        : task(nullptr), converged(false), maxIterations(maxItr) {}

void NewtonGaussSolver::setTask(Task *task) {
    this->task = dynamic_cast<LSMTask*>(task);
    if (!this->task) {
        throw std::invalid_argument("Task must be of type LSMTask");
    }
    result = this->task->getValues();
}

// inverse method substitutions for upper-triangle matrix R and vector y
Matrix<> backSubstitution(const Matrix<> &R, const Matrix<> &y) {
    size_t n = R.rows_size();
    Matrix<> x(n, 1);
    // Go from last to first
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = y(i, 0);
        for (size_t j = i + 1; j < n; ++j) {
            sum -= R(i, j) * x(j, 0);
        }
        if (std::fabs(R(i, i)) < 1e-14) {
            // Singular matrix, can set x(i,0) = 0 or throw
            x(i, 0) = 0.0;
        } else {
            x(i, 0) = sum / R(i, i);
        }
    }
    return x;
}

void NewtonGaussSolver::optimize() {
    if (!task) {
        throw std::runtime_error("Task is not set");
    }

    const double epsilon = 1e-6;
    int iteration = 0;
    converged = false;

    while (iteration < maxIterations) {
        auto [residuals, J] = task->linearizeFunction();

        result = task->getValues();

        Matrix<> JT = J.transpose();
        Matrix<> H = JT * J;
        Matrix<> g = JT * residuals;

        QR qrSolver(H);
        qrSolver.qr();
        Matrix<> Q = qrSolver.Q();  // n x n
        Matrix<> R = qrSolver.R();  // n x n

        Matrix<> Qt = Q.transpose();
        Matrix<> y = Qt * g;        // n x 1

        Matrix<> delta = backSubstitution(R, y);

        for (size_t i = 0; i < result.size(); ++i) {
            result[i] -= delta(i, 0);
        }

        task->setError(result);

        double currentError = task->getError();
        // std::cout << currentError << '\n';

        double delta_norm = 0.0;
        for (size_t i = 0; i < delta.rows_size(); ++i) {
            delta_norm += delta(i, 0) * delta(i, 0);
        }
        delta_norm = std::sqrt(delta_norm);

        if (delta_norm < epsilon) {
            converged = true;
            break;
        }
        ++iteration;
    }

    std::cout << "Newton-Gauss converged after " << iteration << " iterations." << std::endl;
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
