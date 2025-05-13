// Created by Eugene Bychkov on 14.11.2024.

#include "LevenbergMarquardtSolver.h"

std::vector<double> LMSolver::getResult() const {
    return m_result;
}

void LMSolver::setTask(Task *task) {
    if (!task) {
        throw std::runtime_error("Task is null");
    }
    c_task = dynamic_cast<LSMTask *>(task);
    if (!c_task) {
        throw std::runtime_error("Task is not LSMTask");
    }
    m_result = c_task->getValues();
    currentError = c_task->getError();
}

double LMSolver::getCurrentError() const {
    return currentError;
}

bool LMSolver::isConverged() const {
    return converged;
}

void LMSolver::optimize() {
    int iteration = 0;
    Matrix<> gradient;
    Matrix<> hessian;

    while (iteration < maxIterations) {
        auto [residuals, jacobian] = c_task->linearizeFunction();
        gradient = jacobian.transpose() * residuals;
        if (gradient.norm() < epsilon1) {
            converged = true;
            break;
        }

        hessian = jacobian.transpose() * jacobian;
        Matrix<> dampedHessian = hessian + Matrix<>::identity(hessian.rows_size(), hessian.cols_size()) * lambda;
        QR dH(dampedHessian);
        dH.qr();
        Matrix<> delta = dH.pseudoInverse() * gradient;

        Matrix<> newResult = Matrix<>(m_result).transpose() - delta;
        std::vector<double> newParams(newResult.rows_size());
        for (std::size_t i = 0; i < newResult.rows_size(); ++i) {
            newParams[i] = newResult(i, 0);
        }

        double newError = c_task->setError(newParams);
        if (newError < currentError) {
            m_result = newParams;
            currentError = newError;
            lambda /= b_decrease;
        } else {
            lambda *= b_increase;
        }

        Matrix<> dParams = newResult - Matrix<>(m_result).transpose();
        if (gradient.norm() < epsilon1 && dParams.norm() < epsilon2) {
            converged = true;
            break;
        }

        ++iteration;
    }
    std::cout << "Levenberg-Marquardt converged after " << iteration << " iterations." << std::endl;
}
