//
// Created by Eugene Bychkov on 14.11.2024.
//
#include "LevenbergMarquardtSolver.h"

std::vector<double> LMSolver::getResult() const {
    return m_result;
}

void LMSolver::setTask(Task *task) {
    if (task == nullptr) {
        throw std::runtime_error("Task is null");
    }
    c_task = dynamic_cast<LSMTask *>(task);
    if (c_task == nullptr) {
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
        double gradientNorm = 0;
        for (int i = 0; i < gradient.rows_size(); ++i)
            gradientNorm += gradient(i, 0) * gradient(i, 0);
        gradientNorm = std::sqrt(gradientNorm);
        if (gradientNorm < epsilon1) {
            converged = true;
            break;
        }
        hessian = jacobian.transpose() * jacobian;

        Matrix<> dampedHessian = hessian + Matrix<>::identity(hessian.rows_size(), hessian.cols_size()) * lambda;
        QR dH = QR(dampedHessian);
        dH.qr();
        Matrix<> delta = dH.pseudoInverse() * gradient;
        Matrix<> newResult = Matrix<>(m_result).transpose() - delta;
        std::vector<double> newParams;
        for (int i = 0; i < newResult.rows_size(); ++i) {
            newParams.push_back(newResult(i, 0));
        }
        double newError = c_task->setError(newParams);
        if (newError < currentError) {
            m_result = newParams;
            currentError = newError;
            lambda /= b_decrease;
        } else {
            lambda *= b_increase;
        }
        gradientNorm = 0;
        for (int i = 0; i < gradient.rows_size(); ++i)
            gradientNorm += gradient(i, 0) * gradient(i, 0);
        gradientNorm = std::sqrt(gradientNorm);
        Matrix<> dParams = newResult - Matrix<>(m_result).transpose();
        double dParamsNorm = 0;
        for (int i = 0; i < dParams.rows_size(); ++i)
            dParamsNorm += dParams(i, 0) * dParams(i, 0);
        dParamsNorm = std::sqrt(dParamsNorm);
        if (gradientNorm < epsilon1 && dParamsNorm < epsilon2) {
            converged = true;
            break;
        }

        ++iteration;
    }
    std::cout << "Levenberg-Marquardt converged after " << iteration << " iterations." << std::endl;
}

