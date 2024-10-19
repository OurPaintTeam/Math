//
// Created by Eugene Bychkov on 19.10.2024.
//

#include "MinimizeTask.h"

Matrix<double> MinimizeTask::getGradient() {
    Matrix<double> gradient(1, c_function->getNumberOfVariables());
    for (int i = 0; i < c_function->getNumberOfVariables(); i++) {
        gradient.setElement(0, i, c_function->getDerivative(i));
    }
    return gradient;
}

Matrix<double> MinimizeTask::getHessian() {
    Matrix<double> hessian(c_function->getNumberOfVariables(), c_function->getNumberOfVariables());
    for (int i = 0; i < c_function->getNumberOfVariables(); i++) {
        for (int j = 0; j < c_function->getNumberOfVariables(); j++) {
            hessian.setElement(i, j, c_function->getMixedDerivative(i, j));
        }
    }
    return hessian;
}
