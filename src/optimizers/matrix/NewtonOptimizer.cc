//
// Created by Eugene Bychkov on 03.11.2024.
//
#include "NewtonOptimizer.h"
#include "QR.h"
NewtonOptimizer::NewtonOptimizer(int maxItr): task(nullptr), converged(false), maxIterations(maxItr){}
void NewtonOptimizer::setTask(TaskMatrix *task){
    this->task = dynamic_cast<TaskMatrix*>(task);
    if (!this->task) {
        throw std::invalid_argument("Task must be of type TaskMatrix");
    }
    converged = false;
    result = this->task->getValues();
}
void NewtonOptimizer::optimize(){
    if(task == nullptr) return;

    converged = false;
    int itr = 0;
    while(itr < maxIterations) {
        result = task->getValues();
        Matrix<> grad = task->gradient();
        double norm = grad.norm();
        if (norm < 1e-6) {
            converged = true;
            break;
        }

        Matrix<> hess = task->hessian();
        Matrix<> regularizedHess =
            hess + Matrix<>::identity(hess.rows_size(), hess.cols_size()) * 1e-4;

        QR qrH(regularizedHess);
        qrH.qr();
        Matrix<> step = qrH.pseudoInverse() * grad;

        for (std::size_t i = 0; i < result.size(); i++) {
            result[i] -= step(i, 0);
        }

        task->setError(result);
        if (step.norm() < 1e-6) {
            converged = true;
            break;
        }

        itr++;
    }
    std::cout << "NewtonOptimizer: " << itr << " iterations" << std::endl;
}
bool NewtonOptimizer::isConverged() const {
    return converged;
}
std::vector<double> NewtonOptimizer::getResult() const {
    return result;
}
double NewtonOptimizer::getCurrentError() const {
    return task->getError();
}