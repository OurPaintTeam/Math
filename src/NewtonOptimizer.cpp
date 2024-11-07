//
// Created by Eugene Bychkov on 03.11.2024.
//
#include "NewtonOptimizer.h"

NewtonOptimizer::NewtonOptimizer(int maxItr): maxIterations(maxItr), task(nullptr), converged(false){}
void NewtonOptimizer::setTask(Task *task){
    this->task = task;
    result = task->getValues();
}
void NewtonOptimizer::optimize(){
    if(task == nullptr) return;
    int itr = 0;
    while(itr < maxIterations) {
        itr++;
        result = task->getValues();
        Matrix<> grad = task->gradient();
        double norm = 0;
        for (int i = 0; i < grad.cols_size(); i++) {
            norm += grad(i, 0) * grad(i, 0);
        }
        norm = std::sqrt(norm);
        if (norm < 1e-6) {
            converged = true;
            break;
        }
        Matrix<> hess = task->hessian();
        Matrix<> step = hess.inverse() * grad;
        for (int i = 0; i < result.size(); i++) {
            result[i] -= step(i, 0);
        }
        double err = task->setError(result);
    }
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