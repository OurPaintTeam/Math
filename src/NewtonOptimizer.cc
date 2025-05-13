#include "NewtonOptimizer.h"

NewtonOptimizer::NewtonOptimizer(int maxItr): task(nullptr), converged(false), maxIterations(maxItr){}
void NewtonOptimizer::setTask(Task *newTask){
    this->task = newTask;
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
        for (std::size_t i = 0; i < grad.cols_size(); i++) {
            norm += grad(i, 0) * grad(i, 0);
        }
        norm = std::sqrt(norm);
        if (norm < 1e-6) {
            converged = true;
            break;
        }
        Matrix<> hess = task->hessian();
        QR qrH = hess;
        qrH.qr();
        Matrix<> HInv = qrH.pseudoInverse();
        Matrix<> step = (HInv + Matrix<>::identity(HInv.cols_size()) * 0.0001) * grad;
        for (std::size_t i = 0; i < result.size(); i++) {
            result[i] -= step(i, 0);
        }
        //double err = task->setError(result);
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