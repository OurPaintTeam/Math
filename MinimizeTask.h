//
// Created by Eugene Bychkov on 19.10.2024.
//

#ifndef MINIMIZEROPTIMIZER_MINIMIZETASK_H
#define MINIMIZEROPTIMIZER_MINIMIZETASK_H
#include <vector>
#include "SimpleMath/Function.h"

class MinimizeTask {
    /* This class will be work only with function R->R */
    Function *c_function; //function that will be minimized
    bool f_isSolved;
    public:
    MinimizeTask(Function *function): c_function(function), f_isSolved(false){}
    Matrix<double> getGradient();
    Matrix<double> getJacobian();
    Matrix<double> getHessian();
    double getError();
    double getParameters();
    bool isSolved();
};


#endif //MINIMIZEROPTIMIZER_MINIMIZETASK_H
