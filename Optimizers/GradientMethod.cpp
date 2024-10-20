//
// Created by Eugene Bychkov on 19.10.2024.
//

#include "GradientMethod.h"
GradientMethod::GradientMethod(MinimizeTask& task): Optimizer(task){}

void GradientMethod::optimize(){
    double value = c_task.getError();
    double alpha = 0.001;
    Matrix<double> grad = c_task.getGradient();
    for (int i = 0; i < 100000 && value > epsl; ++i){
        grad = c_task.getGradient();
        std::vector<double> newParam = c_task.getParameters();
        for(int j = 0; j < newParam.size(); ++j){
            newParam[j] -= alpha * grad.getElement(0, j);
        }
        c_task.setParameters(newParam);
        value = c_task.getError();
    }
    if(value < epsl){
        for (const auto& i: c_task.getParameters()){
            std::cout << i<<std::endl;
        }
        isSolved= true;
    }
}
