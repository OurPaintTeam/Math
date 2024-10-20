//
// Created by Eugene Bychkov on 19.10.2024.
//

#ifndef MINIMIZEROPTIMIZER_GRADIENTMETHOD_H
#define MINIMIZEROPTIMIZER_GRADIENTMETHOD_H
#include "Optimizer.h"
class GradientMethod: public Optimizer{
public:
    GradientMethod(MinimizeTask& task);
    void optimize() override;
};


#endif //MINIMIZEROPTIMIZER_GRADIENTMETHOD_H
