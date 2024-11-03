//
// Created by Eugene Bychkov on 20.10.2024.
//

#ifndef MINIMIZEROPTIMIZER_OPTIMIZER_H
#define MINIMIZEROPTIMIZER_OPTIMIZER_H

#include "Task.h"
#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void setTask(Task* task) = 0;

    virtual void optimize() = 0;

    virtual std::vector<double> getResult() const = 0;

    virtual bool isConverged() const = 0;

    virtual double getCurrentError() const = 0;
};

#endif //MINIMIZEROPTIMIZER_OPTIMIZER_H