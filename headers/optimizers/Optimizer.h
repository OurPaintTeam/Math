#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_OPTIMIZER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_OPTIMIZER_H_

#include "TaskF.h"
#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void optimize() = 0;

    virtual std::vector<double> getResult() const = 0;

    virtual bool isConverged() const = 0;

    virtual double getCurrentError() const = 0;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_OPTIMIZER_H_