#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_GRADIENTOPTIMIZER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_GRADIENTOPTIMIZER_H_

#include "Optimizer.h"
#include "TaskF.h"
#include <vector>

class GradientOptimizer : public Optimizer {
private:
    Task *task;
    std::vector<double> result;
    bool converged;
    double learningRate;
    int maxIterations;

public:
    GradientOptimizer(double lr = 0.01, int maxIter = 1000);

    void setTask(Task *tsk) override;

    void optimize() override;

    std::vector<double> getResult() const override;

    bool isConverged() const override;

    double getCurrentError() const override;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_GRADIENTOPTIMIZER_H_