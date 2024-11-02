//
// Created by Eugene Bychkov on 25.10.2024.
//

#ifndef MINIMIZEROPTIMIZER_GRADIENTOPTIMIZER_H
#define MINIMIZEROPTIMIZER_GRADIENTOPTIMIZER_H

#include "Optimizer.h"
#include "Task.h"
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

#endif //MINIMIZEROPTIMIZER_GRADIENTOPTIMIZER_H