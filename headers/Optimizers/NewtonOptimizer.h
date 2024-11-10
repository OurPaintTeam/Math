//
// Created by Eugene Bychkov on 03.11.2024.
//

#ifndef MINIMIZEROPTIMIZER_NEWTONOPTIMIZER_H
#define MINIMIZEROPTIMIZER_NEWTONOPTIMIZER_H

#include "Optimizer.h"
#include "../decomposition/QR.h"
class NewtonOptimizer : public Optimizer {
    Task *task;
    std::vector<double> result;
    bool converged;
    int maxIterations;
public:
    NewtonOptimizer(int maxItr = 1000);

    void optimize() override;

    std::vector<double> getResult() const override;

    bool isConverged() const override;

    void setTask(Task *task) override;

    double getCurrentError() const override;
};

#endif //MINIMIZEROPTIMIZER_NEWTONOPTIMIZER_H
