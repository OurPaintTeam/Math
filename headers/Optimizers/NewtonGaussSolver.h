//
// Created by Eugene Bychkov on 03.11.2024.
//

#ifndef MINIMIZEROPTIMIZER_NEWTONGAUSSESOLVER_H
#define MINIMIZEROPTIMIZER_NEWTONGAUSSESOLVER_H
#include "Optimizer.h"
#include "LSMTask.h"
class NewtonGaussSolver : public Optimizer {
    LSMTask *task;
    std::vector<double> result;
    bool converged;
    int maxIterations;
public:
    NewtonGaussSolver(int maxItr = 1000);

    void optimize() override;

    std::vector<double> getResult() const override;

    bool isConverged() const override;

    void setTask(Task *task) override;

    double getCurrentError() const override;
};

#endif //MINIMIZEROPTIMIZER_NEWTONGAUSSESOLVER_H
