#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_NEWTONGAUSSESOLVER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_NEWTONGAUSSESOLVER_H_

#include <stdexcept>
#include <cmath>

#include "Optimizer.h"
#include "LSMTask.h"
#include "QR.h"

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

#endif // ! MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_NEWTONGAUSSESOLVER_H_
