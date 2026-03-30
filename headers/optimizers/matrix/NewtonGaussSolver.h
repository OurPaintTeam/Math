#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_NEWTONGAUSSESOLVER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_NEWTONGAUSSESOLVER_H_

#include "MatrixOptimizer.h"
#include "LSMTask.h"

class NewtonGaussSolver : public MatrixOptimizer {
    LSMTask *task;
    std::vector<double> result;
    bool converged;
    int maxIterations;
public:
    NewtonGaussSolver(int maxItr = 1000);

    void optimize() override;

    std::vector<double> getResult() const override;

    bool isConverged() const override;

    void setTask(TaskMatrix *task) override;

    double getCurrentError() const override;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_NEWTONGAUSSESOLVER_H_
