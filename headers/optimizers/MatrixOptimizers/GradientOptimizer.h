#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_GRADIENTOPTIMIZER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_GRADIENTOPTIMIZER_H_

#include "MatrixOptimizer.h"
#include <vector>

class GradientOptimizer : public MatrixOptimizer {
private:
    TaskMatrix *task;
    std::vector<double> result;
    bool converged;
    double learningRate;
    int maxIterations;

public:
    GradientOptimizer(double lr = 0.01, int maxIter = 1000);

    void setTask(TaskMatrix *tsk) override;

    void optimize() override;

    std::vector<double> getResult() const override;

    bool isConverged() const override;

    double getCurrentError() const override;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_GRADIENTOPTIMIZER_H_