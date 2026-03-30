#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_STAHATICGRADIENTOPTIMIZER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_STAHATICGRADIENTOPTIMIZER_H_

#include "EigenOptimizer.h"
#include "LSMFORLMTask.h"
#include <Eigen/Dense>
#include <vector>
#include <random>

class StochasticGradientOptimizer : public EigenOptimizer {
private:
    LSMFORLMTask* task;
    Eigen::VectorXd result;
    bool converged;
    double learningRate;
    double epsilon;
    int maxIterations;
    std::mt19937 rng;

public:
    StochasticGradientOptimizer(double lr = 0.01, int maxIter = 1000,
                                double eps = 1e-6, unsigned seed = 42);

    void setTask(TaskEigen* newTask) override;
    void optimize() override;
    std::vector<double> getResult() const override;
    bool isConverged() const override;
    double getCurrentError() const override;
};

#endif
