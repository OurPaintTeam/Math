#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_ADAMOPTIMIZER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_ADAMOPTIMIZER_H_

#include "EigenOptimizer.h"
#include "LSMFORLMTask.h"
#include <Eigen/Dense>
#include <vector>

class AdamOptimizer : public EigenOptimizer {
private:
    LSMFORLMTask* task;
    Eigen::VectorXd result;
    bool converged;
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    double convergenceEps;
    int maxIterations;

public:
    AdamOptimizer(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
                  double eps = 1e-8, double convEps = 1e-6, int maxIter = 1000);

    void setTask(TaskEigen* newTask) override;
    void optimize() override;
    std::vector<double> getResult() const override;
    bool isConverged() const override;
    double getCurrentError() const override;
};

#endif
