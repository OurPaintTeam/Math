#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_LEVENBERGMARQUARDTSOLVER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_LEVENBERGMARQUARDTSOLVER_H_

#include "Optimizer.h"
#include "LSMTask.h"
#include "QR.h"
#include <vector>
#include <cmath>
#include <stdexcept>

class LMSolver : public Optimizer {
private:
    LSMTask *c_task;
    std::vector<double> m_result;
    bool converged;
    double currentError;
    double lambda;
    double b_increase;
    double b_decrease;
    double epsilon1;
    double epsilon2;
    int maxIterations;

public:
    LMSolver(double initLambda = 1.0, double b_increase = 2.0, double b_decrease = 2.0,
             double epsilon1 = 1e-6, double epsilon2 = 1e-6, int maxIterations = 100)
            : lambda(initLambda), b_increase(b_increase), b_decrease(b_decrease), epsilon1(epsilon1),
              epsilon2(epsilon2),
              maxIterations(maxIterations), converged(false) {}

    void setTask(Task *task) override;

    void optimize() override;

    std::vector<double> getResult() const override;

    double getCurrentError() const override;

    bool isConverged() const override;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_LEVENBERGMARQUARDTSOLVER_H_
