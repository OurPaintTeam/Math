#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSELEVENBERGMARQUARDTSOLVER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSELEVENBERGMARQUARDTSOLVER_H_

#include "MatrixOptimizer.h"
#include "SparseLSMTask.h"
#include <vector>

class SparseLMSolver : public MatrixOptimizer {
private:
    SparseLSMTask* c_task;
    std::vector<double> m_result;
    bool converged;
    double currentError;
    double initialLambda;
    double lambda;
    double nu;
    double epsilon1;
    double epsilon2;
    double errorTolerance;
    int maxIterations;
    int performedIterations;

public:
    SparseLMSolver(int maxIterations = 100,
                   double initLambda = 1e-3,
                   double epsilon1 = 1e-8,
                   double epsilon2 = 1e-8,
                   double errorTolerance = 1e-8);

    void setTask(TaskMatrix* task) override;

    void optimize() override;

    std::vector<double> getResult() const override;

    double getCurrentError() const override;

    bool isConverged() const override;

    int getIterationCount() const;
};

#endif // MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSELEVENBERGMARQUARDTSOLVER_H_
