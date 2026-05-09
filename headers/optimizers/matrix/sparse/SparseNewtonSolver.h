#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSENEWTONSOLVER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSENEWTONSOLVER_H_

#include "MatrixOptimizer.h"
#include "SparseQR.h"
#include "SparseSystemTask.h"

#include <memory>
#include <vector>

class SparseNewtonSolver : public MatrixOptimizer {
private:
    SparseSystemTask* c_task;
    std::vector<double> m_result;
    SparseMatrix<> m_jacobian;
    std::unique_ptr<SparseQR> m_linearSolver;
    bool converged;
    double currentError;
    double epsilon1;
    double epsilon2;
    double errorTolerance;
    double linearSolverDamping;
    int maxIterations;
    int performedIterations;

public:
    SparseNewtonSolver(int maxIterations = 100,
                       double epsilon1 = 1e-8,
                       double epsilon2 = 1e-8,
                       double errorTolerance = 1e-8,
                       double linearSolverDamping = 1e-8);

    void setTask(TaskMatrix* task) override;

    void optimize() override;

    std::vector<double> getResult() const override;

    double getCurrentError() const override;

    bool isConverged() const override;

    int getIterationCount() const;
};

#endif // MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSENEWTONSOLVER_H_
