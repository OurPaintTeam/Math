#ifndef MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSEDOGLEGSOLVER_H_
#define MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSEDOGLEGSOLVER_H_

#include "MatrixOptimizer.h"
#include "SparseLSMTask.h"
#include "SparseQR.h"

#include <memory>
#include <vector>

class SparseDogLegSolver : public MatrixOptimizer {
private:
    SparseLSMTask* c_task;
    std::vector<double> m_result;
    SparseMatrix<> m_normalMatrix;
    std::unique_ptr<SparseQR> m_linearSolver;
    bool converged;
    double currentError;
    double initialTrustRegionRadius;
    double trustRegionRadius;
    double maxTrustRegionRadius;
    double eta;
    double epsilon1;
    double epsilon2;
    double errorTolerance;
    double linearSolverDamping;
    int maxIterations;
    int performedIterations;

    Matrix<> buildDogLegStep(const Matrix<>& gradient, const Matrix<>& gaussNewtonStep) const;

public:
    SparseDogLegSolver(int maxIterations = 100,
                       double initialTrustRegionRadius = 1.0,
                       double maxTrustRegionRadius = 100.0,
                       double eta = 1e-4,
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

#endif // MINIMIZEROPTIMIZER_HEADERS_OPTIMIZERS_MATRIX_SPARSE_SPARSEDOGLEGSOLVER_H_
