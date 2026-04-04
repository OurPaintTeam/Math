#ifndef MINIMIZEROPTIMIZER_HEADERS_DECOMPOSITION_SPARSEQR_H_
#define MINIMIZEROPTIMIZER_HEADERS_DECOMPOSITION_SPARSEQR_H_

#include "Matrix.h"
#include "SparseMatrix.h"
#include <vector>

class SparseQR {
private:
    SparseMatrix<> _A;
    SparseMatrix<> _R_sparse;
    size_t _m = 0;
    size_t _n = 0;
    size_t _min_mn = 0;
    double _pivot_threshold = 0.0;
    double _factorization_threshold = 0.0;
    size_t _numerical_rank = 0;
    size_t _leading_numerical_rank = 0;
    std::vector<size_t> _ref_ptr;
    std::vector<size_t> _ref_idx;
    std::vector<double> _ref_val;
    std::vector<double> _ref_beta;
    std::vector<size_t> _col_counts;
    std::vector<size_t> _zero_cols;
    std::vector<size_t> _active_cols;
    std::vector<size_t> _perm;
    std::vector<size_t> _perm_inv;
    std::vector<size_t> _workspace_outer;
    std::vector<size_t> _workspace_inner;
    std::vector<double> _workspace_val;
    std::vector<size_t> _etree;
    std::vector<size_t> _first_row_elt;
    bool _use_default_threshold = true;
    bool _preprocessed = false;
    bool _analyzed = false;

    double computeDefaultThreshold() const;
    double resolvePivotThreshold() const;
    void preprocessProblem();
    void analyzeStructure();
    void buildPermutedWorkspace();
    void buildColumnEliminationTree();
    void factorizeNumeric();
    SparseMatrix<> buildSparseR(const std::vector<std::vector<size_t>>& column_rows,
                                const std::vector<std::vector<double>>& column_values) const;
    Matrix<> applyQ(const Matrix<>& B) const;
    Matrix<> applyQt(const Matrix<>& B) const;
    Matrix<> solveUpperTriangular(const Matrix<>& rhs, size_t effective_rank) const;

public:
    explicit SparseQR(const SparseMatrix<>& A);

    SparseQR(const SparseQR& other);
    SparseQR(SparseQR&& other) noexcept;

    SparseQR& operator=(const SparseQR& other);
    SparseQR& operator=(SparseQR&& other) noexcept;

    friend bool operator==(const SparseQR& A, const SparseQR& B);
    friend bool operator!=(const SparseQR& A, const SparseQR& B);

    void qr();
    void analyze();
    void factorize();
    void setPivotThreshold(double threshold);
    Matrix<> solve(const Matrix<>& b, double damping = 1e-8) const;
    Matrix<> pseudoInverse(double damping = 1e-8) const;
    size_t rank(double tol = -1.0) const;

    SparseMatrix<> A() const;
    Matrix<> Q() const;
    Matrix<> R() const;
};

#endif // !MINIMIZEROPTIMIZER_HEADERS_DECOMPOSITION_SPARSEQR_H_
