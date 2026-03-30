#ifndef MINIMIZEROPTIMIZER_HEADERS_DECOMPOSITION_SPARSEQR_H_
#define MINIMIZEROPTIMIZER_HEADERS_DECOMPOSITION_SPARSEQR_H_

#include "Matrix.h"
#include "SparseMatrix.h"
#include <vector>

class SparseQR {
private:
    SparseMatrix<> _A;
    Matrix<> _R;
    std::vector<size_t> _ref_ptr;
    std::vector<size_t> _ref_idx;
    std::vector<double> _ref_val;
    std::vector<double> _ref_beta;
    std::vector<size_t> _perm;
    std::vector<size_t> _perm_inv;

    Matrix<> applyQ(const Matrix<>& B) const;
    Matrix<> applyQt(const Matrix<>& B) const;
    Matrix<> solveUpperTriangular(const Matrix<>& rhs, double damping) const;

public:
    explicit SparseQR(const SparseMatrix<>& A);

    SparseQR(const SparseQR& other);
    SparseQR(SparseQR&& other) noexcept;

    SparseQR& operator=(const SparseQR& other);
    SparseQR& operator=(SparseQR&& other) noexcept;

    friend bool operator==(const SparseQR& A, const SparseQR& B);
    friend bool operator!=(const SparseQR& A, const SparseQR& B);

    void qr();
    Matrix<> solve(const Matrix<>& b, double damping = 1e-8) const;
    Matrix<> pseudoInverse(double damping = 1e-8) const;
    size_t rank(double tol = -1.0) const;

    SparseMatrix<> A() const;
    Matrix<> Q() const;
    Matrix<> R() const;
};

#endif // !MINIMIZEROPTIMIZER_HEADERS_DECOMPOSITION_SPARSEQR_H_
