#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "Matrix.h"
#include "SparseMatrix.h"
#include "SparseQR.h"

namespace {

bool DenseApproxEqual(const Matrix<double>& A, const Matrix<double>& B, double eps = 1e-9) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) {
        return false;
    }

    for (size_t i = 0; i < A.rows_size(); ++i) {
        for (size_t j = 0; j < A.cols_size(); ++j) {
            if (std::abs(A(i, j) - B(i, j)) > eps) {
                return false;
            }
        }
    }

    return true;
}

bool IsUpperTriangular(const Matrix<double>& A, double eps = 1e-9) {
    for (size_t i = 0; i < A.rows_size(); ++i) {
        for (size_t j = 0; j < A.cols_size() && j < i; ++j) {
            if (std::abs(A(i, j)) > eps) {
                return false;
            }
        }
    }
    return true;
}

bool IsOrthonormal(const Matrix<double>& Q, double eps = 1e-9) {
    Matrix<double> qtq = Q.transpose() * Q;
    Matrix<double> identity = Matrix<double>::identity(qtq.rows_size(), qtq.cols_size());
    return DenseApproxEqual(qtq, identity, eps);
}

Matrix<double> BuildSparseLikeDense(size_t rows, size_t cols, double density, uint32_t seed) {
    Matrix<double> result(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> val(-5.0, 5.0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (prob(gen) < density) {
                result(i, j) = val(gen);
            }
        }
    }

    return result;
}

void CheckFactorization(const Matrix<double>& dense, double eps = 1e-8) {
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();

    Matrix<double> restored = qr.Q() * qr.R();

    EXPECT_TRUE(DenseApproxEqual(restored, dense, eps));
    EXPECT_TRUE(IsOrthonormal(qr.Q(), eps));
    EXPECT_TRUE(IsUpperTriangular(qr.R(), eps));
}

} // namespace

TEST(SparseQRTests, matrixConstructor) {
    SparseMatrix<> A(3, 3);
    A.insert(0, 0, 1.0);
    A.insert(1, 1, 2.0);

    SparseQR qr(A);
    EXPECT_EQ(qr.A(), A);
    EXPECT_EQ(qr.Q(), Matrix<>());
    EXPECT_EQ(qr.R(), Matrix<>());

    SparseMatrix<> empty;
    EXPECT_THROW(SparseQR bad(empty), std::runtime_error);
}

TEST(SparseQRTests, rectangularSparseMatrixDecomposition) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 0.0},
        {4.0, 0.0, 5.0},
        {0.0, 6.0, 0.0}
    };
    CheckFactorization(dense, 1e-8);
}

TEST(SparseQRTests, squareSparseMatrixDecomposition) {
    Matrix<double> dense = {
        {2.0, 0.0, -2.0, 0.0},
        {0.0, 5.0, 0.0, 1.0},
        {-2.0, 0.0, 3.0, 0.0},
        {0.0, 1.0, 0.0, 4.0}
    };
    CheckFactorization(dense, 1e-8);
}

TEST(SparseQRTests, rankDeficientMatrixDecomposition) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0, 1.0},
        {2.0, 0.0, 4.0, 2.0},
        {0.0, 3.0, 0.0, 0.0},
        {0.0, 6.0, 0.0, 0.0}
    };

    CheckFactorization(dense, 1e-8);
}

TEST(SparseQRTests, wideSparseMatrixDecomposition) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0, 0.0, 3.0},
        {0.0, 4.0, 0.0, 5.0, 0.0},
        {6.0, 0.0, 7.0, 0.0, 8.0}
    };

    CheckFactorization(dense, 1e-8);
}

TEST(SparseQRTests, matrixWithZeroColumnsDecomposition) {
    Matrix<double> dense = {
        {0.0, 1.0, 0.0, 2.0},
        {0.0, 0.0, 0.0, 3.0},
        {0.0, 4.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 5.0}
    };

    CheckFactorization(dense, 1e-8);
}

TEST(SparseQRTests, diagonalMatrixDecomposition) {
    Matrix<double> dense = {
        {5.0, 0.0, 0.0, 0.0},
        {0.0, -2.0, 0.0, 0.0},
        {0.0, 0.0, 7.0, 0.0},
        {0.0, 0.0, 0.0, 3.0}
    };

    CheckFactorization(dense, 1e-8);
}

TEST(SparseQRTests, randomSparseMatricesDecomposition) {
    const std::vector<uint32_t> seeds = {7u, 42u, 1337u, 2026u};

    for (uint32_t seed : seeds) {
        Matrix<double> dense = BuildSparseLikeDense(7, 5, 0.28, seed);
        CheckFactorization(dense, 1e-7);
    }
}
