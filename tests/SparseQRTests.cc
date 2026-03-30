#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

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

std::vector<size_t> ColumnOrdering(const Matrix<double>& dense) {
    std::vector<size_t> counts(dense.cols_size(), 0);
    for (size_t i = 0; i < dense.rows_size(); ++i) {
        for (size_t j = 0; j < dense.cols_size(); ++j) {
            if (dense(i, j) != 0.0) {
                ++counts[j];
            }
        }
    }
    std::vector<size_t> perm(dense.cols_size());
    for (size_t j = 0; j < dense.cols_size(); ++j) {
        perm[j] = j;
    }
    std::sort(perm.begin(), perm.end(), [&counts](size_t a, size_t b) {
        if (counts[a] != counts[b]) {
            return counts[a] < counts[b];
        }
        return a < b;
    });
    return perm;
}

Matrix<double> ApplyColumnOrdering(const Matrix<double>& dense) {
    std::vector<size_t> perm = ColumnOrdering(dense);
    Matrix<double> ordered(dense.rows_size(), dense.cols_size());
    for (size_t j = 0; j < dense.cols_size(); ++j) {
        for (size_t i = 0; i < dense.rows_size(); ++i) {
            ordered(i, j) = dense(i, perm[j]);
        }
    }
    return ordered;
}

double ResidualNorm(const Matrix<double>& A, const Matrix<double>& x, const Matrix<double>& b) {
    Matrix<double> r = A * x - b;
    return r.norm();
}

Eigen::SparseMatrix<double> ToEigenSparse(const SparseMatrix<double>& matrix) {
    Eigen::SparseMatrix<double> result(
        static_cast<int>(matrix.rows_size()),
        static_cast<int>(matrix.cols_size()));
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(matrix.nonZeros());
    for (size_t row = 0; row < matrix.rows_size(); ++row) {
        for (SparseMatrix<double>::InnerIterator it(matrix, row); it; ++it) {
            triplets.emplace_back(
                static_cast<int>(it.row()),
                static_cast<int>(it.col()),
                it.value());
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

Eigen::MatrixXd EigenFromDense(const Matrix<double>& matrix) {
    Eigen::MatrixXd result(
        static_cast<int>(matrix.rows_size()),
        static_cast<int>(matrix.cols_size()));
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result(i, j) = matrix(static_cast<size_t>(i), static_cast<size_t>(j));
        }
    }
    return result;
}

Matrix<double> DenseFromEigen(const Eigen::MatrixXd& matrix) {
    Matrix<double> result(static_cast<size_t>(matrix.rows()), static_cast<size_t>(matrix.cols()));
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            result(static_cast<size_t>(i), static_cast<size_t>(j)) = matrix(i, j);
        }
    }
    return result;
}

void CheckFactorization(const Matrix<double>& dense, double eps = 1e-8) {
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();

    Matrix<double> restored = qr.Q() * qr.R();
    Matrix<double> ordered = ApplyColumnOrdering(dense);

    EXPECT_TRUE(DenseApproxEqual(restored, ordered, eps));
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

TEST(SparseQRTests, solveLeastSquaresComparedToEigen) {
    Matrix<double> dense = BuildSparseLikeDense(120, 80, 0.06, 77u);
    SparseMatrix<double> sparse(dense);
    Matrix<double> b(120, 2);
    for (size_t i = 0; i < b.rows_size(); ++i) {
        b(i, 0) = std::sin(static_cast<double>(i) * 0.1);
        b(i, 1) = std::cos(static_cast<double>(i) * 0.2);
    }

    SparseQR qr(sparse);
    qr.qr();
    Matrix<double> xSparse = qr.solve(b);
    double sparseResidual = ResidualNorm(dense, xSparse, b);

    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparse);
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);
    Matrix<double> xEigen = DenseFromEigen(eigenQr.solve(EigenFromDense(b)));
    double eigenResidual = ResidualNorm(dense, xEigen, b);

    EXPECT_TRUE(std::isfinite(sparseResidual));
    EXPECT_TRUE(std::isfinite(eigenResidual));
    EXPECT_LE(sparseResidual, eigenResidual * 100.0 + 1e-6);
}

TEST(SparseQRTests, rankEstimationForDependentColumns) {
    Matrix<double> dense = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 6.0},
        {0.0, 1.0, 1.0},
        {0.0, 2.0, 2.0}
    };
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();
    EXPECT_EQ(qr.rank(), 2u);
}
