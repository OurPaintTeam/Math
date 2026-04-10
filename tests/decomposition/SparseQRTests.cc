#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "Matrix.h"
#include "SparseMatrix.h"
#include "SparseQR.h"

namespace {

constexpr size_t kMaxRowNnzForGeometry = 12;

void EnforceRowNnzCap(Matrix<double>& matrix, size_t max_row_nnz = kMaxRowNnzForGeometry) {
    for (size_t i = 0; i < matrix.rows_size(); ++i) {
        std::vector<std::pair<double, size_t>> nz;
        nz.reserve(matrix.cols_size());
        for (size_t j = 0; j < matrix.cols_size(); ++j) {
            const double v = matrix(i, j);
            if (v != 0.0) {
                nz.emplace_back(std::abs(v), j);
            }
        }
        if (nz.size() <= max_row_nnz) continue;
        std::nth_element(
            nz.begin(),
            nz.begin() + static_cast<std::ptrdiff_t>(max_row_nnz),
            nz.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<char> keep(matrix.cols_size(), 0);
        for (size_t k = 0; k < max_row_nnz; ++k) {
            keep[nz[k].second] = 1;
        }
        for (size_t j = 0; j < matrix.cols_size(); ++j) {
            if (!keep[j]) matrix(i, j) = 0.0;
        }
    }
}

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

    EnforceRowNnzCap(result);
    return result;
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

    Matrix<double> qr_product = qr.Q() * qr.R();
    const auto& perm = qr.colsPermutation();
    Matrix<double> AP(dense.rows_size(), dense.cols_size());
    for (size_t j = 0; j < dense.cols_size(); ++j) {
        for (size_t i = 0; i < dense.rows_size(); ++i) {
            AP(i, j) = dense(i, perm[j]);
        }
    }

    EXPECT_TRUE(DenseApproxEqual(qr_product, AP, eps));
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

TEST(SparseQRTests, solveSquareFullRankSystem) {
    Matrix<double> dense = {
        {4.0, 1.0, 0.0},
        {0.0, 3.0, -1.0},
        {2.0, 0.0, 5.0}
    };
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();

    Matrix<double> xExpected(3, 1);
    xExpected(0, 0) = 1.5;
    xExpected(1, 0) = -2.0;
    xExpected(2, 0) = 0.5;
    Matrix<double> b = dense * xExpected;
    Matrix<double> x = qr.solve(b, 0.0);

    EXPECT_TRUE(DenseApproxEqual(x, xExpected, 1e-10));
    EXPECT_EQ(qr.rank(), 3u);
    EXPECT_TRUE(IsUpperTriangular(qr.R(), 1e-10));
}

TEST(SparseQRTests, solveTallFullRankSystemAndThinQ) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 0.0},
        {4.0, 0.0, 5.0},
        {0.0, 6.0, 1.0},
        {1.0, 2.0, 0.0}
    };
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();

    Matrix<double> q = qr.Q();
    EXPECT_EQ(q.rows_size(), dense.rows_size());
    EXPECT_EQ(q.cols_size(), std::min(dense.rows_size(), dense.cols_size()));
    EXPECT_TRUE(IsOrthonormal(q, 1e-10));

    Matrix<double> xExpected(3, 1);
    xExpected(0, 0) = 1.0;
    xExpected(1, 0) = -2.0;
    xExpected(2, 0) = 3.0;
    Matrix<double> b = dense * xExpected;
    Matrix<double> x = qr.solve(b, 0.0);

    EXPECT_TRUE(DenseApproxEqual(x, xExpected, 1e-9));
    EXPECT_EQ(qr.rank(), dense.cols_size());
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

TEST(SparseQRTests, scaleAwareThresholdMatchesEigenAndTolCanOverride) {
    Matrix<double> dense = {
        {1.0e10, 1.0e10},
        {0.0, 1.0e-8},
        {0.0, 0.0},
        {0.0, 0.0}
    };
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();

    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparse);
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);

    EXPECT_EQ(qr.rank(), static_cast<size_t>(eigenQr.rank()));
    EXPECT_EQ(qr.rank(), 1u);
    EXPECT_EQ(qr.rank(1e-12), 2u);
    EXPECT_LT(std::abs(qr.R()(1, 1)), 1e-6);
}

TEST(SparseQRTests, setPivotThresholdControlsNumericalRank) {
    Matrix<double> dense = {
        {1.0, 1.0},
        {0.0, 1.0e-6},
        {0.0, 0.0}
    };
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.setPivotThreshold(1.0e-5);
    qr.qr();

    EXPECT_EQ(qr.rank(), 1u);
    EXPECT_EQ(qr.rank(1.0e-7), 2u);
}

TEST(SparseQRTests, solveRejectsUnderdeterminedSystemsExplicitly) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 4.0}
    };
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();

    Matrix<double> b(2, 1);
    b(0, 0) = 1.0;
    b(1, 0) = 2.0;

    try {
        (void)qr.solve(b, 0.0);
        FAIL() << "Expected solve() to reject underdetermined systems.";
    } catch (const std::runtime_error& err) {
        EXPECT_NE(std::string(err.what()).find("pseudoInverse"), std::string::npos);
    }
}

TEST(SparseQRTests, solveRejectsRankDeficientSystemsExplicitly) {
    Matrix<double> dense = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 6.0},
        {0.0, 1.0, 1.0},
        {0.0, 2.0, 2.0}
    };
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();

    Matrix<double> b(4, 1);
    b(0, 0) = 1.0;
    b(1, 0) = 2.0;
    b(2, 0) = 3.0;
    b(3, 0) = 4.0;

    try {
        (void)qr.solve(b, 0.0);
        FAIL() << "Expected solve() to reject rank-deficient systems.";
    } catch (const std::runtime_error& err) {
        EXPECT_NE(std::string(err.what()).find("rank-deficient"), std::string::npos);
    }
}

TEST(SparseQRTests, pseudoInverseTallFullRankHasExpectedShapeAndAction) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 0.0},
        {4.0, 0.0, 5.0},
        {0.0, 6.0, 1.0},
        {1.0, 2.0, 0.0}
    };
    SparseMatrix<double> sparse(dense);
    SparseQR qr(sparse);
    qr.qr();

    Matrix<double> pinv = qr.pseudoInverse(0.0);
    EXPECT_EQ(pinv.rows_size(), dense.cols_size());
    EXPECT_EQ(pinv.cols_size(), dense.rows_size());

    Matrix<double> leftIdentity = pinv * dense;
    Matrix<double> expected = Matrix<double>::identity(dense.cols_size(), dense.cols_size());
    EXPECT_TRUE(DenseApproxEqual(leftIdentity, expected, 1e-8));
}

TEST(SparseQRTests, numericPivotingFullRankRectangularMatchesEigenRank) {
    Matrix<double> dense = BuildSparseLikeDense(60, 35, 0.08, 2027u);
    for (size_t i = 0; i < dense.cols_size(); ++i) {
        dense(i % dense.rows_size(), i) += 1.0;
    }
    SparseMatrix<double> sparse(dense);

    SparseQR qrPivot(sparse);
    qrPivot.qr();

    SparseQR qrNoPivot(sparse);
    qrNoPivot.setNumericPivotingEnabled(false);
    qrNoPivot.qr();

    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparse);
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);
    const size_t rankEigen = static_cast<size_t>(eigenQr.rank());

    const size_t errPivot =
        (qrPivot.rank() > rankEigen) ? (qrPivot.rank() - rankEigen) : (rankEigen - qrPivot.rank());
    const size_t errNoPivot =
        (qrNoPivot.rank() > rankEigen) ? (qrNoPivot.rank() - rankEigen) : (rankEigen - qrNoPivot.rank());
    EXPECT_LE(errPivot, errNoPivot);
}

TEST(SparseQRTests, numericPivotingImprovesIllConditionedResidualAgainstNoPivot) {
    Matrix<double> dense(80, 40);
    for (size_t i = 0; i < 40; ++i) {
        dense(i, i) = 1.0;
    }
    for (size_t i = 0; i < 40; ++i) {
        dense(i, 0) += std::pow(10.0, -10.0) * static_cast<double>(i + 1);
        dense(i + 40, i) = std::pow(10.0, -8.0) * static_cast<double>((i % 7) + 1);
        dense(i + 40, 0) += std::pow(10.0, -12.0) * static_cast<double>(i + 3);
    }

    Matrix<double> xTrue(40, 1);
    for (size_t i = 0; i < 40; ++i) xTrue(i, 0) = std::sin(0.03 * static_cast<double>(i + 1));
    Matrix<double> b = dense * xTrue;
    for (size_t i = 0; i < b.rows_size(); ++i) {
        b(i, 0) += 1e-10 * std::cos(0.11 * static_cast<double>(i));
    }

    SparseMatrix<double> sparse(dense);
    SparseQR qrPivot(sparse);
    qrPivot.qr();
    Matrix<double> xPivot = qrPivot.solve(b, 1e-12);
    const double resPivot = ResidualNorm(dense, xPivot, b);

    SparseQR qrNoPivot(sparse);
    qrNoPivot.setNumericPivotingEnabled(false);
    qrNoPivot.qr();
    Matrix<double> xNoPivot = qrNoPivot.solve(b, 1e-12);
    const double resNoPivot = ResidualNorm(dense, xNoPivot, b);

    EXPECT_LE(resPivot, resNoPivot * 1.05 + 1e-12);
}

TEST(SparseQRTests, numericPivotingRankDeficientTracksEigenBetterThanNoPivot) {
    Matrix<double> dense(90, 45);
    for (size_t i = 0; i < 30; ++i) {
        dense(i, i) = 2.0;
        dense(i + 30, i) = -1.0;
        dense(i + 60, i) = 0.5;
    }
    for (size_t i = 0; i < 15; ++i) {
        dense(i, 30 + i) = 1.0;
        dense(i + 30, 30 + i) = 2.0;
        dense(i + 60, 30 + i) = -3.0;
        dense(i, 15 + i) += 1e-10;
    }

    SparseMatrix<double> sparse(dense);
    SparseQR qrPivot(sparse);
    qrPivot.qr();

    SparseQR qrNoPivot(sparse);
    qrNoPivot.setNumericPivotingEnabled(false);
    qrNoPivot.qr();

    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparse);
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);
    const size_t rankEigen = static_cast<size_t>(eigenQr.rank());

    const size_t errPivot =
        (qrPivot.rank() > rankEigen) ? (qrPivot.rank() - rankEigen) : (rankEigen - qrPivot.rank());
    const size_t errNoPivot =
        (qrNoPivot.rank() > rankEigen) ? (qrNoPivot.rank() - rankEigen) : (rankEigen - qrNoPivot.rank());
    EXPECT_LE(errPivot, errNoPivot);
}

TEST(SparseQRTests, numericPivotingNearlyDependentColumnsImprovesRankEstimate) {
    Matrix<double> dense(70, 35);
    for (size_t i = 0; i < 35; ++i) {
        dense(i, i) = 1.0;
    }
    for (size_t i = 0; i < 15; ++i) {
        for (size_t r = 0; r < 70; ++r) {
            dense(r, 20 + i) = dense(r, i) + 1e-11 * static_cast<double>((r + i) % 5 + 1);
        }
    }

    SparseMatrix<double> sparse(dense);
    SparseQR qrPivot(sparse);
    qrPivot.qr();

    SparseQR qrNoPivot(sparse);
    qrNoPivot.setNumericPivotingEnabled(false);
    qrNoPivot.qr();

    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparse);
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);
    const size_t rankEigen = static_cast<size_t>(eigenQr.rank());

    const size_t errPivot =
        (qrPivot.rank() > rankEigen) ? (qrPivot.rank() - rankEigen) : (rankEigen - qrPivot.rank());
    const size_t errNoPivot =
        (qrNoPivot.rank() > rankEigen) ? (qrNoPivot.rank() - rankEigen) : (rankEigen - qrNoPivot.rank());
    EXPECT_LE(errPivot, errNoPivot);
}
