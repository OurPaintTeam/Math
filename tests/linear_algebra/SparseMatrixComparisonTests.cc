#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "Matrix.h"
#include "SparseMatrix.h"
#include <Eigen/Sparse>

namespace {

bool DenseEqual(const Matrix<double>& A, const Matrix<double>& B, double eps = 1e-10) {
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

Matrix<double> BuildSparseLikeDense(size_t n, double density, uint32_t seed) {
    Matrix<double> A(n, n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> val(-10.0, 10.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (prob(gen) < density) {
                A(i, j) = val(gen);
            }
        }
    }
    return A;
}

double TimeDenseMs(const Matrix<double>& A, int repeats, Matrix<double>& out) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        out = A * A;
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double TimeOurSparseMs(const SparseMatrix<double>& A, int repeats, SparseMatrix<double>& out) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        out = A * A;
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double TimeEigenSparseMs(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, int repeats,
                         Eigen::SparseMatrix<double, Eigen::RowMajor>& out) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        out = A * A;
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct BenchResult {
    double denseMs;
    double ourMs;
    double eigenMs;
};

BenchResult RunCase(size_t n, double density, int repeats, uint32_t seed) {
    Matrix<double> dense = BuildSparseLikeDense(n, density, seed);
    SparseMatrix<double> ourSparse(dense);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ourSparse.nonZeros());
    for (size_t i = 0; i < dense.rows_size(); ++i) {
        for (size_t j = 0; j < dense.cols_size(); ++j) {
            const double v = dense(i, j);
            if (v != 0.0) {
                triplets.emplace_back(static_cast<int>(i), static_cast<int>(j), v);
            }
        }
    }

    Eigen::SparseMatrix<double, Eigen::RowMajor> eigenSparse(static_cast<int>(n), static_cast<int>(n));
    eigenSparse.setFromTriplets(triplets.begin(), triplets.end());

    Matrix<double> denseMul(n, n);
    SparseMatrix<double> ourMul(n, n);
    Eigen::SparseMatrix<double, Eigen::RowMajor> eigenMul(static_cast<int>(n), static_cast<int>(n));

    const double denseMs = TimeDenseMs(dense, repeats, denseMul);
    const double ourMs = TimeOurSparseMs(ourSparse, repeats, ourMul);
    const double eigenMs = TimeEigenSparseMs(eigenSparse, repeats, eigenMul);

    Matrix<double> ourDenseMul = ourMul.toDense();
    EXPECT_TRUE(DenseEqual(ourDenseMul, denseMul, 1e-8));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double ev = eigenMul.coeff(static_cast<int>(i), static_cast<int>(j));
            EXPECT_NEAR(ourMul(i, j), ev, 1e-8);
        }
    }

    EXPECT_GT(denseMs, 0.0);
    EXPECT_GT(ourMs, 0.0);
    EXPECT_GT(eigenMs, 0.0);

    std::cout << "n=" << n
              << " density=" << density
              << " dense_ms=" << denseMs
              << " our_sparse_ms=" << ourMs
              << " eigen_sparse_ms=" << eigenMs
              << std::endl;

    return {denseMs, ourMs, eigenMs};
}

} // namespace

TEST(SparseMatrixComparisonTests, TimingCase1) {
    const auto r = RunCase(220, 0.02, 2, 42u);
    EXPECT_LE(r.ourMs, r.denseMs * 1.05);
    RecordProperty("dense_ms", r.denseMs);
    RecordProperty("our_sparse_ms", r.ourMs);
    RecordProperty("eigen_sparse_ms", r.eigenMs);
}

TEST(SparseMatrixComparisonTests, TimingCase2) {
    const auto r = RunCase(320, 0.01, 2, 1337u);
    EXPECT_LE(r.ourMs, r.denseMs * 1.05);
    RecordProperty("dense_ms", r.denseMs);
    RecordProperty("our_sparse_ms", r.ourMs);
    RecordProperty("eigen_sparse_ms", r.eigenMs);
}

TEST(SparseMatrixComparisonTests, TimingCase3) {
    const auto r = RunCase(420, 0.006, 1, 777u);
    EXPECT_LE(r.ourMs, r.denseMs * 1.05);
    RecordProperty("dense_ms", r.denseMs);
    RecordProperty("our_sparse_ms", r.ourMs);
    RecordProperty("eigen_sparse_ms", r.eigenMs);
}
