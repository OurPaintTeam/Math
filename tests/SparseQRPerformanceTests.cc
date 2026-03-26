#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "Matrix.h"
#include "QR.h"
#include "SparseMatrix.h"
#include "SparseQR.h"

namespace {

bool DenseApproxEqual(const Matrix<double>& A, const Matrix<double>& B, double eps = 1e-8) {
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

Matrix<double> BuildSparseLikeDense(size_t rows, size_t cols, double density, uint32_t seed) {
    Matrix<double> result(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> val(-10.0, 10.0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (prob(gen) < density) {
                result(i, j) = val(gen);
            }
        }
    }

    return result;
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

Matrix<double> DenseFromEigen(const Eigen::MatrixXd& matrix) {
    Matrix<double> result(static_cast<size_t>(matrix.rows()), static_cast<size_t>(matrix.cols()));
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            result(static_cast<size_t>(i), static_cast<size_t>(j)) = matrix(i, j);
        }
    }
    return result;
}

double TimeDenseQrMs(const Matrix<double>& A, int repeats, Matrix<double>& q, Matrix<double>& r) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        QR qr(A);
        qr.qrIMGS();
        q = qr.Q();
        r = qr.R();
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double TimeSparseQrMs(const SparseMatrix<double>& A, int repeats, Matrix<double>& q, Matrix<double>& r) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        SparseQR qr(A);
        qr.qr();
        q = qr.Q();
        r = qr.R();
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double TimeEigenSparseQrMs(const Eigen::SparseMatrix<double>& A, int repeats, Matrix<double>& qOut, Matrix<double>& rOut) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
        qr.compute(A);

        const int thin_cols = std::min(A.rows(), A.cols());
        Eigen::MatrixXd thinIdentity = Eigen::MatrixXd::Identity(A.rows(), thin_cols);
        Eigen::MatrixXd q = qr.matrixQ() * thinIdentity;
        Eigen::MatrixXd r = Eigen::MatrixXd(qr.matrixR()).topRows(thin_cols);

        // Время таймера должно включать только извлечение Q/R,
        // а плотную реконструкцию оставим на сравнение вне тайминга.
        qOut = DenseFromEigen(q);
        rOut = DenseFromEigen(r);
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct BenchResult {
    double denseMs;
    double sparseMs;
    double eigenMs;
};

BenchResult RunCase(size_t rows, size_t cols, double density, int repeats, uint32_t seed) {
    Matrix<double> dense = BuildSparseLikeDense(rows, cols, density, seed);
    SparseMatrix<double> sparse(dense);
    Eigen::SparseMatrix<double> eigenSparse = ToEigenSparse(sparse);

    Matrix<double> denseQ;
    Matrix<double> denseR;
    Matrix<double> sparseQ;
    Matrix<double> sparseR;
    Matrix<double> eigenQ;
    Matrix<double> eigenR;
    Matrix<double> eigenRestored;

    const double denseMs = TimeDenseQrMs(dense, repeats, denseQ, denseR);
    const double sparseMs = TimeSparseQrMs(sparse, repeats, sparseQ, sparseR);
    const double eigenMs = TimeEigenSparseQrMs(eigenSparse, repeats, eigenQ, eigenR);

    EXPECT_TRUE(DenseApproxEqual(denseQ * denseR, dense, 1e-6));
    EXPECT_TRUE(DenseApproxEqual(sparseQ * sparseR, dense, 1e-6));

    // Для Eigen реконструкция включает обратную перестановку столбцов.
    // Она делается отдельно от тайминга для корректного сравнения скоростей.
    {
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
        qr.compute(eigenSparse);

        const int thin_cols = std::min(eigenSparse.rows(), eigenSparse.cols());
        Eigen::MatrixXd thinIdentity = Eigen::MatrixXd::Identity(eigenSparse.rows(), thin_cols);
        Eigen::MatrixXd q = qr.matrixQ() * thinIdentity;
        Eigen::MatrixXd r = Eigen::MatrixXd(qr.matrixR()).topRows(thin_cols);
        Eigen::MatrixXd reconstructed = q * r * qr.colsPermutation().inverse();
        eigenRestored = DenseFromEigen(reconstructed);
    }

    EXPECT_TRUE(DenseApproxEqual(eigenRestored, dense, 1e-6));

    EXPECT_GT(denseMs, 0.0);
    EXPECT_GT(sparseMs, 0.0);
    EXPECT_GT(eigenMs, 0.0);

    std::cout << "rows=" << rows
              << " cols=" << cols
              << " density=" << density
              << " dense_qr_ms=" << denseMs
              << " sparse_qr_ms=" << sparseMs
              << " eigen_sparse_qr_ms=" << eigenMs
              << std::endl;

    return {denseMs, sparseMs, eigenMs};
}

} // namespace

TEST(SparseQRPerformanceTests, Small_80x60) {
    RunCase(80, 60, 0.05, 3, 42u);
}

TEST(SparseQRPerformanceTests, Small_120x90) {
    RunCase(120, 90, 0.03, 3, 1337u);
}

TEST(SparseQRPerformanceTests, Medium_200x150) {
    RunCase(200, 150, 0.03, 2, 7u);
}

TEST(SparseQRPerformanceTests, Medium_300x200) {
    RunCase(300, 200, 0.025, 2, 2026u);
}

TEST(SparseQRPerformanceTests, Large_500x350) {
    RunCase(500, 350, 0.02, 1, 99u);
}

TEST(SparseQRPerformanceTests, Large_700x500) {
    RunCase(700, 500, 0.015, 1, 555u);
}

TEST(SparseQRPerformanceTests, XLarge_1000x700) {
    RunCase(1000, 700, 0.01, 1, 12345u);
}
