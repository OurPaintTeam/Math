#include <algorithm>
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

constexpr size_t kMaxRowNnzForGeometry = 12;

void EnforceRowNnzCap(Matrix<double>& matrix, size_t max_row_nnz = kMaxRowNnzForGeometry) {
    for (size_t i = 0; i < matrix.rows_size(); ++i) {
        std::vector<std::pair<double, size_t>> nz;
        nz.reserve(matrix.cols_size());
        for (size_t j = 0; j < matrix.cols_size(); ++j) {
            const double v = matrix(i, j);
            if (v != 0.0) nz.emplace_back(std::abs(v), j);
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

bool DenseApproxEqual(const Matrix<double>& A, const Matrix<double>& B, double eps = 1e-8) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) return false;
    for (size_t i = 0; i < A.rows_size(); ++i)
        for (size_t j = 0; j < A.cols_size(); ++j)
            if (std::abs(A(i, j) - B(i, j)) > eps) return false;
    return true;
}

Matrix<double> BuildSparseLikeDense(size_t rows, size_t cols, double density, uint32_t seed) {
    Matrix<double> result(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> val(-10.0, 10.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (prob(gen) < density) result(i, j) = val(gen);
    EnforceRowNnzCap(result);
    return result;
}

Matrix<double> BuildRankDeficient(size_t rows, size_t cols, size_t target_rank,
                                   double density, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> val(-5.0, 5.0);
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    Matrix<double> L(rows, target_rank);
    Matrix<double> R(target_rank, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < target_rank; ++j)
            if (prob(gen) < density * 3.0) L(i, j) = val(gen);
    for (size_t i = 0; i < target_rank; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (prob(gen) < density * 3.0) R(i, j) = val(gen);
    Matrix<double> result = L * R;
    EnforceRowNnzCap(result);
    return result;
}

Eigen::SparseMatrix<double> ToEigenSparse(const SparseMatrix<double>& matrix) {
    Eigen::SparseMatrix<double> result(
        static_cast<int>(matrix.rows_size()),
        static_cast<int>(matrix.cols_size()));
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(matrix.nonZeros());
    for (size_t row = 0; row < matrix.rows_size(); ++row)
        for (SparseMatrix<double>::InnerIterator it(matrix, row); it; ++it)
            triplets.emplace_back(
                static_cast<int>(it.row()),
                static_cast<int>(it.col()),
                it.value());
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

Matrix<double> DenseFromEigen(const Eigen::MatrixXd& matrix) {
    Matrix<double> result(static_cast<size_t>(matrix.rows()), static_cast<size_t>(matrix.cols()));
    for (int i = 0; i < matrix.rows(); ++i)
        for (int j = 0; j < matrix.cols(); ++j)
            result(static_cast<size_t>(i), static_cast<size_t>(j)) = matrix(i, j);
    return result;
}

Eigen::MatrixXd EigenFromDense(const Matrix<double>& matrix) {
    Eigen::MatrixXd result(
        static_cast<int>(matrix.rows_size()),
        static_cast<int>(matrix.cols_size()));
    for (int i = 0; i < result.rows(); ++i)
        for (int j = 0; j < result.cols(); ++j)
            result(i, j) = matrix(static_cast<size_t>(i), static_cast<size_t>(j));
    return result;
}

Matrix<double> BuildDenseRhs(size_t rows, size_t rhsCols, uint32_t seed) {
    Matrix<double> b(rows, rhsCols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> val(-3.0, 3.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < rhsCols; ++j)
            b(i, j) = val(gen);
    return b;
}

double ResidualNorm(const Matrix<double>& A, const Matrix<double>& x, const Matrix<double>& b) {
    Matrix<double> r = A * x - b;
    return r.norm();
}

struct FactorizationTimes {
    double analyzeMs;
    double factorizeMs;
    double totalMs;
};

FactorizationTimes TimeSparseQrSplit(const SparseMatrix<double>& A, int repeats) {
    double aTotal = 0, fTotal = 0, tTotal = 0;
    for (int i = 0; i < repeats; ++i) {
        SparseQR qr(A);
        auto t0 = std::chrono::steady_clock::now();
        qr.analyze();
        auto t1 = std::chrono::steady_clock::now();
        qr.factorize();
        auto t2 = std::chrono::steady_clock::now();
        aTotal += std::chrono::duration<double, std::milli>(t1 - t0).count();
        fTotal += std::chrono::duration<double, std::milli>(t2 - t1).count();
        tTotal += std::chrono::duration<double, std::milli>(t2 - t0).count();
    }
    return {aTotal / repeats, fTotal / repeats, tTotal / repeats};
}

double TimeEigenQrMs(const Eigen::SparseMatrix<double>& A, int repeats) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
        qr.compute(A);
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
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

double TimeSparseQrMs(const SparseMatrix<double>& A, int repeats,
                      Matrix<double>& q, Matrix<double>& r,
                      std::vector<size_t>& perm) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        SparseQR qr(A);
        qr.qr();
        q = qr.Q();
        r = qr.R();
        perm = qr.colsPermutation();
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double TimeSparseQrReuseMs(const SparseMatrix<double>& A, int repeats,
                           Matrix<double>& q, Matrix<double>& r,
                           std::vector<size_t>& perm) {
    SparseQR qr(A);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        qr.qr();
        q = qr.Q();
        r = qr.R();
        perm = qr.colsPermutation();
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double TimeEigenSparseQrMs(const Eigen::SparseMatrix<double>& A, int repeats,
                           Matrix<double>& qOut, Matrix<double>& rOut) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
        qr.compute(A);
        const int thin_cols = std::min(A.rows(), A.cols());
        Eigen::MatrixXd thinIdentity = Eigen::MatrixXd::Identity(A.rows(), thin_cols);
        Eigen::MatrixXd q = qr.matrixQ() * thinIdentity;
        Eigen::MatrixXd r = Eigen::MatrixXd(qr.matrixR()).topRows(thin_cols);
        qOut = DenseFromEigen(q);
        rOut = DenseFromEigen(r);
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double TimeSparseSolveOnlyMs(SparseQR& qr, const Matrix<double>& b, int repeats, Matrix<double>& xOut) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        xOut = qr.solve(b);
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double TimeEigenSolveOnlyMs(Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>& qr,
                            const Eigen::MatrixXd& b,
                            int repeats,
                            Matrix<double>& xOut) {
    auto start = std::chrono::steady_clock::now();
    Eigen::MatrixXd x;
    for (int i = 0; i < repeats; ++i) {
        x = qr.solve(b);
    }
    auto end = std::chrono::steady_clock::now();
    xOut = DenseFromEigen(x);
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

    Matrix<double> denseQ, denseR, sparseQ, sparseR, eigenQ, eigenR;
    std::vector<size_t> sparsePerm;

    const double denseMs = TimeDenseQrMs(dense, repeats, denseQ, denseR);
    const double sparseMs = TimeSparseQrMs(sparse, repeats, sparseQ, sparseR, sparsePerm);
    const double eigenMs = TimeEigenSparseQrMs(eigenSparse, repeats, eigenQ, eigenR);

    EXPECT_TRUE(DenseApproxEqual(denseQ * denseR, dense, 1e-6));

    Matrix<double> AP_sparse(dense.rows_size(), dense.cols_size());
    for (size_t j = 0; j < dense.cols_size(); ++j)
        for (size_t i = 0; i < dense.rows_size(); ++i)
            AP_sparse(i, j) = dense(i, sparsePerm[j]);
    EXPECT_TRUE(DenseApproxEqual(sparseQ * sparseR, AP_sparse, 1e-6));

    std::cout << "rows=" << rows
              << " cols=" << cols
              << " density=" << density
              << " dense_qr_ms=" << denseMs
              << " sparse_qr_ms=" << sparseMs
              << " eigen_sparse_qr_ms=" << eigenMs
              << std::endl;

    return {denseMs, sparseMs, eigenMs};
}

void RunFactorizationBench(size_t rows, size_t cols, double density,
                           int repeats, uint32_t seed) {
    Matrix<double> dense = BuildSparseLikeDense(rows, cols, density, seed);
    SparseMatrix<double> sparse(dense);
    Eigen::SparseMatrix<double> eigenSparse = ToEigenSparse(sparse);

    auto split = TimeSparseQrSplit(sparse, repeats);
    double eigenMs = TimeEigenQrMs(eigenSparse, repeats);

    std::cout << "factorize_bench"
              << " rows=" << rows << " cols=" << cols
              << " density=" << density
              << " analyze_ms=" << split.analyzeMs
              << " factorize_ms=" << split.factorizeMs
              << " total_ms=" << split.totalMs
              << " eigen_ms=" << eigenMs
              << " ratio=" << (split.totalMs / std::max(eigenMs, 0.001))
              << std::endl;
}

void RunSolveCase(size_t rows, size_t cols, size_t rhsCols, double density, int repeats,
                  uint32_t seedA, uint32_t seedB, double residualFactor)
{
    Matrix<double> denseA = BuildSparseLikeDense(rows, cols, density, seedA);
    SparseMatrix<double> sparseA(denseA);
    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparseA);
    Matrix<double> b = BuildDenseRhs(rows, rhsCols, seedB);

    SparseQR sparseQr(sparseA);
    sparseQr.qr();

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);

    const Eigen::MatrixXd bEigen = EigenFromDense(b);

    Matrix<double> xSparse, xEigen;

    const double sparseSolveMs = TimeSparseSolveOnlyMs(sparseQr, b, repeats, xSparse);
    const double eigenSolveMs = TimeEigenSolveOnlyMs(eigenQr, bEigen, repeats, xEigen);

    const double sparseResidual = ResidualNorm(denseA, xSparse, b);
    const double eigenResidual = ResidualNorm(denseA, xEigen, b);

    EXPECT_GT(sparseSolveMs, 0.0);
    EXPECT_GT(eigenSolveMs, 0.0);
    EXPECT_TRUE(std::isfinite(sparseResidual));
    EXPECT_TRUE(std::isfinite(eigenResidual));
    EXPECT_LE(sparseResidual, eigenResidual * residualFactor + 1e-6);

    const double inv = 1.0 / static_cast<double>(repeats);
    std::cout << "solve_case=" << rows << "x" << cols
              << " rhs_cols=" << rhsCols
              << " repeats=" << repeats
              << " sparse_solve_total_ms=" << sparseSolveMs
              << " sparse_solve_per_run_ms=" << (sparseSolveMs * inv)
              << " eigen_solve_total_ms=" << eigenSolveMs
              << " eigen_solve_per_run_ms=" << (eigenSolveMs * inv)
              << " sparse_residual=" << sparseResidual
              << " eigen_residual=" << eigenResidual
              << std::endl;
}

void RunRepeatedSolveBench(size_t rows, size_t cols, size_t rhsCols,
                           double density, int numSolves,
                           uint32_t seedA, uint32_t seedB) {
    Matrix<double> denseA = BuildSparseLikeDense(rows, cols, density, seedA);
    SparseMatrix<double> sparseA(denseA);
    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparseA);

    SparseQR sparseQr(sparseA);
    auto t0 = std::chrono::steady_clock::now();
    sparseQr.qr();
    auto t1 = std::chrono::steady_clock::now();
    double sparseFactMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    t0 = std::chrono::steady_clock::now();
    eigenQr.compute(eigenA);
    t1 = std::chrono::steady_clock::now();
    double eigenFactMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::mt19937 gen(seedB);
    std::uniform_real_distribution<double> val(-3.0, 3.0);

    double sparseSolveTotalMs = 0, eigenSolveTotalMs = 0;
    for (int s = 0; s < numSolves; ++s) {
        Matrix<double> b(rows, rhsCols);
        Eigen::MatrixXd bEigen(static_cast<int>(rows), static_cast<int>(rhsCols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < rhsCols; ++j) {
                double v = val(gen);
                b(i, j) = v;
                bEigen(static_cast<int>(i), static_cast<int>(j)) = v;
            }
        }

        Matrix<double> xs;
        t0 = std::chrono::steady_clock::now();
        xs = sparseQr.solve(b);
        t1 = std::chrono::steady_clock::now();
        sparseSolveTotalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

        t0 = std::chrono::steady_clock::now();
        Eigen::MatrixXd xe = eigenQr.solve(bEigen);
        t1 = std::chrono::steady_clock::now();
        eigenSolveTotalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::cout << "repeated_solve"
              << " rows=" << rows << " cols=" << cols
              << " rhs=" << rhsCols << " solves=" << numSolves
              << " sparse_fact_ms=" << sparseFactMs
              << " eigen_fact_ms=" << eigenFactMs
              << " sparse_solve_total_ms=" << sparseSolveTotalMs
              << " eigen_solve_total_ms=" << eigenSolveTotalMs
              << " sparse_total_ms=" << (sparseFactMs + sparseSolveTotalMs)
              << " eigen_total_ms=" << (eigenFactMs + eigenSolveTotalMs)
              << std::endl;
}

} // namespace

// ---- Original factorization benchmarks ----

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

// ---- Factorization-only (analyze + factorize split timing) ----

TEST(SparseQRPerformanceTests, FactorizeSplit_500x350_d002) {
    RunFactorizationBench(500, 350, 0.02, 2, 99u);
}

TEST(SparseQRPerformanceTests, FactorizeSplit_1000x700_d001) {
    RunFactorizationBench(1000, 700, 0.01, 1, 12345u);
}

TEST(SparseQRPerformanceTests, FactorizeSplit_2000x1400_d005) {
    RunFactorizationBench(2000, 1400, 0.005, 1, 22222u);
}

// ---- Different density sweep ----

TEST(SparseQRPerformanceTests, DensitySweep_500x350_d001) {
    RunFactorizationBench(500, 350, 0.001, 2, 1001u);
}

TEST(SparseQRPerformanceTests, DensitySweep_500x350_d005) {
    RunFactorizationBench(500, 350, 0.005, 2, 1002u);
}

TEST(SparseQRPerformanceTests, DensitySweep_500x350_d01) {
    RunFactorizationBench(500, 350, 0.01, 2, 1003u);
}

TEST(SparseQRPerformanceTests, DensitySweep_500x350_d05) {
    RunFactorizationBench(500, 350, 0.05, 1, 1004u);
}

TEST(SparseQRPerformanceTests, DensitySweep_500x350_d1) {
    RunFactorizationBench(500, 350, 0.1, 1, 1005u);
}

// ---- Different aspect ratios ----

TEST(SparseQRPerformanceTests, Aspect_1000x100_d005) {
    RunFactorizationBench(1000, 100, 0.05, 2, 2001u);
}

TEST(SparseQRPerformanceTests, Aspect_1000x500_d005) {
    RunFactorizationBench(1000, 500, 0.005, 1, 2002u);
}

TEST(SparseQRPerformanceTests, Aspect_1000x900_d003) {
    RunFactorizationBench(1000, 900, 0.003, 1, 2003u);
}

TEST(SparseQRPerformanceTests, Aspect_500x500_d01) {
    RunFactorizationBench(500, 500, 0.01, 2, 2004u);
}

// ---- Solve benchmarks ----

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_300x200) {
    RunSolveCase(300, 200, 4, 0.03, 3, 2026u, 2027u, 100.0);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_500x350) {
    RunSolveCase(500, 350, 3, 0.02, 2, 99u, 100u, 150.0);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_700x500) {
    RunSolveCase(700, 500, 3, 0.015, 1, 555u, 556u, 150.0);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_1000x700) {
    RunSolveCase(1000, 700, 2, 0.01, 1, 12345u, 12346u, 200.0);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_2000x1400) {
    RunSolveCase(2000, 1400, 2, 0.005, 1, 22222u, 22223u, 250.0);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_3000x2000) {
    RunSolveCase(3000, 2000, 1, 0.003, 1, 33333u, 33334u, 300.0);
}

// ---- Repeated solve (1 factorization + N solves) ----

TEST(SparseQRPerformanceTests, RepeatedSolve_500x350_10rhs) {
    RunRepeatedSolveBench(500, 350, 1, 0.02, 10, 99u, 100u);
}

TEST(SparseQRPerformanceTests, RepeatedSolve_1000x700_20rhs) {
    RunRepeatedSolveBench(1000, 700, 1, 0.01, 20, 12345u, 12346u);
}

TEST(SparseQRPerformanceTests, RepeatedSolve_2000x1400_10rhs) {
    RunRepeatedSolveBench(2000, 1400, 1, 0.005, 10, 22222u, 22223u);
}

TEST(SparseQRPerformanceTests, RepeatedSolve_500x350_multirhs) {
    RunRepeatedSolveBench(500, 350, 4, 0.02, 5, 99u, 101u);
}

// ---- Structural analysis reuse ----

TEST(SparseQRPerformanceTests, ReuseStructuralAnalysis_1000x700) {
    Matrix<double> dense = BuildSparseLikeDense(1000, 700, 0.01, 12345u);
    SparseMatrix<double> sparse(dense);
    Matrix<double> freshQ, freshR, reuseQ, reuseR;
    std::vector<size_t> freshPerm, reusePerm;

    const double freshMs = TimeSparseQrMs(sparse, 3, freshQ, freshR, freshPerm);
    const double reuseMs = TimeSparseQrReuseMs(sparse, 3, reuseQ, reuseR, reusePerm);

    EXPECT_TRUE(DenseApproxEqual(freshQ * freshR, reuseQ * reuseR, 1e-6));
    EXPECT_GT(freshMs, 0.0);
    EXPECT_GT(reuseMs, 0.0);

    std::cout << "reuse_case=1000x700"
              << " fresh_sparse_qr_ms=" << freshMs
              << " reused_sparse_qr_ms=" << reuseMs
              << std::endl;
}

// ---- Rank-deficient factorization timing ----

TEST(SparseQRPerformanceTests, RankDeficient_500x200_rank100) {
    Matrix<double> dense = BuildRankDeficient(500, 200, 100, 0.05, 9876u);
    SparseMatrix<double> sparse(dense);
    Eigen::SparseMatrix<double> eigenSparse = ToEigenSparse(sparse);

    auto split = TimeSparseQrSplit(sparse, 2);
    double eigenMs = TimeEigenQrMs(eigenSparse, 2);

    SparseQR qr(sparse);
    qr.qr();
    size_t r = qr.rank();

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenSparse);
    int eigenRank = eigenQr.rank();

    std::cout << "rank_deficient 500x200"
              << " target_rank=100"
              << " detected_rank=" << r
              << " eigen_rank=" << eigenRank
              << " sparse_ms=" << split.totalMs
              << " eigen_ms=" << eigenMs
              << std::endl;

    // Under row-nnz cap (<=12), post-sparsification rank can deviate from target_rank.
    // Validate consistency against Eigen rank instead of hard target window.
    EXPECT_EQ(r, static_cast<size_t>(eigenRank));
}

TEST(SparseQRPerformanceTests, RankDeficient_1000x500_rank200) {
    Matrix<double> dense = BuildRankDeficient(1000, 500, 200, 0.02, 5555u);
    SparseMatrix<double> sparse(dense);
    Eigen::SparseMatrix<double> eigenSparse = ToEigenSparse(sparse);

    auto split = TimeSparseQrSplit(sparse, 1);
    double eigenMs = TimeEigenQrMs(eigenSparse, 1);

    SparseQR qr(sparse);
    qr.qr();
    size_t r = qr.rank();

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenSparse);
    int eigenRank = eigenQr.rank();

    std::cout << "rank_deficient 1000x500"
              << " target_rank=200"
              << " detected_rank=" << r
              << " eigen_rank=" << eigenRank
              << " sparse_ms=" << split.totalMs
              << " eigen_ms=" << eigenMs
              << std::endl;

    EXPECT_EQ(r, static_cast<size_t>(eigenRank));
}
