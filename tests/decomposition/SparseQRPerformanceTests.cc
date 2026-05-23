#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

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
constexpr double kResidualCompareTolerance = 1.0e-7;
constexpr double kKnownSolutionTolerance = 1.0e-8;

double NonZeroRandom(std::mt19937& gen, double lo = -10.0, double hi = 10.0) {
    std::uniform_real_distribution<double> val(lo, hi);
    double v = val(gen);
    if (std::abs(v) < 0.25) {
        v += (v < 0.0) ? -0.25 : 0.25;
    }
    return v;
}

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
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (prob(gen) < density) result(i, j) = NonZeroRandom(gen);
    EnforceRowNnzCap(result);
    return result;
}

Matrix<double> BuildRankDeficient(size_t rows, size_t cols, size_t target_rank,
                                   double density, uint32_t seed) {
    if (target_rank == 0 || target_rank > cols || cols > rows) {
        throw std::invalid_argument("BuildRankDeficient requires 0 < target_rank <= cols <= rows.");
    }

    Matrix<double> result(rows, cols);
    const size_t tail_rows = rows - target_rank;
    const size_t requested_extra =
        static_cast<size_t>(std::ceil(std::max(0.0, density) * static_cast<double>(rows)));
    const size_t extra_per_independent =
        std::min<size_t>(tail_rows, std::min<size_t>(2, requested_extra));

    for (size_t col = 0; col < target_rank; ++col) {
        result(col, col) = 1.0;
        for (size_t extra = 0; extra < extra_per_independent; ++extra) {
            const size_t row =
                target_rank + ((col * 131u + extra * 17u + seed) % tail_rows);
            result(row, col) = ((col + extra) % 2 == 0) ? 0.5 : -0.25;
        }
    }

    for (size_t col = target_rank; col < cols; ++col) {
        const size_t source_col = (col - target_rank) % target_rank;
        const double scale = ((col - target_rank) % 2 == 0) ? -1.0 : 1.0;
        for (size_t row = 0; row < rows; ++row) {
            result(row, col) = scale * result(row, source_col);
        }
    }

    return result;
}

Matrix<double> BuildGeometryLikeMatrix(size_t rows, size_t cols, size_t rowNnz, uint32_t seed) {
    if (rowNnz == 0 || rowNnz > cols) {
        throw std::invalid_argument("BuildGeometryLikeMatrix requires 0 < rowNnz <= cols.");
    }

    Matrix<double> result(rows, cols);
    std::mt19937 gen(seed);
    std::vector<size_t> columns(cols);
    std::iota(columns.begin(), columns.end(), 0);

    for (size_t row = 0; row < rows; ++row) {
        std::shuffle(columns.begin(), columns.end(), gen);
        for (size_t k = 0; k < rowNnz; ++k) {
            result(row, columns[k]) = NonZeroRandom(gen);
        }
    }

    return result;
}

Matrix<double> BuildFullColumnRankMatrix(size_t rows, size_t cols, size_t rowNnz, uint32_t seed) {
    if (cols > rows) {
        throw std::invalid_argument("BuildFullColumnRankMatrix requires cols <= rows.");
    }

    rowNnz = std::max<size_t>(1, std::min(rowNnz, cols));
    Matrix<double> result(rows, cols);
    std::mt19937 gen(seed);

    for (size_t row = 0; row < cols; ++row) {
        result(row, row) = 2.0 + 0.01 * static_cast<double>((row % 13) + 1);

        std::vector<size_t> candidates(row);
        std::iota(candidates.begin(), candidates.end(), 0);
        std::shuffle(candidates.begin(), candidates.end(), gen);
        const size_t extra = std::min(rowNnz - 1, candidates.size());
        for (size_t k = 0; k < extra; ++k) {
            result(row, candidates[k]) = NonZeroRandom(gen, -1.0, 1.0);
        }
    }

    std::vector<size_t> columns(cols);
    std::iota(columns.begin(), columns.end(), 0);
    for (size_t row = cols; row < rows; ++row) {
        std::shuffle(columns.begin(), columns.end(), gen);
        for (size_t k = 0; k < rowNnz; ++k) {
            result(row, columns[k]) = NonZeroRandom(gen);
        }
    }

    return result;
}

Matrix<double> BuildSamePatternWithNewValues(const Matrix<double>& pattern, uint32_t seed) {
    Matrix<double> result(pattern.rows_size(), pattern.cols_size());
    std::mt19937 gen(seed);
    for (size_t row = 0; row < pattern.rows_size(); ++row) {
        for (size_t col = 0; col < pattern.cols_size(); ++col) {
            if (pattern(row, col) != 0.0) {
                result(row, col) = NonZeroRandom(gen);
            }
        }
    }
    return result;
}

Eigen::SparseMatrix<double> ToEigenSparse(const SparseMatrix<double>& matrix) {
    SparseMatrix<double> buffer;
    const SparseMatrix<double>* compressed = &matrix;
    if (matrix.rowMutableMode()) {
        buffer = matrix.compressed();
        compressed = &buffer;
    }

    Eigen::SparseMatrix<double> result(
        static_cast<int>(compressed->rows_size()),
        static_cast<int>(compressed->cols_size()));
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(compressed->nonZeros());
    for (size_t col = 0; col < compressed->cols_size(); ++col)
        for (SparseMatrix<double>::InnerIterator it(*compressed, col); it; ++it)
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
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < rhsCols; ++j)
            b(i, j) = NonZeroRandom(gen, -3.0, 3.0);
    return b;
}

double ResidualNorm(const Matrix<double>& A, const Matrix<double>& x, const Matrix<double>& b) {
    Matrix<double> r = A * x - b;
    return r.norm();
}

double RelativeMatrixError(const Matrix<double>& actual, const Matrix<double>& expected) {
    return (actual - expected).norm() / std::max(1.0, expected.norm());
}

struct MatrixStats {
    size_t nnz;
    double actualDensity;
};

MatrixStats GetMatrixStats(const SparseMatrix<double>& matrix) {
    const size_t nnz = matrix.nonZeros();
    const double total = static_cast<double>(matrix.rows_size()) * static_cast<double>(matrix.cols_size());
    return {nnz, total > 0.0 ? static_cast<double>(nnz) / total : 0.0};
}

size_t RowNnzFromDensity(size_t cols, double density) {
    const double requested = std::ceil(std::max(0.0, density) * static_cast<double>(cols));
    return std::max<size_t>(
        1,
        std::min<size_t>(kMaxRowNnzForGeometry, static_cast<size_t>(requested)));
}

struct TimingStats {
    double avgMs = 0.0;
    double medianMs = 0.0;
    int repeats = 0;
};

TimingStats StatsFromSamples(std::vector<double> samples) {
    TimingStats stats;
    stats.repeats = static_cast<int>(samples.size());
    if (samples.empty()) {
        return stats;
    }

    stats.avgMs = std::accumulate(samples.begin(), samples.end(), 0.0)
                  / static_cast<double>(samples.size());
    std::sort(samples.begin(), samples.end());
    const size_t mid = samples.size() / 2;
    stats.medianMs = (samples.size() % 2 == 0)
        ? 0.5 * (samples[mid - 1] + samples[mid])
        : samples[mid];
    return stats;
}

template <typename Fn>
TimingStats MeasureWithWarmup(int repeats, Fn&& fn) {
    if (repeats <= 0) {
        throw std::invalid_argument("repeats must be positive.");
    }

    fn();

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return StatsFromSamples(std::move(samples));
}

struct FactorizationTimes {
    TimingStats analyze;
    TimingStats factorize;
    TimingStats total;
};

double RatioMyToEigen(double myMs, double eigenMs) {
    return myMs / std::max(eigenMs, 1e-9);
}

FactorizationTimes TimeSparseQrSplit(const SparseMatrix<double>& A, int repeats) {
    {
        SparseQR qr(A);
        qr.analyze();
        qr.factorize();
    }

    std::vector<double> analyzeSamples;
    std::vector<double> factorizeSamples;
    std::vector<double> totalSamples;
    analyzeSamples.reserve(static_cast<size_t>(repeats));
    factorizeSamples.reserve(static_cast<size_t>(repeats));
    totalSamples.reserve(static_cast<size_t>(repeats));

    for (int i = 0; i < repeats; ++i) {
        SparseQR qr(A);
        auto t0 = std::chrono::steady_clock::now();
        qr.analyze();
        auto t1 = std::chrono::steady_clock::now();
        qr.factorize();
        auto t2 = std::chrono::steady_clock::now();
        analyzeSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        factorizeSamples.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        totalSamples.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());
    }

    return {
        StatsFromSamples(std::move(analyzeSamples)),
        StatsFromSamples(std::move(factorizeSamples)),
        StatsFromSamples(std::move(totalSamples))
    };
}

double TimeEigenQrMs(const Eigen::SparseMatrix<double>& A, int repeats, TimingStats* statsOut = nullptr) {
    TimingStats stats = MeasureWithWarmup(repeats, [&]() {
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
        qr.compute(A);
    });
    if (statsOut) {
        *statsOut = stats;
    }
    return stats.avgMs;
}

double TimeDenseQrMs(const Matrix<double>& A, int repeats, Matrix<double>& q, Matrix<double>& r,
                     TimingStats* statsOut = nullptr) {
    TimingStats stats = MeasureWithWarmup(repeats, [&]() {
        QR qr(A);
        qr.qrIMGS();
        q = qr.Q();
        r = qr.R();
    });
    if (statsOut) {
        *statsOut = stats;
    }
    return stats.avgMs;
}

double TimeSparseQrMs(const SparseMatrix<double>& A, int repeats,
                      Matrix<double>& q, Matrix<double>& r,
                      std::vector<size_t>& perm,
                      TimingStats* statsOut = nullptr) {
    TimingStats stats = MeasureWithWarmup(repeats, [&]() {
        SparseQR qr(A);
        qr.qr();
        q = qr.Q();
        r = qr.R();
        perm = qr.colsPermutation();
    });
    if (statsOut) {
        *statsOut = stats;
    }
    return stats.avgMs;
}

double TimeEigenSparseQrMs(const Eigen::SparseMatrix<double>& A, int repeats,
                           Matrix<double>& qOut, Matrix<double>& rOut,
                           TimingStats* statsOut = nullptr) {
    TimingStats stats = MeasureWithWarmup(repeats, [&]() {
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
        qr.compute(A);
        const int thin_cols = std::min(A.rows(), A.cols());
        Eigen::MatrixXd thinIdentity = Eigen::MatrixXd::Identity(A.rows(), thin_cols);
        Eigen::MatrixXd q = qr.matrixQ() * thinIdentity;
        Eigen::MatrixXd r = Eigen::MatrixXd(qr.matrixR()).topRows(thin_cols);
        qOut = DenseFromEigen(q);
        rOut = DenseFromEigen(r);
    });
    if (statsOut) {
        *statsOut = stats;
    }
    return stats.avgMs;
}

double TimeSparseSolveOnlyMs(SparseQR& qr, const Matrix<double>& b, int repeats, Matrix<double>& xOut,
                             TimingStats* statsOut = nullptr) {
    TimingStats stats = MeasureWithWarmup(repeats, [&]() {
        xOut = qr.solve(b);
    });
    if (statsOut) {
        *statsOut = stats;
    }
    return stats.avgMs;
}

double TimeEigenSolveOnlyMs(Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>& qr,
                            const Eigen::MatrixXd& b,
                            int repeats,
                            Matrix<double>& xOut,
                            TimingStats* statsOut = nullptr) {
    Eigen::MatrixXd x;
    TimingStats stats = MeasureWithWarmup(repeats, [&]() {
        x = qr.solve(b);
    });
    if (statsOut) {
        *statsOut = stats;
    }
    xOut = DenseFromEigen(x);
    return stats.avgMs;
}

struct BenchResult {
    double denseMs;
    double sparseMs;
    double eigenMs;
};

bool CheckEigenExplicitReconstruction(const Matrix<double>& dense,
                                      const Eigen::SparseMatrix<double>& eigenSparse) {
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
    qr.compute(eigenSparse);
    const int thin_cols = std::min(eigenSparse.rows(), eigenSparse.cols());
    Eigen::MatrixXd thinIdentity = Eigen::MatrixXd::Identity(eigenSparse.rows(), thin_cols);
    Eigen::MatrixXd q = qr.matrixQ() * thinIdentity;
    Eigen::MatrixXd r = Eigen::MatrixXd(qr.matrixR()).topRows(thin_cols);
    Eigen::MatrixXd ap = EigenFromDense(dense) * qr.colsPermutation();
    return DenseApproxEqual(DenseFromEigen(q * r), DenseFromEigen(ap), 1e-6);
}

BenchResult RunCase(size_t rows, size_t cols, double density, int repeats, uint32_t seed) {
    Matrix<double> dense = BuildSparseLikeDense(rows, cols, density, seed);
    SparseMatrix<double> sparse(dense);
    Eigen::SparseMatrix<double> eigenSparse = ToEigenSparse(sparse);
    const MatrixStats stats = GetMatrixStats(sparse);

    Matrix<double> denseQ, denseR, sparseQ, sparseR, eigenQ, eigenR;
    std::vector<size_t> sparsePerm;
    TimingStats denseStats, sparseStats, eigenStats;

    const double denseMs = TimeDenseQrMs(dense, repeats, denseQ, denseR, &denseStats);
    const double sparseMs = TimeSparseQrMs(sparse, repeats, sparseQ, sparseR, sparsePerm, &sparseStats);
    const double eigenMs = TimeEigenSparseQrMs(eigenSparse, repeats, eigenQ, eigenR, &eigenStats);

    EXPECT_TRUE(DenseApproxEqual(denseQ * denseR, dense, 1e-6));

    Matrix<double> AP_sparse(dense.rows_size(), dense.cols_size());
    for (size_t j = 0; j < dense.cols_size(); ++j)
        for (size_t i = 0; i < dense.rows_size(); ++i)
            AP_sparse(i, j) = dense(i, sparsePerm[j]);
    EXPECT_TRUE(DenseApproxEqual(sparseQ * sparseR, AP_sparse, 1e-6));
    EXPECT_TRUE(CheckEigenExplicitReconstruction(dense, eigenSparse));

    std::cout << "explicit_factors_bench"
              << " rows=" << rows
              << " cols=" << cols
              << " requested_density=" << density
              << " nnz=" << stats.nnz
              << " actual_density=" << stats.actualDensity
              << " repeats=" << repeats
              << " dense_avg_ms=" << denseMs
              << " dense_median_ms=" << denseStats.medianMs
              << " sparse_avg_ms=" << sparseMs
              << " sparse_median_ms=" << sparseStats.medianMs
              << " eigen_avg_ms=" << eigenMs
              << " eigen_median_ms=" << eigenStats.medianMs
              << " ratio_avg=" << RatioMyToEigen(sparseMs, eigenMs)
              << " ratio_median=" << RatioMyToEigen(sparseStats.medianMs, eigenStats.medianMs)
              << std::endl;

    return {denseMs, sparseMs, eigenMs};
}

void RunFactorizationBenchForMatrix(const Matrix<double>& dense,
                                    int repeats,
                                    const std::string& sourceLabel,
                                    const std::string& sourceValue) {
    SparseMatrix<double> sparse(dense);
    Eigen::SparseMatrix<double> eigenSparse = ToEigenSparse(sparse);
    const MatrixStats stats = GetMatrixStats(sparse);

    auto split = TimeSparseQrSplit(sparse, repeats);
    TimingStats eigenStats;
    double eigenMs = TimeEigenQrMs(eigenSparse, repeats, &eigenStats);

    std::cout << "factorization_bench"
              << " scenario=analyze_plus_factorize"
              << " rows=" << dense.rows_size()
              << " cols=" << dense.cols_size()
              << " " << sourceLabel << "=" << sourceValue
              << " nnz=" << stats.nnz
              << " actual_density=" << stats.actualDensity
              << " repeats=" << repeats
              << " sparse_analyze_avg_ms=" << split.analyze.avgMs
              << " sparse_analyze_median_ms=" << split.analyze.medianMs
              << " sparse_factorize_avg_ms=" << split.factorize.avgMs
              << " sparse_factorize_median_ms=" << split.factorize.medianMs
              << " sparse_total_avg_ms=" << split.total.avgMs
              << " sparse_total_median_ms=" << split.total.medianMs
              << " eigen_compute_avg_ms=" << eigenMs
              << " eigen_compute_median_ms=" << eigenStats.medianMs
              << " ratio_avg=" << RatioMyToEigen(split.total.avgMs, eigenMs)
              << " ratio_median=" << RatioMyToEigen(split.total.medianMs, eigenStats.medianMs)
              << std::endl;
}

void RunFactorizationBench(size_t rows, size_t cols, double density,
                           int repeats, uint32_t seed) {
    Matrix<double> dense = BuildSparseLikeDense(rows, cols, density, seed);
    RunFactorizationBenchForMatrix(dense, repeats, "requested_density", std::to_string(density));
}

void RunGeometryFactorizationBench(size_t rows, size_t cols, size_t rowNnz,
                                   int repeats, uint32_t seed) {
    Matrix<double> dense = BuildGeometryLikeMatrix(rows, cols, rowNnz, seed);
    RunFactorizationBenchForMatrix(dense, repeats, "row_nnz", std::to_string(rowNnz));
}

void RunSolveCase(size_t rows, size_t cols, size_t rhsCols, double density, int repeats,
                  uint32_t seedA, uint32_t seedB)
{
    Matrix<double> denseA =
        BuildFullColumnRankMatrix(rows, cols, RowNnzFromDensity(cols, density), seedA);
    SparseMatrix<double> sparseA(denseA);
    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparseA);
    const MatrixStats stats = GetMatrixStats(sparseA);
    Matrix<double> b = BuildDenseRhs(rows, rhsCols, seedB);

    SparseQR sparseQr(sparseA);
    sparseQr.qr();

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);

    const Eigen::MatrixXd bEigen = EigenFromDense(b);

    Matrix<double> xSparse, xEigen;
    TimingStats sparseSolveStats, eigenSolveStats;

    const double sparseSolveMs =
        TimeSparseSolveOnlyMs(sparseQr, b, repeats, xSparse, &sparseSolveStats);
    const double eigenSolveMs =
        TimeEigenSolveOnlyMs(eigenQr, bEigen, repeats, xEigen, &eigenSolveStats);

    const double sparseResidual = ResidualNorm(denseA, xSparse, b);
    const double eigenResidual = ResidualNorm(denseA, xEigen, b);
    const double residualDiff =
        std::abs(sparseResidual - eigenResidual) / std::max(1.0, eigenResidual);

    EXPECT_GT(sparseSolveMs, 0.0);
    EXPECT_GT(eigenSolveMs, 0.0);
    EXPECT_TRUE(std::isfinite(sparseResidual));
    EXPECT_TRUE(std::isfinite(eigenResidual));
    EXPECT_LE(std::abs(sparseResidual - eigenResidual),
              kResidualCompareTolerance * std::max(1.0, eigenResidual));

    std::cout << "solve_bench"
              << " scenario=least_squares_random_rhs"
              << " rows=" << rows
              << " cols=" << cols
              << " requested_density=" << density
              << " nnz=" << stats.nnz
              << " actual_density=" << stats.actualDensity
              << " rhs_cols=" << rhsCols
              << " repeats=" << repeats
              << " sparse_avg_ms=" << sparseSolveMs
              << " sparse_median_ms=" << sparseSolveStats.medianMs
              << " eigen_avg_ms=" << eigenSolveMs
              << " eigen_median_ms=" << eigenSolveStats.medianMs
              << " ratio_avg=" << RatioMyToEigen(sparseSolveMs, eigenSolveMs)
              << " ratio_median=" << RatioMyToEigen(sparseSolveStats.medianMs, eigenSolveStats.medianMs)
              << " sparse_residual=" << sparseResidual
              << " eigen_residual=" << eigenResidual
              << " residual_relative_diff=" << residualDiff
              << std::endl;
}

void RunRepeatedSolveBench(size_t rows, size_t cols, size_t rhsCols,
                           double density, int numSolves,
                           uint32_t seedA, uint32_t seedB) {
    Matrix<double> denseA =
        BuildFullColumnRankMatrix(rows, cols, RowNnzFromDensity(cols, density), seedA);
    SparseMatrix<double> sparseA(denseA);
    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparseA);
    const MatrixStats stats = GetMatrixStats(sparseA);

    const int factorRepeats = rows >= 2000 ? 3 : (rows >= 1000 ? 5 : 10);
    auto sparseFactorStats = MeasureWithWarmup(factorRepeats, [&]() {
        SparseQR qr(sparseA);
        qr.qr();
    });
    auto eigenFactorStats = MeasureWithWarmup(factorRepeats, [&]() {
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
        qr.compute(eigenA);
    });

    SparseQR sparseQr(sparseA);
    sparseQr.qr();
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);

    std::vector<Matrix<double>> rhs;
    std::vector<Eigen::MatrixXd> rhsEigen;
    rhs.reserve(static_cast<size_t>(numSolves));
    rhsEigen.reserve(static_cast<size_t>(numSolves));
    for (int s = 0; s < numSolves; ++s) {
        rhs.push_back(BuildDenseRhs(rows, rhsCols, seedB + static_cast<uint32_t>(s)));
        rhsEigen.push_back(EigenFromDense(rhs.back()));
    }

    (void)sparseQr.solve(rhs.front());
    (void)eigenQr.solve(rhsEigen.front());

    std::vector<Matrix<double>> sparseSolutions;
    std::vector<Matrix<double>> eigenSolutions;
    std::vector<double> sparseSolveSamples;
    std::vector<double> eigenSolveSamples;
    sparseSolutions.reserve(static_cast<size_t>(numSolves));
    eigenSolutions.reserve(static_cast<size_t>(numSolves));
    sparseSolveSamples.reserve(static_cast<size_t>(numSolves));
    eigenSolveSamples.reserve(static_cast<size_t>(numSolves));

    for (int s = 0; s < numSolves; ++s) {
        auto t0 = std::chrono::steady_clock::now();
        Matrix<double> xs = sparseQr.solve(rhs[static_cast<size_t>(s)]);
        auto t1 = std::chrono::steady_clock::now();
        sparseSolveSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        sparseSolutions.push_back(std::move(xs));
    }

    for (int s = 0; s < numSolves; ++s) {
        auto t0 = std::chrono::steady_clock::now();
        Eigen::MatrixXd xe = eigenQr.solve(rhsEigen[static_cast<size_t>(s)]);
        auto t1 = std::chrono::steady_clock::now();
        eigenSolveSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        eigenSolutions.push_back(DenseFromEigen(xe));
    }

    double maxResidualRelativeDiff = 0.0;
    double sumResidualRelativeDiff = 0.0;
    for (int s = 0; s < numSolves; ++s) {
        const Matrix<double>& b = rhs[static_cast<size_t>(s)];
        const double sparseResidual =
            ResidualNorm(denseA, sparseSolutions[static_cast<size_t>(s)], b);
        const double eigenResidual =
            ResidualNorm(denseA, eigenSolutions[static_cast<size_t>(s)], b);
        const double relativeDiff =
            std::abs(sparseResidual - eigenResidual) / std::max(1.0, eigenResidual);
        maxResidualRelativeDiff = std::max(maxResidualRelativeDiff, relativeDiff);
        sumResidualRelativeDiff += relativeDiff;

        EXPECT_LE(std::abs(sparseResidual - eigenResidual),
                  kResidualCompareTolerance * std::max(1.0, eigenResidual));
    }

    const TimingStats sparseSolveStats = StatsFromSamples(std::move(sparseSolveSamples));
    const TimingStats eigenSolveStats = StatsFromSamples(std::move(eigenSolveSamples));
    const double avgResidualRelativeDiff =
        sumResidualRelativeDiff / static_cast<double>(numSolves);

    std::cout << "repeated_solve_bench"
              << " scenario=factor_once_solve_many"
              << " rows=" << rows
              << " cols=" << cols
              << " requested_density=" << density
              << " nnz=" << stats.nnz
              << " actual_density=" << stats.actualDensity
              << " rhs_cols=" << rhsCols
              << " solve_repeats=" << numSolves
              << " factor_repeats=" << factorRepeats
              << " sparse_factor_avg_ms=" << sparseFactorStats.avgMs
              << " sparse_factor_median_ms=" << sparseFactorStats.medianMs
              << " eigen_factor_avg_ms=" << eigenFactorStats.avgMs
              << " eigen_factor_median_ms=" << eigenFactorStats.medianMs
              << " sparse_solve_avg_ms=" << sparseSolveStats.avgMs
              << " sparse_solve_median_ms=" << sparseSolveStats.medianMs
              << " eigen_solve_avg_ms=" << eigenSolveStats.avgMs
              << " eigen_solve_median_ms=" << eigenSolveStats.medianMs
              << " ratio_avg="
              << RatioMyToEigen(
                     sparseFactorStats.avgMs + sparseSolveStats.avgMs * numSolves,
                     eigenFactorStats.avgMs + eigenSolveStats.avgMs * numSolves)
              << " max_residual_relative_diff=" << maxResidualRelativeDiff
              << " avg_residual_relative_diff=" << avgResidualRelativeDiff
              << std::endl;
}

void RunKnownSolutionCase(size_t rows, size_t cols, size_t rhsCols,
                          size_t rowNnz, uint32_t seedA, uint32_t seedX) {
    Matrix<double> denseA = BuildFullColumnRankMatrix(rows, cols, rowNnz, seedA);
    SparseMatrix<double> sparseA(denseA);
    Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparseA);
    const MatrixStats stats = GetMatrixStats(sparseA);

    Matrix<double> xExpected = BuildDenseRhs(cols, rhsCols, seedX);
    Matrix<double> b = denseA * xExpected;
    const Eigen::MatrixXd bEigen = EigenFromDense(b);

    SparseQR sparseQr(sparseA);
    sparseQr.qr();
    Matrix<double> xSparse = sparseQr.solve(b, 0.0);

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);
    Matrix<double> xEigen = DenseFromEigen(eigenQr.solve(bEigen));

    const double sparseRelativeResidual =
        ResidualNorm(denseA, xSparse, b) / std::max(1.0, b.norm());
    const double eigenRelativeResidual =
        ResidualNorm(denseA, xEigen, b) / std::max(1.0, b.norm());
    const double sparseXError = RelativeMatrixError(xSparse, xExpected);
    const double eigenXError = RelativeMatrixError(xEigen, xExpected);
    const double solverDiff = RelativeMatrixError(xSparse, xEigen);

    EXPECT_EQ(sparseQr.rank(), cols);
    EXPECT_EQ(static_cast<size_t>(eigenQr.rank()), cols);
    EXPECT_LE(sparseRelativeResidual, kKnownSolutionTolerance);
    EXPECT_LE(eigenRelativeResidual, kKnownSolutionTolerance);
    EXPECT_LE(sparseXError, kKnownSolutionTolerance);
    EXPECT_LE(eigenXError, kKnownSolutionTolerance);
    EXPECT_LE(solverDiff, kKnownSolutionTolerance);

    std::cout << "solve_correctness"
              << " scenario=known_full_rank_solution"
              << " rows=" << rows
              << " cols=" << cols
              << " row_nnz=" << rowNnz
              << " nnz=" << stats.nnz
              << " actual_density=" << stats.actualDensity
              << " rhs_cols=" << rhsCols
              << " sparse_rank=" << sparseQr.rank()
              << " eigen_rank=" << eigenQr.rank()
              << " sparse_relative_residual=" << sparseRelativeResidual
              << " eigen_relative_residual=" << eigenRelativeResidual
              << " sparse_x_relative_error=" << sparseXError
              << " eigen_x_relative_error=" << eigenXError
              << " sparse_vs_eigen_x_relative_diff=" << solverDiff
              << std::endl;
}

void CheckFullRankSolveAgainstKnownSolution(const Matrix<double>& denseA,
                                            const SparseMatrix<double>& sparseA,
                                            const Eigen::SparseMatrix<double>& eigenA,
                                            uint32_t seedX) {
    Matrix<double> xExpected = BuildDenseRhs(denseA.cols_size(), 1, seedX);
    Matrix<double> b = denseA * xExpected;

    SparseQR sparseQr(sparseA);
    sparseQr.qr();
    Matrix<double> xSparse = sparseQr.solve(b, 0.0);

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenA);
    Matrix<double> xEigen = DenseFromEigen(eigenQr.solve(EigenFromDense(b)));

    EXPECT_LE(ResidualNorm(denseA, xSparse, b) / std::max(1.0, b.norm()),
              kKnownSolutionTolerance);
    EXPECT_LE(RelativeMatrixError(xSparse, xExpected), kKnownSolutionTolerance);
    EXPECT_LE(RelativeMatrixError(xSparse, xEigen), kKnownSolutionTolerance);
}

void RunReuseStructuralAnalysisBench(size_t rows, size_t cols, size_t rowNnz,
                                     int matrixCount, uint32_t seedPattern) {
    Matrix<double> pattern = BuildFullColumnRankMatrix(rows, cols, rowNnz, seedPattern);
    Matrix<double> warmDense = BuildSamePatternWithNewValues(pattern, seedPattern + 1000u);
    SparseMatrix<double> warmSparse(warmDense);

    std::vector<Matrix<double>> denseMatrices;
    std::vector<SparseMatrix<double>> sparseMatrices;
    denseMatrices.reserve(static_cast<size_t>(matrixCount));
    sparseMatrices.reserve(static_cast<size_t>(matrixCount));
    for (int i = 0; i < matrixCount; ++i) {
        denseMatrices.push_back(
            BuildSamePatternWithNewValues(pattern, seedPattern + 2000u + static_cast<uint32_t>(i)));
        sparseMatrices.emplace_back(denseMatrices.back());
    }

    const MatrixStats stats = GetMatrixStats(sparseMatrices.front());

    {
        SparseQR warmQr(warmSparse);
        warmQr.analyze();
        warmQr.factorize();
    }

    std::vector<double> freshSamples;
    freshSamples.reserve(static_cast<size_t>(matrixCount));
    for (const SparseMatrix<double>& sparse : sparseMatrices) {
        const auto t0 = std::chrono::steady_clock::now();
        SparseQR qr(sparse);
        qr.analyze();
        qr.factorize();
        const auto t1 = std::chrono::steady_clock::now();
        freshSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    SparseQR reuseQr(warmSparse);
    reuseQr.analyze();
    reuseQr.factorize(warmSparse);

    std::vector<double> reuseSamples;
    reuseSamples.reserve(static_cast<size_t>(matrixCount));
    for (const SparseMatrix<double>& sparse : sparseMatrices) {
        const auto t0 = std::chrono::steady_clock::now();
        reuseQr.factorize(sparse);
        const auto t1 = std::chrono::steady_clock::now();
        reuseSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const TimingStats freshStats = StatsFromSamples(std::move(freshSamples));
    const TimingStats reuseStats = StatsFromSamples(std::move(reuseSamples));

    for (int i = 0; i < matrixCount; ++i) {
        Eigen::SparseMatrix<double> eigenA = ToEigenSparse(sparseMatrices[static_cast<size_t>(i)]);
        CheckFullRankSolveAgainstKnownSolution(
            denseMatrices[static_cast<size_t>(i)],
            sparseMatrices[static_cast<size_t>(i)],
            eigenA,
            seedPattern + 3000u + static_cast<uint32_t>(i));
    }

    std::cout << "structural_reuse_bench"
              << " scenario=same_pattern_changed_values"
              << " rows=" << rows
              << " cols=" << cols
              << " row_nnz=" << rowNnz
              << " nnz=" << stats.nnz
              << " actual_density=" << stats.actualDensity
              << " matrices=" << matrixCount
              << " fresh_repeats=" << freshStats.repeats
              << " reuse_repeats=" << reuseStats.repeats
              << " fresh_analyze_factorize_avg_ms=" << freshStats.avgMs
              << " fresh_analyze_factorize_median_ms=" << freshStats.medianMs
              << " reuse_numeric_factorize_avg_ms=" << reuseStats.avgMs
              << " reuse_numeric_factorize_median_ms=" << reuseStats.medianMs
              << " ratio_avg=" << RatioMyToEigen(reuseStats.avgMs, freshStats.avgMs)
              << " ratio_median=" << RatioMyToEigen(reuseStats.medianMs, freshStats.medianMs)
              << std::endl;
}

void RunRankDeficientCase(size_t rows, size_t cols, size_t targetRank,
                          double density, int repeats, uint32_t seed) {
    Matrix<double> dense = BuildRankDeficient(rows, cols, targetRank, density, seed);
    SparseMatrix<double> sparse(dense);
    Eigen::SparseMatrix<double> eigenSparse = ToEigenSparse(sparse);
    const MatrixStats stats = GetMatrixStats(sparse);

    auto split = TimeSparseQrSplit(sparse, repeats);
    TimingStats eigenStats;
    double eigenMs = TimeEigenQrMs(eigenSparse, repeats, &eigenStats);

    SparseQR qr(sparse);
    qr.qr();
    size_t r = qr.rank();

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> eigenQr;
    eigenQr.compute(eigenSparse);
    const int eigenRank = eigenQr.rank();

    std::cout << "rank_deficient_bench"
              << " scenario=known_rank_by_column_copies"
              << " rows=" << rows
              << " cols=" << cols
              << " target_rank=" << targetRank
              << " requested_density=" << density
              << " nnz=" << stats.nnz
              << " actual_density=" << stats.actualDensity
              << " repeats=" << repeats
              << " sparse_rank=" << r
              << " eigen_rank=" << eigenRank
              << " sparse_total_avg_ms=" << split.total.avgMs
              << " sparse_total_median_ms=" << split.total.medianMs
              << " eigen_compute_avg_ms=" << eigenMs
              << " eigen_compute_median_ms=" << eigenStats.medianMs
              << " ratio_avg=" << RatioMyToEigen(split.total.avgMs, eigenMs)
              << " ratio_median=" << RatioMyToEigen(split.total.medianMs, eigenStats.medianMs)
              << std::endl;

    EXPECT_EQ(r, targetRank);
    EXPECT_EQ(static_cast<size_t>(eigenRank), targetRank);
    EXPECT_EQ(r, static_cast<size_t>(eigenRank));
}

} // namespace

// ---- Explicit Q/R reconstruction benchmarks ----

TEST(SparseQRPerformanceTests, Small_80x60) {
    RunCase(80, 60, 0.05, 10, 42u);
}

TEST(SparseQRPerformanceTests, Small_120x90) {
    RunCase(120, 90, 0.03, 10, 1337u);
}

TEST(SparseQRPerformanceTests, Medium_200x150) {
    RunCase(200, 150, 0.03, 10, 7u);
}

TEST(SparseQRPerformanceTests, Medium_300x200) {
    RunCase(300, 200, 0.025, 10, 2026u);
}

TEST(SparseQRPerformanceTests, Large_500x350) {
    RunCase(500, 350, 0.02, 5, 99u);
}

TEST(SparseQRPerformanceTests, Large_700x500) {
    RunCase(700, 500, 0.015, 3, 555u);
}

TEST(SparseQRPerformanceTests, XLarge_1000x700) {
    RunCase(1000, 700, 0.01, 3, 12345u);
}

// ---- Factorization-only (analyze + factorize split timing) ----

TEST(SparseQRPerformanceTests, FactorizeSplit_500x350_d002) {
    RunFactorizationBench(500, 350, 0.02, 10, 99u);
}

TEST(SparseQRPerformanceTests, FactorizeSplit_1000x700_d001) {
    RunFactorizationBench(1000, 700, 0.01, 5, 12345u);
}

TEST(SparseQRPerformanceTests, FactorizeSplit_2000x1400_d005) {
    RunFactorizationBench(2000, 1400, 0.005, 3, 22222u);
}

// ---- Geometry-like row nnz sweep ----

TEST(SparseQRPerformanceTests, RowNnzSweep_500x350_2) {
    RunGeometryFactorizationBench(500, 350, 2, 10, 1001u);
}

TEST(SparseQRPerformanceTests, RowNnzSweep_500x350_4) {
    RunGeometryFactorizationBench(500, 350, 4, 10, 1002u);
}

TEST(SparseQRPerformanceTests, RowNnzSweep_500x350_6) {
    RunGeometryFactorizationBench(500, 350, 6, 10, 1003u);
}

TEST(SparseQRPerformanceTests, RowNnzSweep_500x350_8) {
    RunGeometryFactorizationBench(500, 350, 8, 10, 1004u);
}

TEST(SparseQRPerformanceTests, RowNnzSweep_500x350_12) {
    RunGeometryFactorizationBench(500, 350, 12, 10, 1005u);
}

// ---- Different aspect ratios ----

TEST(SparseQRPerformanceTests, Aspect_1000x100_d005) {
    RunFactorizationBench(1000, 100, 0.05, 10, 2001u);
}

TEST(SparseQRPerformanceTests, Aspect_1000x500_d005) {
    RunFactorizationBench(1000, 500, 0.005, 5, 2002u);
}

TEST(SparseQRPerformanceTests, Aspect_1000x900_d003) {
    RunFactorizationBench(1000, 900, 0.003, 3, 2003u);
}

TEST(SparseQRPerformanceTests, Aspect_500x500_d01) {
    RunFactorizationBench(500, 500, 0.01, 5, 2004u);
}

// ---- Solve benchmarks ----

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_300x200) {
    RunSolveCase(300, 200, 4, 0.03, 10, 2026u, 2027u);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_500x350) {
    RunSolveCase(500, 350, 3, 0.02, 10, 99u, 100u);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_700x500) {
    RunSolveCase(700, 500, 3, 0.015, 5, 555u, 556u);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_1000x700) {
    RunSolveCase(1000, 700, 2, 0.01, 5, 12345u, 12346u);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_2000x1400) {
    RunSolveCase(2000, 1400, 2, 0.005, 3, 22222u, 22223u);
}

TEST(SparseQRPerformanceTests, SolveComparison_LeastSquares_3000x2000) {
    RunSolveCase(3000, 2000, 1, 0.003, 3, 33333u, 33334u);
}

// ---- Solve correctness with known exact solution ----

TEST(SparseQRPerformanceTests, SolveCorrectness_KnownFullRank_300x200) {
    RunKnownSolutionCase(300, 200, 3, 8, 4242u, 4243u);
}

TEST(SparseQRPerformanceTests, SolveCorrectness_KnownFullRank_1000x700) {
    RunKnownSolutionCase(1000, 700, 2, 8, 5252u, 5253u);
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
    RunRepeatedSolveBench(500, 350, 4, 0.02, 10, 99u, 101u);
}

// ---- Structural analysis reuse ----

TEST(SparseQRPerformanceTests, ReuseStructuralAnalysis_1000x700) {
    RunReuseStructuralAnalysisBench(1000, 700, 8, 5, 12345u);
}

// ---- Rank-deficient factorization timing ----

TEST(SparseQRPerformanceTests, RankDeficient_500x200_rank100) {
    RunRankDeficientCase(500, 200, 100, 0.05, 10, 9876u);
}

TEST(SparseQRPerformanceTests, RankDeficient_1000x500_rank200) {
    RunRankDeficientCase(1000, 500, 200, 0.02, 5, 5555u);
}
