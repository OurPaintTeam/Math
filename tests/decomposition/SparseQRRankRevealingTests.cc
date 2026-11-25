#include "SparseQR.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <random>

namespace {

void expectFactorization(const SparseQR& qr, size_t rank, double tolerance = 1e-9) {
    ASSERT_EQ(qr.rank(), rank);
    const auto a = qr.A().toDense();
    const auto q = qr.Q();
    const auto r = qr.R();
    const auto reconstructed = q * r;
    const auto& permutation = qr.colsPermutation();
    auto sorted = permutation;
    std::sort(sorted.begin(), sorted.end());
    for (size_t column = 0; column < a.cols_size(); ++column) {
        ASSERT_EQ(sorted[column], column);
        for (size_t row = 0; row < a.rows_size(); ++row)
            EXPECT_NEAR(reconstructed(row, column), a(row, permutation[column]), tolerance);
    }
    for (size_t index = 0; index < rank; ++index) EXPECT_GT(std::abs(r(index, index)), 1e-12);
    const auto identity = q.transpose() * q;
    for (size_t row = 0; row < identity.rows_size(); ++row)
        for (size_t column = 0; column < identity.cols_size(); ++column)
            EXPECT_NEAR(identity(row, column), row == column ? 1 : 0, tolerance);
}

TEST(SparseQRRankRevealing, IndependentColumnAfterDuplicateAndZeroColumns) {
    for (bool zero : {false, true}) {
        Matrix<> a(2, 3);
        a(0, 0) = a(1, 0) = zero ? 0 : 1;
        a(0, 1) = a(1, 1) = a(0, 2) = 1;
        a(1, 2) = 2;
        SparseQR qr{SparseMatrix<>(a)};
        qr.qrRankRevealing();
        expectFactorization(qr, 2);
    }
}

TEST(SparseQRRankRevealing, PreservesKnownRanksUnderRowAndColumnPermutations) {
    std::mt19937 random(47);
    for (size_t rows = 1; rows <= 8; ++rows) {
        for (size_t columns = 1; columns <= 8; ++columns) {
            const auto rank = random() % (std::min(rows, columns) + 1);
            Matrix<> left(rows, rank), right(rank, columns);
            for (size_t row = 0; row < rows; ++row)
                for (size_t column = 0; column < rank; ++column)
                    left(row, column) = row < rank ? (row == column ? 1 : 0) : static_cast<int>(random() % 5) - 2;
            for (size_t row = 0; row < rank; ++row)
                for (size_t column = 0; column < columns; ++column)
                    right(row, column) = column < rank ? (row == column ? 1 : 0) : static_cast<int>(random() % 5) - 2;
            const auto product = left * right;
            std::vector<size_t> rowOrder(rows), columnOrder(columns);
            std::iota(rowOrder.begin(), rowOrder.end(), 0);
            std::iota(columnOrder.begin(), columnOrder.end(), 0);
            std::shuffle(rowOrder.begin(), rowOrder.end(), random);
            std::shuffle(columnOrder.begin(), columnOrder.end(), random);
            Matrix<> a(rows, columns);
            for (size_t row = 0; row < rows; ++row)
                for (size_t column = 0; column < columns; ++column)
                    a(row, column) = product(rowOrder[row], columnOrder[column]);
            SparseQR qr{SparseMatrix<>(a)};
            qr.setPivotThreshold(1e-10);
            qr.qrRankRevealing();
            expectFactorization(qr, rank);
        }
    }
}

TEST(SparseQRRankRevealing, ReusesSymbolicOrderingAfterNumericalPermutationAndRankChanges) {
    Matrix<> a(2, 3);
    a(0, 0) = a(1, 0) = a(0, 1) = a(1, 1) = a(0, 2) = 1;
    a(1, 2) = 2;
    SparseQR qr{SparseMatrix<>(a)};
    qr.qrRankRevealing();
    expectFactorization(qr, 2);
    a(1, 2) = 1;
    qr.setMatrix(SparseMatrix<>(a));
    qr.qrRankRevealing();
    expectFactorization(qr, 1);
    a(1, 1) = 2;
    qr.setMatrix(SparseMatrix<>(a));
    qr.qrRankRevealing();
    expectFactorization(qr, 2);
    SparseQR copy(qr);
    expectFactorization(copy, 2);
    qr.qr();
    EXPECT_EQ(qr.colsPermutation().size(), 3U);
}

TEST(SparseQRRankRevealing, ThresholdCanBeChangedWithoutLosingSymbolicStructure) {
    Matrix<> a(2, 2);
    a(0, 0) = 1;
    a(1, 1) = 1e-10;
    SparseQR qr{SparseMatrix<>(a)};
    qr.setPivotThreshold(1e-8);
    qr.qrRankRevealing();
    expectFactorization(qr, 1);
    qr.setPivotThreshold(1e-12);
    qr.qrRankRevealing();
    expectFactorization(qr, 2);
}

} // namespace
