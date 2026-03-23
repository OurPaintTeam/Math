#include <cmath>
#include <vector>
#include <algorithm>

#include "gtest/gtest.h"
#include "SparseMatrix.h"

namespace {

bool DenseEqual(const Matrix<double>& A, const Matrix<double>& B, double eps = 1e-12) {
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

} // namespace

TEST(SparseMatrixTests, defaultConstructor) {
    SparseMatrix<> A;
    EXPECT_EQ(A.rows_size(), 0);
    EXPECT_EQ(A.cols_size(), 0);
    EXPECT_EQ(A.nonZeros(), 0);
    EXPECT_THROW(A(0, 0), std::out_of_range);
}

TEST(SparseMatrixTests, sizeConstructors) {
    SparseMatrix<> A(3);
    EXPECT_EQ(A.rows_size(), 3);
    EXPECT_EQ(A.cols_size(), 3);
    EXPECT_EQ(A.nonZeros(), 0);

    SparseMatrix<> B(2, 7);
    EXPECT_EQ(B.rows_size(), 2);
    EXPECT_EQ(B.cols_size(), 7);
    EXPECT_EQ(B.nonZeros(), 0);
}

TEST(SparseMatrixTests, denseConversionRoundTrip) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0},
        {0.0, 0.0, 0.0},
        {3.0, 4.0, 0.0}
    };
    SparseMatrix<> sp(dense);
    EXPECT_EQ(sp.rows_size(), 3);
    EXPECT_EQ(sp.cols_size(), 3);
    EXPECT_EQ(sp.nonZeros(), 4);

    Matrix<double> restored = sp.toDense();
    EXPECT_TRUE(DenseEqual(restored, dense));
}

TEST(SparseMatrixTests, insertCoeffAndCoeffRef) {
    SparseMatrix<> sp(4, 4);

    sp.insert(0, 1, 2.5);
    sp.insert(3, 3, -1.0);
    EXPECT_EQ(sp(0, 1), 2.5);
    EXPECT_EQ(sp.coeff(3, 3), -1.0);
    EXPECT_EQ(sp(2, 2), 0.0);

    sp.coeffRef(2, 0) = 9.0;
    EXPECT_EQ(sp(2, 0), 9.0);

    sp.coeffRef(0, 1) = 10.0;
    EXPECT_EQ(sp(0, 1), 10.0);
}

TEST(SparseMatrixTests, setFromTripletsWithDuplicates) {
    SparseMatrix<> sp(3, 3);
    std::vector<Triplet<double>> triplets = {
        {0, 0, 1.0},
        {0, 2, 3.0},
        {1, 1, 2.0},
        {0, 2, 4.5},
        {2, 0, -1.0}
    };

    sp.setFromTriplets(triplets.begin(), triplets.end());
    EXPECT_EQ(sp.nonZeros(), 4);
    EXPECT_DOUBLE_EQ(sp(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(sp(0, 2), 7.5);
    EXPECT_DOUBLE_EQ(sp(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(sp(2, 0), -1.0);
}

TEST(SparseMatrixTests, sparseAddSubMul) {
    Matrix<double> A_dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 0.0},
        {4.0, 0.0, 0.0}
    };
    Matrix<double> B_dense = {
        {0.0, 5.0, 0.0},
        {6.0, 0.0, 0.0},
        {0.0, 0.0, 7.0}
    };
    SparseMatrix<> A(A_dense);
    SparseMatrix<> B(B_dense);

    SparseMatrix<> C_add = A + B;
    SparseMatrix<> C_sub = A - B;
    SparseMatrix<> C_mul = A * B;

    EXPECT_TRUE(DenseEqual(C_add.toDense(), A_dense + B_dense));
    EXPECT_TRUE(DenseEqual(C_sub.toDense(), A_dense - B_dense));
    EXPECT_TRUE(DenseEqual(C_mul.toDense(), A_dense * B_dense));
}

TEST(SparseMatrixTests, scalarOperationsAndCompound) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 0.0},
        {4.0, 0.0, 0.0}
    };
    SparseMatrix<> A(dense);

    SparseMatrix<> B = A * 2.0;
    SparseMatrix<> C = B / 2.0;
    EXPECT_TRUE(DenseEqual(B.toDense(), dense * 2.0));
    EXPECT_TRUE(DenseEqual(C.toDense(), dense));

    SparseMatrix<> D = A;
    D *= 3.0;
    D /= 3.0;
    EXPECT_TRUE(DenseEqual(D.toDense(), dense));
}

TEST(SparseMatrixTests, transposeTraceDiagNorm) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 0.0},
        {4.0, 0.0, -5.0}
    };
    SparseMatrix<> A(dense);

    SparseMatrix<> At = A.transpose();
    Matrix<double> expectedT = dense.transpose();
    EXPECT_TRUE(DenseEqual(At.toDense(), expectedT));

    A.setTranspose();
    EXPECT_TRUE(DenseEqual(A.toDense(), expectedT));

    SparseMatrix<> B(dense);
    EXPECT_DOUBLE_EQ(B.trace(), -1.0);
    auto d = B.diag();
    ASSERT_EQ(d.size(), 3);
    EXPECT_DOUBLE_EQ(d[0], 1.0);
    EXPECT_DOUBLE_EQ(d[1], 3.0);
    EXPECT_DOUBLE_EQ(d[2], -5.0);

    double expectedNorm = std::sqrt(1.0 + 4.0 + 9.0 + 16.0 + 25.0);
    EXPECT_NEAR(B.norm(), expectedNorm, 1e-12);
}

TEST(SparseMatrixTests, getRowGetColAndSubmatrix) {
    Matrix<double> dense = {
        {1.0, 0.0, 2.0, 0.0},
        {0.0, 3.0, 0.0, 4.0},
        {5.0, 0.0, 6.0, 0.0}
    };
    SparseMatrix<> A(dense);

    auto row1 = A.getRow(1);
    ASSERT_EQ(row1.size(), 4);
    EXPECT_DOUBLE_EQ(row1[0], 0.0);
    EXPECT_DOUBLE_EQ(row1[1], 3.0);
    EXPECT_DOUBLE_EQ(row1[2], 0.0);
    EXPECT_DOUBLE_EQ(row1[3], 4.0);

    auto col2 = A.getCol(2);
    ASSERT_EQ(col2.size(), 3);
    EXPECT_DOUBLE_EQ(col2[0], 2.0);
    EXPECT_DOUBLE_EQ(col2[1], 0.0);
    EXPECT_DOUBLE_EQ(col2[2], 6.0);

    SparseMatrix<> sub = A.getSubmatrix(1, 1, 2, 2);
    Matrix<double> expectedSub = {
        {3.0, 0.0},
        {0.0, 6.0}
    };
    EXPECT_TRUE(DenseEqual(sub.toDense(), expectedSub));
}

TEST(SparseMatrixTests, sparseDenseProducts) {
    Matrix<double> A_dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 0.0}
    };
    Matrix<double> B_dense = {
        {0.0, 4.0},
        {5.0, 0.0},
        {0.0, 6.0}
    };
    SparseMatrix<> A_sp(A_dense);
    SparseMatrix<> B_sp(B_dense);

    Matrix<double> left = A_sp * B_dense;
    Matrix<double> right = A_dense * B_sp;
    Matrix<double> expected = A_dense * B_dense;

    EXPECT_TRUE(DenseEqual(left, expected));
    EXPECT_TRUE(DenseEqual(right, expected));
}

TEST(SparseMatrixTests, identityAndZeroes) {
    SparseMatrix<> I = SparseMatrix<>::identity(4);
    SparseMatrix<> Z = SparseMatrix<>::zeroes(4, 4);

    Matrix<double> expectedI = Matrix<>::identity(4);
    Matrix<double> expectedZ = Matrix<>::zeroes(4, 4);

    EXPECT_TRUE(DenseEqual(I.toDense(), expectedI));
    EXPECT_TRUE(DenseEqual(Z.toDense(), expectedZ));
}
