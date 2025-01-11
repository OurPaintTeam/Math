#include <gtest/gtest.h>
#include <initializer_list>

#include "Matrix.h"

// default constructor
TEST(MatrixTests, defaultConstructor) {
    Matrix<> test;
    EXPECT_EQ(test.rows_size(), 0);
    EXPECT_EQ(test.cols_size(), 0);
    EXPECT_THROW(test(1, 2), std::out_of_range);
    EXPECT_THROW(test(0, 0), std::out_of_range);
}

// constructor(row, col)
TEST(MatrixTests, rowColConstructor) {
    Matrix<> test(3, 5);
    EXPECT_EQ(test.rows_size(), 3);
    EXPECT_EQ(test.cols_size(), 5);
}

// constructor(row, col, value)
TEST(MatrixTests, rowColValConstructor) {
    Matrix<> test(1, 2, 7);
    EXPECT_EQ(test.rows_size(), 1);
    EXPECT_EQ(test.cols_size(), 2);
    EXPECT_EQ(test(0, 0),  7);
    EXPECT_EQ(test(0, 1), 7);
}

// constructor(size)
TEST(MatrixTests, sizeConstructor) {
    Matrix<> test(20);
    EXPECT_EQ(test.rows_size(), 20);
    EXPECT_EQ(test.cols_size(), 20);

    // WARNING! IF I WANT TO USE SIZE CONSTRUCTOR I SHOULD USE DOUBLE OR FLOAT IN TEMPLATE, I COULDN'T USE INT OR SIZE_TYPE IN TEMPLATE, ONLY DOUBLE OR FLOAT
}

// constructor(std::vector)
TEST(MatrixTests, vecConstructor) {
    std::vector<double> vec = { 1, 2, 3 };
    Matrix<> test(vec);
    EXPECT_EQ(test.rows_size(), 1);
    EXPECT_EQ(test.cols_size(), 3);
    EXPECT_EQ(test(0, 0), 1);
    EXPECT_EQ(test(0, 1), 2);
    EXPECT_EQ(test(0, 2), 3);
}

// constructor(std::vector<vector>)
TEST(MatrixTests, vecVecConstructor) {
    std::vector<std::vector<double>> vec = { {1, 2}, {3, 4} };
    Matrix<> test(vec);
    EXPECT_EQ(test.rows_size(), 2);
    EXPECT_EQ(test.cols_size(), 2);
    EXPECT_EQ(test(0, 0), 1);
    EXPECT_EQ(test(0, 1), 2);
    EXPECT_EQ(test(1, 0), 3);
    EXPECT_EQ(test(1, 1), 4);
}

// copy constructor
TEST(MatrixTests, copyConstuctor) {
    Matrix<> A(3, 5);
    Matrix<> B(A);
    EXPECT_TRUE(A == B);
    EXPECT_TRUE(A.rows_size() == B.rows_size());
    EXPECT_EQ(A.cols_size(), B.cols_size());
}

// move constructor
TEST(MatrixTests, moveConstuctor) {
    Matrix<> A(3, 5);
    Matrix<> B(std::move(A));
    EXPECT_EQ(B.rows_size(), 3);
    EXPECT_EQ(B.cols_size(), 5);
}

// Initialize list constructor
TEST(MatrixTests, initializerListListConstuctor) {
    Matrix<> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(0, 1), 2);
    EXPECT_EQ(A(0, 2), 3);

    EXPECT_EQ(A(1, 0), 4);
    EXPECT_EQ(A(1, 1), 5);
    EXPECT_EQ(A(1, 2), 6);

    EXPECT_EQ(A(2, 0), 7);
    EXPECT_EQ(A(2, 1), 8);
    EXPECT_EQ(A(2, 2), 9);
    bool flag = false;
    try
    {
        Matrix<> B = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8}
        };
    }
    catch (...)
    {
        flag = true;
    }
    EXPECT_TRUE(flag);
}

// Initialize list constructor
TEST(MatrixTests, initializerListConstuctor) {
    Matrix<> A = { 1.0, 2.0, 3.0 };
    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(0, 1), 2);
    EXPECT_EQ(A(0, 2), 3);
}

// copy assignment operator
TEST(MatrixTests, copyAssignmentOperator) {
    Matrix<> A(3, 5);
    Matrix<> B = A;
    EXPECT_EQ(A, B);
    EXPECT_EQ(A.rows_size(), B.rows_size());
    EXPECT_EQ(A.cols_size(), B.cols_size());
}

// move assignment operator
TEST(MatrixTests, moveAssignmentOperator) {
    Matrix<> A(3, 5);
    Matrix<> B = std::move(A);
    EXPECT_EQ(B.rows_size(), 3);
    EXPECT_EQ(B.cols_size(), 5);
}

// Initialize list assignment operator
TEST(MatrixTests, initializerListAssignOperator) {
    Matrix<> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    EXPECT_EQ(A(0, 0), 1.0);
    EXPECT_EQ(A(0, 1), 2.0);
    EXPECT_EQ(A(0, 2), 3.0);

    EXPECT_EQ(A(1, 0), 4.0);
    EXPECT_EQ(A(1, 1), 5.0);
    EXPECT_EQ(A(1, 2), 6.0);

    EXPECT_EQ(A(2, 0), 7.0);
    EXPECT_EQ(A(2, 1), 8.0);
    EXPECT_EQ(A(2, 2), 9.0);

    A = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    EXPECT_EQ(A.rows_size(), 2);
    EXPECT_EQ(A.cols_size(), 2);
    EXPECT_EQ(A(0, 0), 1.0);
    EXPECT_EQ(A(0, 1), 2.0);
    EXPECT_EQ(A(1, 0), 3.0);
    EXPECT_EQ(A(1, 1), 4.0);
}

// Matrix + Matrix
TEST(MatrixTests, MatrixSumOperator) {
    Matrix<> A = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    Matrix<> B = {
        {4.0, 3.0},
        {2.0, 1.0}
    };

    Matrix<> C = A + B;
    EXPECT_EQ(C(0, 0), 5.0);
    EXPECT_EQ(C(0, 1), 5.0);
    EXPECT_EQ(C(1, 0), 5.0);
    EXPECT_EQ(C(1, 1), 5.0);
}

// Matrix - Matrix
TEST(MatrixTests, MatrixSubOperator) {
    Matrix<> A = {
        {1, 2},
        {3, 4}
    };
    Matrix<> B = {
        {1, 2},
        {3, 4}
    };

    Matrix<> C = A - B;
    EXPECT_EQ(C(0, 0), 0.0);
    EXPECT_EQ(C(0, 1), 0.0);
    EXPECT_EQ(C(1, 0), 0.0);
    EXPECT_EQ(C(1, 1), 0.0);
}

// Matrix * Matrix
TEST(MatrixTests, MatrixMultiplyOperator) {
    Matrix<> A = {
        {4, 6},
        {1, 3}
    };
    Matrix<> B = {
        {1, 7},
        {2, 5}
    };

    Matrix<> C = A * B;
    EXPECT_EQ(C(0, 0), 16.0);
    EXPECT_EQ(C(0, 1), 58.0);
    EXPECT_EQ(C(1, 0), 7.0);
    EXPECT_EQ(C(1, 1), 22.0);
}

// Matrix += Matrix
TEST(MatrixTests, MatrixSumAssignOperator) {
    Matrix<> A = {
        {1, 2},
        {3, 4}
    };
    Matrix<> B = {
        {4, 3},
        {2, 1}
    };

    A += B;
    EXPECT_EQ(A(0, 0), 5.0);
    EXPECT_EQ(A(0, 1), 5.0);
    EXPECT_EQ(A(1, 0), 5.0);
    EXPECT_EQ(A(1, 1), 5.0);
}


// Matrix -= Matrix
TEST(MatrixTests, MatrixSubAssignOperator) {
    Matrix<> A = {
        {1, 2},
        {3, 4}
    };
    Matrix<> B = {
        {2, 3},
        {4, 5}
    };

    A -= B;
    EXPECT_EQ(A(0, 0), -1.0);
    EXPECT_EQ(A(0, 1), -1.0);
    EXPECT_EQ(A(1, 0), -1.0);
    EXPECT_EQ(A(1, 1), -1.0);
}

// Matrix *= Matrix
TEST(MatrixTests, MatrixMultiplyAssignOperator) {
    Matrix<> A = {
    {4, 6},
    {1, 3}
    };
    Matrix<> B = {
        {1, 7},
        {2, 5}
    };

    A *= B;
    EXPECT_EQ(A(0, 0), 16.0);
    EXPECT_EQ(A(0, 1), 58.0);
    EXPECT_EQ(A(1, 0), 7.0);
    EXPECT_EQ(A(1, 1), 22.0);
}

// Matrix + List
TEST(MatrixTests, MatrixListSumOperator) {
    Matrix<> A = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::initializer_list<std::initializer_list<double>> B = {
        {4.0, 3.0},
        {2.0, 1.0}
    };

    Matrix<> C = A + B;
    EXPECT_EQ(C(0, 0), 5.0);
    EXPECT_EQ(C(0, 1), 5.0);
    EXPECT_EQ(C(1, 0), 5.0);
    EXPECT_EQ(C(1, 1), 5.0);
}

// Matrix - List
TEST(MatrixTests, MatrixListSubOperator) {
    Matrix<> A = {
        {1, 2},
        {3, 4}
    };
    std::initializer_list<std::initializer_list<double>> B = {
        {1, 2},
        {3, 4}
    };

    Matrix<> C = A - B;
    EXPECT_EQ(C(0, 0), 0.0);
    EXPECT_EQ(C(0, 1), 0.0);
    EXPECT_EQ(C(1, 0), 0.0);
    EXPECT_EQ(C(1, 1), 0.0);
}

// Matrix * List
TEST(MatrixTests, MatrixListMultiplyOperator) {
    Matrix<> A = {
        {4, 6},
        {1, 3}
    };
    std::initializer_list<std::initializer_list<double>> B = {
        {1, 7},
        {2, 5}
    };

    Matrix<> C = A * B;
    EXPECT_EQ(C(0, 0), 16.0);
    EXPECT_EQ(C(0, 1), 58.0);
    EXPECT_EQ(C(1, 0), 7.0);
    EXPECT_EQ(C(1, 1), 22.0);
}

// Matrix += List
TEST(MatrixTests, MatrixListSumAssignOperator) {
    Matrix<> A = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::initializer_list<std::initializer_list<double>> B = {
        {4.0, 3.0},
        {2.0, 1.0}
    };

    A += B;
    EXPECT_EQ(A(0, 0), 5.0);
    EXPECT_EQ(A(0, 1), 5.0);
    EXPECT_EQ(A(1, 0), 5.0);
    EXPECT_EQ(A(1, 1), 5.0);
}

// Matrix -= List
TEST(MatrixTests, MatrixListSubAssignOperator) {
    Matrix<> A = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::initializer_list<std::initializer_list<double>> B = {
        {4.0, 3.0},
        {2.0, 1.0}
    };

    A -= B;
    EXPECT_EQ(A(0, 0), -3.0);
    EXPECT_EQ(A(0, 1), -1.0);
    EXPECT_EQ(A(1, 0), 1.0);
    EXPECT_EQ(A(1, 1), 3.0);
}

// Matrix *= List
TEST(MatrixTests, MatrixListMultiplyAssignOperator) {
    Matrix<> A = {
        {4, 6},
        {1, 3}
    };
    std::initializer_list<std::initializer_list<double>> B = {
        {1, 7},
        {2, 5}
    };

    A *= B;
    EXPECT_EQ(A(0, 0), 16.0);
    EXPECT_EQ(A(0, 1), 58.0);
    EXPECT_EQ(A(1, 0), 7.0);
    EXPECT_EQ(A(1, 1), 22.0);
}

// Matrix + scalar ========================
TEST(MatrixTests, SumOperatorMatrixScalar) {
    Matrix<> A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A = A + 2.0;
    EXPECT_EQ(A(0, 0), 3.0);
    EXPECT_EQ(A(0, 1), 4.0);
    EXPECT_EQ(A(1, 0), 5.0);
    EXPECT_EQ(A(1, 1), 6.0);
}

// Matrix - scalar
TEST(MatrixTests, SubOperatorMatrixScalar) {
    Matrix<> A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A = A - 2.0;
    EXPECT_EQ(A(0, 0), -1.0);
    EXPECT_EQ(A(0, 1), 0.0);
    EXPECT_EQ(A(1, 0), 1.0);
    EXPECT_EQ(A(1, 1), 2.0);
}

// Matrix * scalar
TEST(MatrixTests, MultiplyOperatorMatrixScalar) {
    Matrix<> A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A = A * 2.0;
    EXPECT_EQ(A(0, 0), 2.0);
    EXPECT_EQ(A(0, 1), 4.0);
    EXPECT_EQ(A(1, 0), 6.0);
    EXPECT_EQ(A(1, 1), 8.0);
}

// Matrix / scalar
TEST(MatrixTests, DivideOperatorMatrixScalar) {
    Matrix<> A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A = A / 2.0;
    EXPECT_EQ(A(0, 0), 0.5);
    EXPECT_EQ(A(0, 1), 1.0);
    EXPECT_EQ(A(1, 0), 1.5);
    EXPECT_EQ(A(1, 1), 2.0);
}

// Matrix += scalar
TEST(MatrixTests, SumAssignOperatorMatrixScalar) {
    Matrix<> A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A += 2.0;
    EXPECT_EQ(A(0, 0), 3.0);
    EXPECT_EQ(A(0, 1), 4.0);
    EXPECT_EQ(A(1, 0), 5.0);
    EXPECT_EQ(A(1, 1), 6.0);
}

// Matrix -= scalar
TEST(MatrixTests, SubAssignOperatorMatrixScalar) {
    Matrix<> A(2, 2);
    A = { {1.0, 2.0},
          {3.0, 4.0} };
    A -= 2.0;
    EXPECT_EQ(A(0, 0), -1.0);
    EXPECT_EQ(A(0, 1), 0.0);
    EXPECT_EQ(A(1, 0), 1.0);
    EXPECT_EQ(A(1, 1), 2.0);
}

// Matrix *= scalar
TEST(MatrixTests, MultiplyAssignOperatorMatrixScalar) {
    Matrix<> A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A *= 2.0;
    EXPECT_EQ(A(0, 0), 2.0);
    EXPECT_EQ(A(0, 1), 4.0);
    EXPECT_EQ(A(1, 0), 6.0);
    EXPECT_EQ(A(1, 1), 8.0);
}

// Matrix /= scalar
TEST(MatrixTests, DivideAssignOperatorMatrixScalar) {
    Matrix<> A(2, 2);
    A = { {1.0, 2.0}, 
          {3.0, 4.0} };
    A /= 2.0;
    EXPECT_EQ(A(0, 0), 0.5);
    EXPECT_EQ(A(0, 1), 1.0);
    EXPECT_EQ(A(1, 0), 1.5);
    EXPECT_EQ(A(1, 1), 2.0);
}

// operator== Matrix and Matrix
TEST(MatrixTests, EqualityOperatorMM) {
    Matrix<> A(3, 5);
    Matrix<> B(3, 5);
    EXPECT_TRUE(A == B);
}

// operator== Matrix and List
TEST(MatrixTests, EqualityOperatorMI) {
    Matrix<> A = {
        {1, 2},
        {3, 4}
    };
    
    std::initializer_list<std::initializer_list<double>> B = {
        {1, 2},
        {3, 4}
    };
    EXPECT_TRUE(A == B);
}

// operator!= Matrix and Matrix
TEST(MatrixTests, InEqualityOperatorMM) {
    Matrix<> A(3, 5);
    Matrix<> B(3, 4);
    EXPECT_TRUE(A != B);
}

// operator!= Matrix and List
TEST(MatrixTests, InEqualityOperatorMI) {
    Matrix<> A = {
        {1, 2},
        {3, 4}
    };

    std::initializer_list<std::initializer_list<double>> B = {
        {1, 2},
        {3, 5}
    };
    EXPECT_TRUE(A != B);
}

// Access operator
TEST(MatrixTests, accessOperator) {
    Matrix<> A(2, 2);
    A(0, 0) = 1;
    A(1, 1) = 7;
    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(1, 1), 7);
}

// Sizes
TEST(MatrixTests, rowAndColSize) {
    Matrix<> A(2, 2);
    Matrix<> B(5, 19);
    A(0, 0) = 1;
    B(4, 18) = 7;
    EXPECT_EQ(A.rows_size(), 2);
    EXPECT_EQ(A.cols_size(), 2);
    EXPECT_EQ(B.rows_size(), 5);
    EXPECT_EQ(B.cols_size(), 19);
}

// getRow
TEST(MatrixTests, getRow) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    std::vector<double> v1 = A.getRow(0);
    std::vector<double> v2 = A.getRow(1);
    std::vector<double> v3 = A.getRow(2);

    EXPECT_EQ(v1[0], 1);
    EXPECT_EQ(v1[1], 2);
    EXPECT_EQ(v1[2], 3);

    EXPECT_EQ(v2[0], 4);
    EXPECT_EQ(v2[1], 5);
    EXPECT_EQ(v2[2], 6);

    EXPECT_EQ(v3[0], 7);
    EXPECT_EQ(v3[1], 8);
    EXPECT_EQ(v3[2], 9);
}

// getCol
TEST(MatrixTests, getCol) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<double> v1 = A.getCol(0);
    std::vector<double> v2 = A.getCol(1);
    std::vector<double> v3 = A.getCol(2);

    EXPECT_EQ(v1[0], 1);
    EXPECT_EQ(v1[1], 4);
    EXPECT_EQ(v1[2], 7);

    EXPECT_EQ(v2[0], 2);
    EXPECT_EQ(v2[1], 5);
    EXPECT_EQ(v2[2], 8);

    EXPECT_EQ(v3[0], 3);
    EXPECT_EQ(v3[1], 6);
    EXPECT_EQ(v3[2], 9);
}

// getRow
TEST(MatrixTests, getRowCount) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<double> v1 = A.getRow(0, 1);
    std::vector<double> v2 = A.getRow(1, 2);
    std::vector<double> v3 = A.getRow(2, 3);

    EXPECT_EQ(v1[0], 1);

    EXPECT_EQ(v2[0], 4);
    EXPECT_EQ(v2[1], 5);

    EXPECT_EQ(v3[0], 7);
    EXPECT_EQ(v3[1], 8);
    EXPECT_EQ(v3[2], 9);
}

// getCol
TEST(MatrixTests, getColCount) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<double> v1 = A.getCol(0, 1);
    std::vector<double> v2 = A.getCol(1, 2);
    std::vector<double> v3 = A.getCol(2, 3);

    EXPECT_EQ(v1[0], 1);

    EXPECT_EQ(v2[0], 2);
    EXPECT_EQ(v2[1], 5);

    EXPECT_EQ(v3[0], 3);
    EXPECT_EQ(v3[1], 6);
    EXPECT_EQ(v3[2], 9);
}

// setCol
TEST(MatrixTests, setColVec) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<double> v1 = { 7, 7, 7 };
    std::vector<double> v2 = { 9, 9 };

    A.setCol(v1, 0);
    A.setCol(v2, 1);

    Matrix<> B = {
        {7, 9, 3},
        {7, 9, 6},
        {7, 8, 9}
    };

    EXPECT_EQ(A, B);
}

// setRow
TEST(MatrixTests, setRowVec) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<double> v1 = { 7, 7, 7 };
    std::vector<double> v2 = { 9, 9 };

    A.setRow(v1, 0);
    A.setRow(v2, 1);

    Matrix<> B = {
        {7, 7, 7},
        {9, 9, 6},
        {7, 8, 9}
    };

    EXPECT_EQ(A, B);
}

// setColMatrix
TEST(MatrixTests, setColMatrix) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    Matrix<> v1 = { 7, 7, 7 };
    Matrix<> v2 = { 9, 9 };
    //Matrix<> v3 = {
    //    {0},
    //    {0},
    //    {0}
    //};

    Matrix<> v3 = { 0, 0, 0 };


    A.setCol(v1, 0);
    A.setCol(v2, 1);
    A.setCol(v3, 2);

    Matrix<> B = {
        {7, 9, 0},
        {7, 9, 0},
        {7, 8, 0}
    };

    EXPECT_EQ(A, B);
}

// setRowMatrix
TEST(MatrixTests, setRowMatrix) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    Matrix<> v1 = { 7, 7, 7 };
    Matrix<> v2 = { 9, 9 };
    Matrix<> v3 = { 0, 0, 0 };

    A.setRow(v1, 0);
    A.setRow(v2, 1);
    A.setRow(v3, 2);

    Matrix<> B = {
        {7, 7, 7},
        {9, 9, 6},
        {0, 0, 0}
    };

    EXPECT_EQ(A, B);
}

// getSubmatrix
TEST(MatrixTests, getSubmatrix) {
    Matrix<int> A = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
    };

    // Extract a 2x2 submatrix starting from row 1, column 1
    Matrix<int> sub = A.getSubmatrix(1, 1, 2, 2);

    Matrix<int> expected = {
            {6, 7},
            {10, 11}
    };

    EXPECT_EQ(sub, expected);
}

// setSubmatrix
TEST(MatrixTests, setSubmatrix) {
    Matrix<int> A = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
    };

    // Create a 2x2 block to insert
    Matrix<int> block = {
            {100, 100},
            {100, 100}
    };

    // Insert the block starting at row 1, column 1
    A.setSubmatrix(1, 1, block);

    Matrix<int> expected = {
            {1, 2, 3, 4},
            {5, 100, 100, 8},
            {9, 100, 100, 12},
            {13, 14, 15, 16}
    };

    EXPECT_EQ(A, expected);
}

// Transpose Matrix
TEST(MatrixTests, transposeMatrix) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    Matrix<> B = {
        {1, 4, 7},
        {2, 5, 8},
        {3, 6, 9}
    };

    A = A.transpose();
    EXPECT_EQ(A, B);
}

// Set transpose Matrix
TEST(MatrixTests, setTransposeMatrix) {
    Matrix<> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    Matrix<> B = {
        {1, 4, 7},
        {2, 5, 8},
        {3, 6, 9}
    };

    A.setTranspose();
    EXPECT_EQ(A, B);
}

// Matrix determinant
TEST(MatrixTests, matrixDeterminant) {
    Matrix<> A = {1};

    Matrix<> B = {
        {1, 2},
        {4, 5},
    };

    Matrix<> C = {
        {1, 4, 7},
        {2, 8, 8},
        {3, 6, 9}
    };

    Matrix<> D = {
        {2, 5, 7, 3},
        {2, 5, 8, 2},
        {3, 6, 9, 1},
        {3, 1, 7, 3}
    };

    Matrix<> E = {
        {1, 4, 7, 3, 0},
        {2, 5, 8, 2, 2},
        {3, 6, 9, 1, 1},
        {3, 1, 7, 3, 9},
        {2, 2, 6, 4, 8}
    };

    double eps = double(10e-10);

    EXPECT_TRUE(A.determinant() < 1 + eps);
    EXPECT_TRUE(B.determinant() > -3.0 - eps);
    EXPECT_TRUE(C.determinant() > -36.0 - eps);
    EXPECT_TRUE(D.determinant() < 50.0 + eps);
    EXPECT_TRUE(E.determinant() < 120.0 + eps);
}

// Matrix inverse
TEST(MatrixTests, matrixInverse) {
    Matrix<> A = {
        {1, 4, 7},
        {2, 8, 8},
        {3, 6, 9}
    };

    Matrix<> B = {
        {-2.0/3.0,-1.0/6.0,2.0/3.0},
        {-1.0/6.0,1.0/3.0,-1.0/6.0},
        {1.0/3.0,-1.0/6.0,0.0}
    };

    double eps = double(10e-15);
    Matrix<> C = A.inverse();
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            EXPECT_TRUE(std::abs(C(i, j)) < std::abs(B(i, j)) + eps);
        }
    }
}

// Matrix setInverse
TEST(MatrixTests, matrixSetInverse) {
    Matrix<> A = {
        {1, 4, 7},
        {2, 8, 8},
        {3, 6, 9}
    };

    Matrix<> B = {
        {-2.0 / 3.0,-1.0 / 6.0,2.0 / 3.0},
        {-1.0 / 6.0,1.0 / 3.0,-1.0 / 6.0},
        {1.0 / 3.0,-1.0 / 6.0,0.0}
    };

    double eps = double(10e-15);
    A.setInverse();
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            EXPECT_TRUE(std::abs(A(i, j)) < std::abs(B(i, j)) + eps);
        }
    }
}

// Minor
TEST(MatrixTests, matrixMinor) {
    Matrix<> A = {
        {1, 4, 7},
        {2, 8, 8},
        {3, 6, 9}
    };

    double B = A.minor(1, 1);

    double C = A.minor(2, 2);

    EXPECT_EQ(B, -12.0);
    EXPECT_EQ(C, 0.0);
}

// Minor static
TEST(MatrixTests, StaticMatrixMinor) {
    Matrix<> A = {
        {1, 4, 7},
        {2, 8, 8},
        {3, 6, 9}
    };

    double B = Matrix<>::minor(1, 1, A);

    double C = Matrix<>::minor(2, 2, A);

    EXPECT_EQ(B, -12.0);
    EXPECT_EQ(C, 0.0);
}

// adj
TEST(MatrixTests, adjMatrix) {
    // 3x3
    Matrix<int> mat = {
            { 1, 2, 3},
            { 4, 5, 6},
            { 7, 8, 9}
    };

    auto subMat = mat.adjoint(0, 1);

    // check sub matrix
    EXPECT_EQ(subMat.rows_size(), 2);
    EXPECT_EQ(subMat.cols_size(), 2);

    // check sub matrix values
    EXPECT_EQ(subMat(0, 0), 4);
    EXPECT_EQ(subMat(0, 1), 6);
    EXPECT_EQ(subMat(1, 0), 7);
    EXPECT_EQ(subMat(1, 1), 9);

    EXPECT_THROW(mat.adjoint(3, 0), std::out_of_range);
    EXPECT_THROW(mat.adjoint(0, 3), std::out_of_range);
    EXPECT_THROW(mat.adjoint(3, 3), std::out_of_range);
}

// adj static
TEST(MatrixTests, adjStaticMatrix) {
    // 4x4
    Matrix<int> mat(4, 4);
    int counter = 1;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            mat(i, j) = counter++;
        }
    }

    auto subMat = Matrix<int>::adjoint(1, 2, mat);

    EXPECT_EQ(subMat.rows_size(), 3);
    EXPECT_EQ(subMat.cols_size(), 3);

    EXPECT_EQ(subMat(0, 0), 1);
    EXPECT_EQ(subMat(0, 1), 2);
    EXPECT_EQ(subMat(0, 2), 4);

    EXPECT_EQ(subMat(1, 0), 9);
    EXPECT_EQ(subMat(1, 1), 10);
    EXPECT_EQ(subMat(1, 2), 12);

    EXPECT_EQ(subMat(2, 0), 13);
    EXPECT_EQ(subMat(2, 1), 14);
    EXPECT_EQ(subMat(2, 2), 16);

    EXPECT_THROW(Matrix<int>::adjoint(4, 0, mat), std::out_of_range);
    EXPECT_THROW(Matrix<int>::adjoint(0, 4, mat), std::out_of_range);
    EXPECT_THROW(Matrix<int>::adjoint(4, 4, mat), std::out_of_range);
}

// Ones matrix (static)
TEST(MatrixTests, onesMatrix) {
    Matrix<> A = Matrix<>::ones(4);
    Matrix<> B = {
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1}
    };
    EXPECT_EQ(A, B);
}

// Ones matrix (static)
TEST(MatrixTests, onesRowColMatrix) {
    Matrix<> A = Matrix<>::ones(4, 5);
    Matrix<> B = {
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1}
    };
    EXPECT_EQ(A, B);
}

// set ones matrix
TEST(MatrixTests, setOnesMatrix) {
    Matrix<> A = {
        {1, 2, 3, 4},
        {1, 4, 23, 5},
        {0, 23, 93, 54},
        {1, 87, 35, 9}
    };
    A.setOnes();
    Matrix<> B = {
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1}
    };
    EXPECT_EQ(A, B);
}

// Identity matrix (static)
TEST(MatrixTests, identityMatrix) {
    Matrix<> A = Matrix<>::identity(4);
    Matrix<> B = {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1}
    };
    EXPECT_EQ(A, B);
}

// Identity matrix (static)
TEST(MatrixTests, identityRowColMatrix) {
    Matrix<> A = Matrix<>::identity(4, 5);
    Matrix<> B = {
            {1, 0, 0, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 0, 1, 0}
    };
    EXPECT_EQ(A, B);
}

// set Identity matrix
TEST(MatrixTests, setIdentityMatrix) {
    Matrix<> A = {
            {1, 2, 3, 4},
            {1, 4, 23, 5},
            {0, 23, 93, 54},
            {1, 87, 35, 9}
    };
    A.setIdentity();
    Matrix<> B = {
            {1, 2, 3, 4},
            {1, 1, 23, 5},
            {0, 23, 1, 54},
            {1, 87, 35, 1}
    };
    EXPECT_EQ(A, B);
}

// Zeroes matrix (static)
TEST(MatrixTests, zeroesSizeMatrix) {
    Matrix<> A = {
            {1, 2, 3, 4},
            {1, 1, 23, 5},
            {0, 23, 1, 54},
            {1, 87, 35, 1}
    };
    A = Matrix<>::zeroes(4);
    Matrix<> B = {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
    };
    EXPECT_EQ(A, B);
}

// Zeroes matrix (static)
TEST(MatrixTests, zeroesRowColMatrix) {
    Matrix<> A = {
            {1, 2, 3, 4},
            {1, 1, 23, 5},
            {0, 23, 1, 54},
            {1, 87, 35, 1}
    };

    A = Matrix<>::zeroes(4, 3);
    Matrix<> B = {
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 }
    };
    EXPECT_EQ(A, B);
}

// set zeroes matrix
TEST(MatrixTests, setZeroesMatrix) {
    Matrix<> A = {
        {1, 2, 3, 4},
        {1, 4, 23, 5},
        {0, 23, 93, 54},
        {1, 87, 35, 9}
    };
    A.setZeroes();
    Matrix<> B = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };
    EXPECT_EQ(A, B);
}



// random matrix
TEST(MatrixTests, randomMatrix) {
    Matrix<int> A = Matrix<int>::random(3, 3, 1, 10);
    Matrix<int> B = Matrix<int>::random(3, 3, 1, 10);
    EXPECT_TRUE(A != B);
}

// diag
TEST(MatrixTests, diagMatrix) {
    Matrix<> A = {
        {1, 2, 3, 4},
        {1, 4, 23, 5},
        {0, 23, 93, 54},
        {1, 87, 35, 9}
    };

    std::vector<double> d = A.diag();
    EXPECT_EQ(d[0], 1);
    EXPECT_EQ(d[1], 4);
    EXPECT_EQ(d[2], 93);
    EXPECT_EQ(d[3], 9);

    std::vector<double> d2 = Matrix<>::diag(A);
    EXPECT_EQ(d2[0], 1);
    EXPECT_EQ(d2[1], 4);
    EXPECT_EQ(d2[2], 93);
    EXPECT_EQ(d2[3], 9);
}

// diag
TEST(MatrixTests, traceMatrix) {
    Matrix<> A = {
        {1, 2, 3, 4},
        {1, 4, 23, 5},
        {0, 23, 93, 54},
        {1, 87, 35, 9}
    };

    double d = A.trace();
    EXPECT_EQ(d, 107);

    double d2 = Matrix<>::trace(A);
    EXPECT_EQ(d2, 107);
}

