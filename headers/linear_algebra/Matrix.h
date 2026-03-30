#ifndef MINIMIZEROPTIMIZER_HEADERS_MATRIX_H_
#define MINIMIZEROPTIMIZER_HEADERS_MATRIX_H_

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <initializer_list>
#include <random>
#include <type_traits>
#include <vector>
#include <concepts>

// Main concept
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Helping vector concept
template <typename T>
concept IsVector = requires {
    typename T::value_type;
    requires std::same_as<T, std::vector<typename T::value_type>>;
};

// Concept for vector of non-vector elements
template <typename T>
concept VectorType = IsVector<T> && (!IsVector<typename T::value_type>);

// Concept for vector of vectors
template <typename T>
concept VectorVectorType = IsVector<T> && IsVector<typename T::value_type>;

// The matrix class
template <Arithmetic T = double>
class Matrix {
public:
    using size_type = size_t;
    using iterator_type = size_t;

public:
    Matrix() = default;
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    ~Matrix();

    explicit Matrix(const size_type& size);
    explicit Matrix(const size_type& rows, const size_type& cols);
    explicit Matrix(const size_type& rows, const size_type& cols, const T& value);

    template <VectorVectorType V>
    explicit Matrix(const V& vec);

    template <VectorType V>
    explicit Matrix(const V& vec);

    Matrix(const std::initializer_list<std::initializer_list<T>>& values);
    Matrix(const std::initializer_list<T> &values);


    // operators

    Matrix<T>& operator=(const Matrix<T>& other);
    Matrix<T>& operator=(Matrix<T>&& other) noexcept;
    Matrix<T>& operator=(const std::initializer_list<std::initializer_list<T>>& values);

    // Matrix and Matrix operators

    //operator+(const Matrix<T>& A, const Matrix<T>& B)
    //operator-(const Matrix<T>& A, const Matrix<T>& B)
    //operator*(const Matrix<T>& A, const Matrix<T>& B)

    //Matrix<V>& operator+=(Matrix<T>& A, const Matrix<T>& B)
    //Matrix<V>& operator-=(Matrix<T>& A, const Matrix<T>& B)
    //Matrix<V>& operator*=(Matrix<T>& A, const Matrix<T>& B)

    //bool operator==(const Matrix<T>& A, const Matrix<T>& B)
    //bool operator!=(const Matrix<T>& A, const Matrix<T>& B)

    // Matrix and scalar operators

    //Matrix<T> operator+(const Matrix<T>& A, const V& scalar)
    //Matrix<T> operator-(const Matrix<T>& A, const V& scalar)
    //Matrix<T> operator*(const Matrix<T>& A, const V& scalar)
    //Matrix<T> operator/(const Matrix<t>& A, const V& scalar)

    //Matrix<T>& operator+=(Matrix<T>& A, const V& scalar)
    //Matrix<T>& operator-=(Matrix<T>& A, const V& scalar)
    //Matrix<T>& operator*=(Matrix<T>& A, const V& scalar)
    //Matrix<T>& operator/=(Matrix<T>& A, const V& scalar)

    // Matrix and List operators

    //Matrix<T> operator+(const Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L)
    //Matrix<T> operator-(const Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L)
    //Matrix<T> operator*(const Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L)

    //Matrix<T>& operator+=(Matrix<T>& A, const std::initializer_list<std::initializer_list<T>> L)
    //Matrix<T>& operator-=(Matrix<T>& A, const std::initializer_list<std::initializer_list<T>> L)
    //Matrix<T>& operator*=(Matrix<T>& A, const std::initializer_list<std::initializer_list<T>> L)

    //bool operator==(const Matrix<T>& A, const std::initializer_list<std::initializer_list<T>> L)
    //bool operator!=(const Matrix<T>& A, const std::initializer_list<std::initializer_list<T>> L)

    // Element Access
    T& operator()(iterator_type rowIndex, iterator_type colIndex);
    T operator()(iterator_type rowIndex, iterator_type colIndex) const;


    // Matrix function

    std::vector<T> getCol(const iterator_type& colI) const;
    std::vector<T> getRow(const iterator_type& rowI) const;

    std::vector<T> getCol(const iterator_type& colI, const size_type& count) const;
    std::vector<T> getRow(const iterator_type& rowI, const size_type& count) const;

    void setCol(const std::vector<T>& colV, const iterator_type& colI) const;
    void setRow(const std::vector<T>& rowV, const iterator_type& rowI) const;

    void setCol(const Matrix<T>& A, const iterator_type& colI) const;
    void setRow(const Matrix<T>& A, const iterator_type& rowI) const;

    Matrix getSubmatrix(const size_type& start, const size_type& end, const size_type& num_rows, const size_type& num_cols) const;
    void setSubmatrix(const size_type& start_row, const size_type& start_col, const Matrix<T>& block);

    Matrix transpose() const;
    void setTranspose();

    T determinant() const;
    static T determinant(const Matrix<T>& mat);

    Matrix inverse() const;
    void setInverse();

    Matrix adjoint(const size_type& i, const size_type& j) const;
    static Matrix adjoint(const size_type& i, const size_type& j, const Matrix<T>& mat);

    T minor(const size_type& i, const size_type& j) const;
    static T minor(const size_type& i, const size_type& j, const Matrix<T>& mat);

    T trace() const;
    static T trace(const Matrix& mat);

    std::vector<T> diag() const;
    static std::vector<T> diag(const Matrix<T>& mat);

    static Matrix ones(const size_type& size);
    static Matrix ones(const size_type& rows, const size_type& cols);
    void setOnes();

    static Matrix identity(const size_type& size);
    static Matrix identity(const size_type& rows, const size_type& cols);
    void setIdentity();

    static Matrix zeroes(const size_type& size);
    static Matrix zeroes(const size_type& rows, const size_type& cols);
    void setZeroes();

    static Matrix random(const size_type& size, const T& leftNum = 0.0, const T& rightNum = 1.0);
    static Matrix random(const size_type& rows, const size_type& cols, const T& leftNum = 0.0, const T& rightNum = 1.0);
    void setRandom(const T& leftNum = 0.0, const T& rightNum = 1.0);
    T norm() const;

    static T norm(const Matrix<T>& mat);
    inline size_type rows_size() const { return rows; }
    inline size_type cols_size() const { return cols; }

private:
    size_type rows = size_type(0);
    size_type cols = size_type(0);
    T** matrix = nullptr;
};

template <Arithmetic T>
inline Matrix<T>::Matrix(const size_type& rows, const size_type& cols) : rows(rows), cols(cols)
{
    matrix = new T *[rows];
    for (iterator_type i = 0; i < rows; i++) {
        matrix[i] = new T[cols];
        for (iterator_type j = 0; j < cols; j++) {
            matrix[i][j] = T();
        }
    }
}

template<Arithmetic T>
inline Matrix<T>::Matrix(const size_type& rows, const size_type& cols, const T& value) : rows(rows), cols(cols)
{
    matrix = new T * [rows];
    for (iterator_type i = 0; i < rows; i++) {
        matrix[i] = new T[cols];
        for (iterator_type j = 0; j < cols; j++) {
            matrix[i][j] = value;
        }
    }
}
template<Arithmetic T>
inline T Matrix<T>::norm() const
{
    T sum = 0;
    for (size_type i = 0; i < rows; i++) {
        for (size_type j = 0; j < cols; j++) {
            sum += matrix[i][j] * matrix[i][j];
        }
    }
    return std::sqrt(sum);
}

template <Arithmetic T>
T Matrix<T>::norm(const Matrix<T>& mat) {
    return mat.norm();
}

template<Arithmetic T>
inline Matrix<T>::Matrix(const size_type& size) : rows(size), cols(size)
{
    matrix = new T * [size];
    for (typename Matrix<T>::iterator_type i = 0; i < size; i++) {
        matrix[i] = new T[cols];
        for (typename Matrix<T>::iterator_type j = 0; j < size; j++) {
            matrix[i][j] = T();
        }
    }
}

template<Arithmetic T>
template<VectorType V>
inline Matrix<T>::Matrix(const V& vec) : rows(1), cols(vec.size())
{
    matrix = new T * [rows];
    matrix[0] = new T[cols];
    for (typename Matrix<T>::iterator_type i = 0; i < cols; i++) {
        matrix[0][i] = vec[i];
    }
}

template<Arithmetic T>
template<VectorVectorType V>
inline Matrix<T>::Matrix(const V& vec) : rows(vec.size()), cols(vec[0].size())
{
    matrix = new T * [rows];
    for (iterator_type i = 0; i < rows; i++) {
        matrix[i] = new T[cols];
        for (iterator_type j = 0; j < cols; j++) {
            matrix[i][j] = vec[i][j];
        }
    }
}

template <Arithmetic T>
inline Matrix<T>::Matrix(const Matrix<T>& other) : rows(other.rows), cols(other.cols) {
    // check matrix and free memory if needed
    if (matrix != NULL){
        for (iterator_type i = 0; i < rows; i++)
        {
            delete[] matrix[i];
        }
        if (rows != 0) delete[] matrix;
    }
    // is rows == 0 or cols == 0 we dont need create matrix
    if (rows != 0 && cols != 0) matrix = new T * [rows];

    for (typename Matrix<T>::iterator_type i = 0; i < rows; i++) {
        matrix[i] = new T[cols];
        for (typename Matrix<T>::iterator_type j = 0; j < cols; j++) {
            matrix[i][j] = other.matrix[i][j];
        }
    }

}

template <Arithmetic T>
inline Matrix<T>::Matrix(Matrix<T>&& other) noexcept : rows(other.rows), cols(other.cols), matrix(other.matrix) {
    other.rows = 0;
    other.cols = 0;
    other.matrix = nullptr;
}

template<Arithmetic T>
inline Matrix<T>::Matrix(const std::initializer_list<std::initializer_list<T>>& values)
{
    rows = values.size();
    cols = values.begin()->size();
    if (rows != 0) matrix = new T*[rows];
    iterator_type i = 0;
    for (const auto& row_values : values) {
        if (row_values.size() != cols) {
            for (typename Matrix<T>::iterator_type j = 0; j < i; j++) {
                delete[] matrix[j];
            }
            if (rows != 0) delete[] matrix;
            throw std::invalid_argument("All rows must have the same number of columns.");
        }
        matrix[i] = new T[cols];
        iterator_type j = 0;
        for (const auto& val : row_values) {
            matrix[i][j] = val;
            ++j;
        }
        ++i;
    }
}

template<Arithmetic T>
inline Matrix<T>::Matrix(const std::initializer_list<T>& values)
{
    rows = 1;
    cols = values.size();

    if (cols == 0) {
        matrix = nullptr;
        return;
    }
    matrix = new T * [rows];
    matrix[0] = new T[cols];
    iterator_type i = 0;
    for (const auto& val : values) {
        matrix[0][i++] = val;
    }
}

template <Arithmetic T>
inline Matrix<T>::~Matrix() {
    for (iterator_type i = 0; i < rows; i++)
    {
        delete[] matrix[i];
    }
    if (rows != 0) delete[] matrix;
}

template<Arithmetic T>
inline Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
{
    if (this == &other) return *this;
    Matrix<T> temp(other);
    std::swap(rows, temp.rows);
    std::swap(cols, temp.cols);
    std::swap(matrix, temp.matrix);
    return *this;
}

template<Arithmetic T>
inline Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept
{
    if (this == &other) return *this;
    Matrix<T> temp(std::move(other));
    std::swap(rows, temp.rows);
    std::swap(cols, temp.cols);
    std::swap(matrix, temp.matrix);
    return *this;
}

template<Arithmetic T>
inline Matrix<T>& Matrix<T>::operator=(const std::initializer_list<std::initializer_list<T>>& values)
{
    for (typename Matrix<T>::iterator_type i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
    rows = values.size();
    cols = values.begin()->size();
    matrix = new T * [rows];

    typename Matrix<T>::iterator_type i = 0;
    for (auto& row_list : values) {
        if (row_list.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns.");
        }
        matrix[i] = new T[cols];
        typename Matrix<T>::iterator_type j = 0;
        for (auto& value : row_list) {
            matrix[i][j++] = value;
        }
        i++;
    }

    return *this;
}

//////////////////////////////////////////////////////////////////////////////////////// out operators start

template <Arithmetic V>
inline constexpr Matrix<V> operator+(const Matrix<V>& A, const Matrix<V>& B) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) {
        throw std::invalid_argument("Matrices must have the same size for addition.");
    }
    Matrix<V> result(A);
    result += B;
    return result;
}

template<Arithmetic V>
inline constexpr Matrix<V> operator-(const Matrix<V>& A, const Matrix<V>& B)
{
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) {
        throw std::invalid_argument("Matrices must have the same size for subtract.");
    }
    Matrix<V> result(A);
    result -= B;
    return result;
}

template<Arithmetic V>
inline constexpr Matrix<V> operator*(const Matrix<V>& A, const Matrix<V>& B)
{
    Matrix<V> result(A);
    result *= B;
    return result;
}

template<Arithmetic V>
inline constexpr Matrix<V> operator+(const Matrix<V>& A, const V& scalar)
{
    Matrix<V> res(A);
    for (typename Matrix<V>::iterator_type i = 0; i < A.rows_size(); i++) {
        for (typename Matrix<V>::iterator_type j = 0; j < A.cols_size(); j++) {
            res(i, j) += scalar;
        }
    }
    return res;
}

template<Arithmetic V>
inline constexpr Matrix<V> operator-(const Matrix<V>& A, const V& scalar)
{
    Matrix<V> res(A);
    for (typename Matrix<V>::iterator_type i = 0; i < A.rows_size(); i++) {
        for (typename Matrix<V>::iterator_type j = 0; j < A.cols_size(); j++) {
            res(i, j) -= scalar;
        }
    }
    return res;
}

template<Arithmetic V>
inline constexpr Matrix<V> operator*(const Matrix<V>& A, const V& scalar)
{
    Matrix<V> res(A);
    for (typename Matrix<V>::iterator_type i = 0; i < A.rows_size(); i++) {
        for (typename Matrix<V>::iterator_type j = 0; j < A.cols_size(); j++) {
            res(i, j) *= scalar;
        }
    }
    return res;
}

template<Arithmetic V>
inline constexpr Matrix<V> operator/(const Matrix<V>& A, const V& scalar)
{
    if (scalar == V())
    {
        throw std::runtime_error("scalar shouldn't be zero");
    }
    Matrix<V> res(A);
    for (typename Matrix<V>::iterator_type i = 0; i < A.rows_size(); i++) {
        for (typename Matrix<V>::iterator_type j = 0; j < A.cols_size(); j++) {
            res(i, j) /= scalar;
        }
    }
    return res;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator+=(Matrix<V>& A, const Matrix<V>& B)
{
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) {
        throw std::invalid_argument("Matrices must have the same size for addition.");
    }
    for (typename Matrix<V>::iterator_type i = 0; i < A.rows_size(); i++) {
        for (typename Matrix<V>::iterator_type j = 0; j < B.cols_size(); j++) {
            A(i, j) += B(i, j);
        }
    }
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator-=(Matrix<V>& A, const Matrix<V>& B)
{
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) {
        throw std::invalid_argument("Matrices must have the same size for addition.");
    }
    for (typename Matrix<V>::iterator_type i = 0; i < A.rows_size(); i++) {
        for (typename Matrix<V>::iterator_type j = 0; j < B.cols_size(); j++) {
            A(i, j) -= B(i, j);
        }
    }
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator*=(Matrix<V>& A, const Matrix<V>& B)
{
    if (A.cols_size() != B.rows_size()) {
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication");
    }

    Matrix<V> C(A.rows_size(), B.cols_size());
    for (typename Matrix<V>::iterator_type i = 0; i < A.rows_size(); i++) {
        for (typename Matrix<V>::iterator_type j = 0; j < B.cols_size(); j++) {
            for (typename Matrix<V>::iterator_type k = 0; k < A.cols_size(); k++) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    A = C;
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator+=(Matrix<V>& A, const V& scalar)
{
    A = A + scalar;
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator-=(Matrix<V>& A, const V& scalar)
{
    A = A - scalar;
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator*=(Matrix<V>& A, const V& scalar)
{
    A = A * scalar;
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator/=(Matrix<V>& A, const V& scalar)
{
    A = A / scalar;
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V> operator+(const Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L)
{
    typename Matrix<V>::size_type r = L.size();
    typename Matrix<V>::size_type c = L.begin()->size();
    if (A.rows_size() != r || A.cols_size() != c) {
        throw std::invalid_argument("Matrices must have the same size for addition.");
    }
    Matrix<V> result(A);
    auto row_it = L.begin();
    for (typename Matrix<V>::iterator_type i = 0; i < r; ++i, ++row_it) {
        auto col_it = row_it->begin();
        for (typename Matrix<V>::iterator_type j = 0; j < c; ++j, ++col_it) {
            result(i, j) += *col_it;
        }
    }
    return result;
}

template<Arithmetic V>
inline constexpr Matrix<V> operator-(const Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L)
{
    typename Matrix<V>::size_type r = L.size();
    typename Matrix<V>::size_type c = L.begin()->size();
    if (A.rows_size() != r || A.cols_size() != c) {
        throw std::invalid_argument("Matrices must have the same size for addition.");
    }
    Matrix<V> result(A);
    auto row_it = L.begin();
    for (typename Matrix<V>::iterator_type i = 0; i < r; ++i, ++row_it) {
        auto col_it = row_it->begin();
        for (typename Matrix<V>::iterator_type j = 0; j < c; ++j, ++col_it) {
            result(i, j) -= *col_it;
        }
    }
    return result;
}

template <Arithmetic V>
inline constexpr Matrix<V> operator*(const Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L) {
    typename Matrix<V>::size_type L_rows = L.size();
    if (L_rows == 0) {
        throw std::invalid_argument("Second matrix not be empty.");
    }
    typename Matrix<V>::size_type L_cols = L.begin()->size();
    for (const auto& rows : L) {
        if (rows.size() != L_cols) {
            throw std::invalid_argument("All string int second initializer_list should be one count of cols.");
        }
    }
    if (A.cols_size() != L_rows) {
        throw std::invalid_argument("A rows != B cols.");
    }
    Matrix<V> C(A.rows_size(), L_cols);
    std::vector<std::vector<V>> L_matrix;
    L_matrix.reserve(L_rows);
    for (const auto& rows : L) {
        L_matrix.emplace_back(rows);
    }
    for (typename Matrix<V>::size_type i = 0; i < A.rows_size(); ++i) {
        for (typename Matrix<V>::size_type j = 0; j < L_cols; ++j) {
            V sum = V{};
            for (typename Matrix<V>::size_type k = 0; k < A.cols_size(); ++k) {
                sum += A(i, k) * L_matrix[k][j];
            }
            C(i, j) = sum;
        }
    }

    return C;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator+=(Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L)
{
    A = A + L;
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator-=(Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L)
{
    A = A - L;
    return A;
}

template<Arithmetic V>
inline constexpr Matrix<V>& operator*=(Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> L)
{
    A = A * L;
    return A;
}

template<Arithmetic V>
inline constexpr bool operator==(const Matrix<V>& A, const Matrix<V>& B)
{
    if (A.rows_size() == B.rows_size() && A.cols_size() == B.cols_size())
    {
        for (typename Matrix<V>::iterator_type i = 0; i < A.rows_size(); ++i)
        {
            for (typename Matrix<V>::iterator_type j = 0; j < A.cols_size(); ++j)
            {
                if (A(i, j) != B(i, j))
                {
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}

template<Arithmetic V>
inline constexpr bool operator==(const Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> values)
{
    if (A.rows_size() != values.size() || A.cols_size() != values.begin()->size())
    {
        throw std::invalid_argument("Matrices must have the same size");
    }
    auto row_it = values.begin();
    for (typename Matrix<V>::size_type i = 0; i < A.rows_size(); ++i, ++row_it)
    {
        auto col_it = row_it->begin();
        for (typename Matrix<V>::size_type j = 0; j < A.cols_size(); ++j, ++col_it)
        {
            if (A(i, j) != *col_it) {
                return false;
            }
        }
    }
    return true;
}

template<Arithmetic V>
inline constexpr bool operator!=(const Matrix<V>& A, const Matrix<V>& B)
{
    return !(A == B);
}

template<Arithmetic V>
inline constexpr bool operator!=(const Matrix<V>& A, const std::initializer_list<std::initializer_list<V>> B)
{
    return !(A == B);
}

//////////////////////////////////////////////////////////////////////////////////////// out operators end



template <Arithmetic T>
inline T& Matrix<T>::operator()(iterator_type rowIndex, iterator_type colIndex) {
    if (rowIndex >= this->rows || colIndex >= this->cols) {
        throw std::out_of_range("Index out of range");
    }
    return matrix[rowIndex][colIndex];
}

template <Arithmetic T>
inline T Matrix<T>::operator()(iterator_type rowIndex, iterator_type colIndex) const {
    if (rowIndex >= this->rows || colIndex >= this->cols) {
        throw std::out_of_range("Index out of range");
    }
    return matrix[rowIndex][colIndex];
}

template<Arithmetic T>
inline std::vector<T> Matrix<T>::getCol(const iterator_type& colI) const
{
    if (colI >= cols)
    {
        throw std::out_of_range("Index out of range");
    }
    std::vector<T> res(rows);
    for (iterator_type i = 0; i < rows; i++)
    {
        res[i] = matrix[i][colI];
    }
    return res;
}

template<Arithmetic T>
inline std::vector<T> Matrix<T>::getRow(const iterator_type& rowI) const
{
    if (rowI >= rows)
    {
        throw std::out_of_range("Index out of range");
    }
    std::vector<T> res(cols);
    for (iterator_type i = 0; i < rows; i++)
    {
        res[i] = matrix[rowI][i];
    }
    return res;
}

template<Arithmetic T>
inline std::vector<T> Matrix<T>::getCol(const iterator_type& colI, const size_type& count) const
{
    if (colI >= cols || count > rows)
    {
        throw std::out_of_range("Index out of range");
    }
    std::vector<T> res(count);
    for (iterator_type i = 0; i < count; i++)
    {
        res[i] = matrix[i][colI];
    }
    return res;
}

template<Arithmetic T>
inline std::vector<T> Matrix<T>::getRow(const iterator_type& rowI, const size_type& count) const
{
    if (rowI >= rows || count > cols)
    {
        throw std::out_of_range("Index out of range");
    }
    std::vector<T> res(count);
    for (iterator_type i = 0; i < count; i++)
    {
        res[i] = matrix[rowI][i];
    }
    return res;
}

template<Arithmetic T>
inline void Matrix<T>::setCol(const std::vector<T>& colV, const iterator_type& colI) const
{
    if (colV.size() > rows || colI >= cols)
    {
        throw std::out_of_range("Index out of range");
    }
    for (iterator_type i = 0; i < colV.size(); i++)
    {
        matrix[i][colI] = colV[i];
    }
}

template<Arithmetic T>
inline void Matrix<T>::setRow(const std::vector<T>& rowV, const iterator_type& rowI) const
{
    if (rowV.size() > cols || rowI >= rows)
    {
        throw std::out_of_range("Index out of range");
    }
    for (iterator_type i = 0; i < rowV.size(); i++)
    {
        matrix[rowI][i] = rowV[i];
    }
}

template<Arithmetic T>
inline void Matrix<T>::setCol(const Matrix<T>& A, const iterator_type& colI) const
{
    // 1xn
    if (A.cols_size() <= cols && A.rows_size() == 1)
    {
        for (iterator_type i = 0; i < A.cols_size(); i++)
        {
            matrix[i][colI] = A(0, i);
        }
    }
    // nx1
    else if (A.rows_size() <= rows && A.cols_size() == 1)
    {
        for (iterator_type i = 0; i < A.rows_size(); i++)
        {
            matrix[i][colI] = A(i, 0);
        }
    }
    else {
        throw std::out_of_range("Index out of range");
    }
}

template<Arithmetic T>
inline void Matrix<T>::setRow(const Matrix<T>& A, const iterator_type& rowI) const
{
    if (A.cols_size() <= cols && A.rows_size() == 1)
    {
        for (iterator_type i = 0; i < A.cols_size(); i++)
        {
            matrix[rowI][i] = A(0, i);
        }
    }
    else if (A.rows_size() <= rows && A.cols_size() == 1)
    {
        for (iterator_type i = 0; i < A.rows_size(); i++)
        {
            matrix[rowI][i] = A(i, 0);
        }
    }
    else {
        throw std::out_of_range("Index out of range");
    }
}

template<Arithmetic T>
Matrix<T> Matrix<T>::getSubmatrix(const size_type& start_row, const size_type& start_col, const size_type& num_rows, const size_type& num_cols) const
{
    // Check
    if (start_row + num_rows > rows || start_col + num_cols > cols) {
        throw std::out_of_range("getSubmatrix: Index out of range");
    }

    // Create target matrix
    Matrix<T> submatrix(num_rows, num_cols);

    for (size_type i = 0; i < num_rows; ++i) {
        for (size_type j = 0; j < num_cols; ++j) {
            submatrix(i, j) = matrix[start_row + i][start_col + j];
        }
    }

    return submatrix;
}

template<Arithmetic T>
void Matrix<T>::setSubmatrix(const size_type& start_row, const size_type& start_col, const Matrix<T>& block) {
    size_type block_rows = block.rows_size();
    size_type block_cols = block.cols_size();

    // Check
    if (start_row + block_rows > rows || start_col + block_cols > cols) {
        throw std::out_of_range("setSubmatrix: Index out of range");
    }

    for (size_type i = 0; i < block_rows; ++i) {
        for (size_type j = 0; j < block_cols; ++j) {
            matrix[start_row + i][start_col + j] = block(i, j);
        }
    }
}


template<Arithmetic T>
inline Matrix<T> Matrix<T>::transpose() const
{
    Matrix<T> tra(cols, rows);
    for (typename Matrix<T>::iterator_type i = 0; i < rows; ++i) {
        for (typename Matrix<T>::iterator_type j = 0; j < cols; ++j) {
            tra.matrix[j][i] = matrix[i][j];
        }
    }
    return tra;
}

template<Arithmetic T>
inline T Matrix<T>::determinant() const
{
    if (rows != cols) {
        throw std::runtime_error("rows != cols: matrix is cannot be to find determinant");
    }
    size_type s = rows;
    if (rows == 0 || cols == 0){
        throw std::runtime_error("rows or cols cannot be equel to zero");
    }
    T** tempMatrix = new T*[s];
    for (size_type i = 0; i < s; ++i) {
        tempMatrix[i] = new T[s];
        for (size_type j = 0; j < s; ++j) {
            tempMatrix[i][j] = matrix[i][j];
        }
    }

    T d = T(1);
    T eps = T(10e-20);
    for (size_t i = 0; i < s; ++i) {
        if (std::abs(tempMatrix[i][i]) < eps) {
            for (size_t k = i + 1; k < s; ++k) {
                if (std::abs(tempMatrix[k][i]) > eps) {
                    for (size_t j = 0; j < s; ++j) {
                        std::swap(tempMatrix[i][j], tempMatrix[k][j]);
                    }
                    d *= T(-1);
                    break;
                }
            }
        }
        if (std::abs(tempMatrix[i][i]) < eps) {
            for (typename Matrix<T>::iterator_type i = 0; i < s; i++)
            {
                delete[] tempMatrix[i];
            }
            if (rows != 0) delete[] tempMatrix;
            return 0;
        }
        for (size_t k = i + 1; k < s; ++k) {
            T factor = tempMatrix[k][i] / tempMatrix[i][i];
            for (size_t j = i; j < s; ++j) {
                tempMatrix[k][j] -= factor * tempMatrix[i][j];
            }
        }
        d *= tempMatrix[i][i];
    }

    for (typename Matrix<T>::iterator_type i = 0; i < s; i++)
    {
        delete[] tempMatrix[i];
    }
    if (rows != 0) delete[] tempMatrix;

    return d;
}

template<Arithmetic T>
inline T Matrix<T>::determinant(const Matrix<T>& mat)
{
    return mat.determinant();
}

template<Arithmetic T>
inline void Matrix<T>::setTranspose()
{
    (*this) = (*this).transpose();
}

template<Arithmetic T>
inline Matrix<T> Matrix<T>::inverse() const
{
    if (rows != cols) {
        throw std::invalid_argument("Inverse can only be computed for square matrices.");
    }

    T det = (*this).determinant();
    if (det == 0) {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }

    T** invMat = new T * [rows];
    for (size_type i = 0; i < rows; ++i) {
        invMat[i] = new T[cols];
    }

    T** tempMatrix = new T * [rows];
    for (size_type i = 0; i < rows; ++i) {
        tempMatrix[i] = new T[rows];
        for (size_type j = 0; j < rows; ++j) {
            tempMatrix[i][j] = matrix[i][j];
        }
    }

    for (size_type i = 0; i < rows; ++i) {
        for (size_type j = 0; j < rows; ++j) {
            invMat[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_type i = 0; i < rows; ++i) {
        if (tempMatrix[i][i] == 0.0) {
            for (size_type j = i + 1; j < rows; ++j) {
                if (tempMatrix[j][i] != 0.0) {
                    for (size_type k = 0; k < rows; ++k) {
                        std::swap(tempMatrix[i][k], tempMatrix[j][k]);
                        std::swap(invMat[i][k], invMat[j][k]);
                    }
                    break;
                }
            }
        }

        T factor = tempMatrix[i][i];
        for (size_type j = 0; j < rows; ++j) {
            tempMatrix[i][j] /= factor;
            invMat[i][j] /= factor;
        }

        for (size_type j = 0; j < rows; ++j) {
            if (j != i) {
                factor = tempMatrix[j][i];
                for (size_type k = 0; k < rows; ++k) {
                    tempMatrix[j][k] -= factor * tempMatrix[i][k];
                    invMat[j][k] -= factor * invMat[i][k];
                }
            }
        }
    }

    Matrix<T> result(rows, cols);
    for (size_type i = 0; i < rows; i++) {
        for (size_type j = 0; j < rows; j++) {
            result(i, j) = invMat[i][j];
        }
    }

    for (typename Matrix<T>::iterator_type i = 0; i < rows; i++)
    {
        delete[] invMat[i];
        delete[] tempMatrix[i];
    }
    delete[] invMat;
    delete[] tempMatrix;

    return result;
}

template<Arithmetic T>
inline Matrix<T> Matrix<T>::adjoint(const size_type& i, const size_type& j) const {
    return adjoint(i, j, *this);
}

template<Arithmetic T>
inline Matrix<T> Matrix<T>::adjoint(const size_type& i, const size_type& j, const Matrix<T>& mat) {
    if (i >= mat.rows || j >= mat.cols) {
        throw std::out_of_range("Index out of range");
    }

    Matrix<T> result(mat.rows - 1, mat.cols - 1);

    size_type result_row = 0, result_col = 0;
    for (size_type row = 0; row < mat.rows; ++row) {
        if (row == i) continue;
        result_col = 0;
        for (size_type col = 0; col < mat.cols; ++col) {
            if (col == j) continue;
            result.matrix[result_row][result_col] = mat.matrix[row][col];
            ++result_col;
        }
        ++result_row;
    }

    return result;
}

template<Arithmetic T>
inline void Matrix<T>::setInverse()
{
    (*this) = (*this).inverse();
}

template<Arithmetic T>
inline T Matrix<T>::minor(const size_type& ix, const size_type& jx) const
{
    if (rows != cols) {
        throw std::invalid_argument("Matrix should be square!");
    }
    std::vector<std::vector<T>> subMatrix;
    for (size_type i = 0; i < rows; ++i)
    {
        if (i == ix) continue;
        std::vector<T> subRow;
        for (size_type j = 0; j < rows; ++j)
        {
            if (j == jx) continue;
            subRow.push_back(matrix[i][j]);
        }
        subMatrix.push_back(subRow);
    }
    Matrix<T> res(rows - 1, cols - 1);
    for (size_type i = 0; i < rows - 1; i++)
    {
        for (size_type j = 0; j < cols - 1; j++)
        {
            res(i, j) = subMatrix[i][j];
        }
    }
    return determinant(res);
}

template<Arithmetic T>
inline T Matrix<T>::minor(const size_type& ix, const size_type& jx, const Matrix<T>& mat)
{
    if (mat.rows != mat.cols) {
        throw std::invalid_argument("Matrix should be square!");
    }
    std::vector<std::vector<T>> subMatrix;
    for (size_type i = 0; i < mat.rows; ++i)
    {
        if (i == ix) continue;
        std::vector<T> subRow;
        for (size_type j = 0; j < mat.rows; ++j)
        {
            if (j == jx) continue;
            subRow.push_back(mat(i, j));
        }
        subMatrix.push_back(subRow);
    }
    Matrix<T> res(mat.rows - 1, mat.cols - 1);
    for (size_type i = 0; i < mat.rows - 1; i++)
    {
        for (size_type j = 0; j < mat.cols - 1; j++)
        {
            res(i, j) = subMatrix[i][j];
        }
    }
    return determinant(res);
}

template<Arithmetic T>
inline T Matrix<T>::trace() const
{
    T result = 0.0;
    size_type min_dim = std::min(rows, cols);
    for (size_type i = 0; i < min_dim; i++)
    {
        result += matrix[i][i];
    }
    return result;
}

template<Arithmetic T>
inline T Matrix<T>::trace(const Matrix<T>& mat)
{
    T result = 0.0;
    size_type min_dim = std::min(mat.rows, mat.cols);
    for (size_type i = 0; i < min_dim; i++)
    {
        result += mat(i, i);
    }
    return result;
}

template<Arithmetic T>
inline std::vector<T> Matrix<T>::diag() const
{
    std::vector<T> result;
    size_type min_dim = std::min(rows, cols);
    for (size_type i = 0; i < min_dim; i++)
    {
        result.push_back(matrix[i][i]);
    }
    return result;
}

template<Arithmetic T>
inline std::vector<T> Matrix<T>::diag(const Matrix<T>& mat)
{
    std::vector<T> result;
    size_type min_dim = std::min(mat.rows_size(), mat.cols_size());
    for (size_type i = 0; i < min_dim; i++)
    {
        result.push_back(mat(i, i));
    }
    return result;
}

template<Arithmetic T>
inline Matrix<T> Matrix<T>::ones(const size_type& size)
{
    Matrix<T> result(size, size);
    for (size_type i = 0; i < size; i++)
    {
        for (size_type j = 0; j < size; j++)
        {
            result(i, j) = T(1);
        }
    }
    return result;
}

template<Arithmetic T>
inline Matrix<T> Matrix<T>::ones(const size_type& rows, const size_type& cols)
{
    Matrix<T> result(rows, cols);
    for (size_type i = 0; i < rows; i++)
    {
        for (size_type j = 0; j < cols; j++)
        {
            result(i, j) = T(1);
        }
    }
    return result;
}

template<Arithmetic T>
inline void Matrix<T>::setOnes()
{
    for (size_type i = 0; i < rows; i++)
    {
        for (size_type j = 0; j < cols; j++)
        {
            matrix[i][j] = T(1);
        }
    }
}

template<Arithmetic T>
Matrix<T> Matrix<T>::identity(const Matrix::size_type &size) {
    Matrix<T> result(size, size);
    for (size_type i = 0; i < size; i++)
    {
        result(i, i) = T(1);
    }
    return result;
}

template<Arithmetic T>
Matrix<T> Matrix<T>::identity(const size_type& rows, const size_type& cols) {
    Matrix<T> result(rows, cols);
    for (size_type i = 0; i < rows; i++) {
        result(i, i) = T(1);
    }
    return result;
}

template<Arithmetic T>
void Matrix<T>::setIdentity() {
    for (size_type i = 0; i < rows; i++)
    {
        matrix[i][i] = T(1);
    }
}

template<Arithmetic T>
inline Matrix<T> Matrix<T>::zeroes(const size_type& size)
{
    Matrix<T> result(size, size);
    for (size_type i = 0; i < size; i++)
    {
        for (size_type j = 0; j < size; j++)
        {
            result(i, j) = T(0);
        }
    }
    return result;
}


template<Arithmetic T>
inline Matrix<T> Matrix<T>::zeroes(const size_type& rows, const size_type& cols)
{
    Matrix<T> result(rows, cols);
    for (size_type i = 0; i < rows; i++)
    {
        for (size_type j = 0; j < cols; j++)
        {
            result(i, j) = T(0);
        }
    }
    return result;
}

template<Arithmetic T>
inline void Matrix<T>::setZeroes()
{
    for (size_type i = 0; i < rows; i++)
    {
        for (size_type j = 0; j < cols; j++)
        {
            matrix[i][j] = T(0);
        }
    }
}

template<Arithmetic T>
inline Matrix<T> Matrix<T>::random(const size_type& size, const T& min, const T& max) {
    Matrix<T> result(size, size);
    // random generator
    std::random_device rd;  // start random number
    std::mt19937 gen(rd()); // Mersenne Twister for generation
    if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        for (size_type i = 0; i < size; ++i) {
            for (size_type j = 0; j < size; ++j) {
                result(i, j) = dist(gen);
            }
        }
    }
    else if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        for (size_type i = 0; i < size; ++i) {
            for (size_type j = 0; j < size; ++j) {
                result(i, j) = dist(gen);
            }
        }
    }
    else {
        static_assert(std::is_arithmetic<T>::value, "Matrix random generation requires an arithmetic type");
    }
    return result;
}

template<Arithmetic T>
inline Matrix<T> Matrix<T>::random(const size_type& rows, const size_type& cols, const T& min, const T& max) {
    Matrix<T> result(rows, cols);
    // random generator
    std::random_device rd;  // start random number
    std::mt19937 gen(rd()); // Mersenne Twister for generation
    if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        for (size_type i = 0; i < rows; ++i) {
            for (size_type j = 0; j < cols; ++j) {
                result(i, j) = dist(gen);
            }
        }
    }
    else if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        for (size_type i = 0; i < rows; ++i) {
            for (size_type j = 0; j < cols; ++j) {
                result(i, j) = dist(gen);
            }
        }
    }
    else {
        static_assert(std::is_arithmetic<T>::value, "Matrix random generation requires an arithmetic type");
    }
    return result;
}

template<Arithmetic T>
inline void Matrix<T>::setRandom(const T& min, const T& max)
{
    // random generator
    std::random_device rd;  // start random number
    std::mt19937 gen(rd()); // Mersenne Twister for generation
    if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        for (size_type i = 0; i < rows; ++i) {
            for (size_type j = 0; j < cols; ++j) {
                matrix(i, j) = dist(gen);
            }
        }
    }
    else if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        for (size_type i = 0; i < rows; ++i) {
            for (size_type j = 0; j < cols; ++j) {
                matrix(i, j) = dist(gen);
            }
        }
    }
    else {
        static_assert(std::is_arithmetic<T>::value, "Matrix random generation requires an arithmetic type");
    }
}

#endif // ! MINIMIZEROPTIMIZER_HEADERS_MATRIX_H_