#ifndef MINIMIZEROPTIMIZER_HEADERS_SPARSEMATRIX_H_
#define MINIMIZEROPTIMIZER_HEADERS_SPARSEMATRIX_H_

#include "Matrix.h"
#include <algorithm>
#include <numeric>

template <Arithmetic T>
struct Triplet {
    size_t row;
    size_t col;
    T value;

    Triplet() = default;
    Triplet(size_t r, size_t c, const T& v) : row(r), col(c), value(v) {}
};

template <Arithmetic T = double>
class SparseMatrix {
public:
    using size_type = size_t;
    using iterator_type = size_t;

    class InnerIterator {
    public:
        InnerIterator(const SparseMatrix& mat, size_type row)
            : mat_(mat), row_(row),
              pos_(row < mat.rows_ ? mat.outer_[row] : 0),
              end_(row < mat.rows_ ? mat.outer_[row + 1] : 0) {}

        InnerIterator& operator++() { ++pos_; return *this; }
        explicit operator bool() const { return pos_ < end_; }

        T value() const { return mat_.values_[pos_]; }
        T& valueRef() { return const_cast<T&>(mat_.values_[pos_]); }
        size_type row() const { return row_; }
        size_type col() const { return mat_.inner_[pos_]; }
        size_type index() const { return mat_.inner_[pos_]; }

    private:
        const SparseMatrix& mat_;
        size_type row_;
        size_type pos_;
        size_type end_;
    };

public:
    SparseMatrix() = default;
    SparseMatrix(const SparseMatrix& other) = default;
    SparseMatrix(SparseMatrix&& other) noexcept = default;
    ~SparseMatrix() = default;

    explicit SparseMatrix(size_type size);
    explicit SparseMatrix(size_type rows, size_type cols);
    explicit SparseMatrix(const Matrix<T>& dense);

    SparseMatrix& operator=(const SparseMatrix& other) = default;
    SparseMatrix& operator=(SparseMatrix&& other) noexcept = default;

    T operator()(iterator_type row, iterator_type col) const;
    T coeff(iterator_type row, iterator_type col) const;
    T& coeffRef(iterator_type row, iterator_type col);

    void insert(size_type row, size_type col, const T& value);

    template <typename InputIt>
    void setFromTriplets(InputIt first, InputIt last);

    void reserve(size_type nnz);
    void prune(const T& threshold = T(0));
    void setZero();

    SparseMatrix transpose() const;
    void setTranspose();

    T norm() const;
    static T norm(const SparseMatrix& mat);

    T trace() const;
    static T trace(const SparseMatrix& mat);

    std::vector<T> diag() const;
    static std::vector<T> diag(const SparseMatrix& mat);

    std::vector<T> getRow(const iterator_type& rowI) const;
    std::vector<T> getCol(const iterator_type& colI) const;
    void getRowEntries(size_type row,
                       std::vector<size_type>& cols,
                       std::vector<T>& values,
                       size_type start_col = 0) const;
    void replaceRow(size_type row,
                    const std::vector<size_type>& cols,
                    const std::vector<T>& values);

    SparseMatrix getSubmatrix(size_type start_row, size_type start_col,
                              size_type num_rows, size_type num_cols) const;

    Matrix<T> toDense() const;

    static SparseMatrix identity(size_type size);
    static SparseMatrix identity(size_type rows, size_type cols);
    static SparseMatrix zeroes(size_type size);
    static SparseMatrix zeroes(size_type rows, size_type cols);

    static SparseMatrix fromCSR(size_type rows, size_type cols,
                                std::vector<T> values,
                                std::vector<size_type> inner,
                                std::vector<size_type> outer);

    inline size_type rows_size() const { return rows_; }
    inline size_type cols_size() const { return cols_; }
    inline size_type nonZeros() const { return values_.size(); }

    inline const std::vector<T>& valueData() const { return values_; }
    inline const std::vector<size_type>& innerIndexData() const { return inner_; }
    inline const std::vector<size_type>& outerIndexData() const { return outer_; }

private:
    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<T> values_;
    std::vector<size_type> inner_;
    std::vector<size_type> outer_;
};

//////////////////////////////////////////////////////////////////////////// Constructors

template <Arithmetic T>
inline SparseMatrix<T>::SparseMatrix(size_type size)
    : rows_(size), cols_(size), outer_(size + 1, 0) {}

template <Arithmetic T>
inline SparseMatrix<T>::SparseMatrix(size_type rows, size_type cols)
    : rows_(rows), cols_(cols), outer_(rows + 1, 0) {}

template <Arithmetic T>
inline SparseMatrix<T>::SparseMatrix(const Matrix<T>& dense)
    : rows_(dense.rows_size()), cols_(dense.cols_size()), outer_(dense.rows_size() + 1, 0)
{
    for (size_type i = 0; i < rows_; ++i) {
        for (size_type j = 0; j < cols_; ++j) {
            T val = dense(i, j);
            if (val != T(0)) {
                values_.push_back(val);
                inner_.push_back(j);
            }
        }
        outer_[i + 1] = values_.size();
    }
}

//////////////////////////////////////////////////////////////////////////// Element access

template <Arithmetic T>
inline T SparseMatrix<T>::operator()(iterator_type row, iterator_type col) const {
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range("Index out of range");

    auto start = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[row]);
    auto end = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[row + 1]);
    auto it = std::lower_bound(start, end, col);

    if (it != end && *it == col)
        return values_[static_cast<size_type>(it - inner_.begin())];

    return T(0);
}

template <Arithmetic T>
inline T SparseMatrix<T>::coeff(iterator_type row, iterator_type col) const {
    return (*this)(row, col);
}

template <Arithmetic T>
inline T& SparseMatrix<T>::coeffRef(iterator_type row, iterator_type col) {
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range("Index out of range");

    auto start = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[row]);
    auto end = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[row + 1]);
    auto it = std::lower_bound(start, end, col);
    auto pos = static_cast<size_type>(it - inner_.begin());

    if (it != end && *it == col)
        return values_[pos];

    values_.insert(values_.begin() + static_cast<std::ptrdiff_t>(pos), T(0));
    inner_.insert(inner_.begin() + static_cast<std::ptrdiff_t>(pos), col);
    for (size_type i = row + 1; i <= rows_; ++i)
        outer_[i]++;

    return values_[pos];
}

//////////////////////////////////////////////////////////////////////////// Building

template <Arithmetic T>
inline void SparseMatrix<T>::insert(size_type row, size_type col, const T& value) {
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range("Index out of range");

    auto start = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[row]);
    auto end = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[row + 1]);
    auto it = std::lower_bound(start, end, col);
    auto pos = static_cast<size_type>(it - inner_.begin());

    if (it != end && *it == col) {
        values_[pos] = value;
        return;
    }

    values_.insert(values_.begin() + static_cast<std::ptrdiff_t>(pos), value);
    inner_.insert(inner_.begin() + static_cast<std::ptrdiff_t>(pos), col);
    for (size_type i = row + 1; i <= rows_; ++i)
        outer_[i]++;
}

template <Arithmetic T>
template <typename InputIt>
inline void SparseMatrix<T>::setFromTriplets(InputIt first, InputIt last) {
    using TripletType = typename std::iterator_traits<InputIt>::value_type;
    std::vector<TripletType> triplets(first, last);

    std::sort(triplets.begin(), triplets.end(), [](const auto& a, const auto& b) {
        return a.row < b.row || (a.row == b.row && a.col < b.col);
    });

    values_.clear();
    inner_.clear();
    outer_.assign(rows_ + 1, 0);

    size_type k = 0;
    for (size_type row = 0; row < rows_; ++row) {
        outer_[row] = values_.size();
        while (k < triplets.size() && triplets[k].row == row) {
            size_type c = triplets[k].col;
            if (c >= cols_)
                throw std::out_of_range("Triplet column index out of range");
            T val = T(0);
            while (k < triplets.size() && triplets[k].row == row && triplets[k].col == c) {
                val += triplets[k].value;
                ++k;
            }
            values_.push_back(val);
            inner_.push_back(c);
        }
    }
    outer_[rows_] = values_.size();
}

template <Arithmetic T>
inline void SparseMatrix<T>::reserve(size_type nnz) {
    values_.reserve(nnz);
    inner_.reserve(nnz);
}

template <Arithmetic T>
inline void SparseMatrix<T>::prune(const T& threshold) {
    std::vector<T> new_values;
    std::vector<size_type> new_inner;
    std::vector<size_type> new_outer(rows_ + 1, 0);

    for (size_type i = 0; i < rows_; ++i) {
        new_outer[i] = new_values.size();
        for (size_type k = outer_[i]; k < outer_[i + 1]; ++k) {
            if (std::abs(values_[k]) > threshold) {
                new_values.push_back(values_[k]);
                new_inner.push_back(inner_[k]);
            }
        }
    }
    new_outer[rows_] = new_values.size();

    values_ = std::move(new_values);
    inner_ = std::move(new_inner);
    outer_ = std::move(new_outer);
}

template <Arithmetic T>
inline void SparseMatrix<T>::setZero() {
    values_.clear();
    inner_.clear();
    std::fill(outer_.begin(), outer_.end(), size_type(0));
}

//////////////////////////////////////////////////////////////////////////// Transpose

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::transpose() const {
    size_type nnz = values_.size();

    std::vector<size_type> count(cols_, 0);
    for (size_type k = 0; k < nnz; ++k)
        count[inner_[k]]++;

    std::vector<size_type> new_outer(cols_ + 1, 0);
    for (size_type j = 0; j < cols_; ++j)
        new_outer[j + 1] = new_outer[j] + count[j];

    std::vector<T> new_values(nnz);
    std::vector<size_type> new_inner(nnz);
    std::vector<size_type> pos(cols_, 0);

    for (size_type i = 0; i < rows_; ++i) {
        for (size_type k = outer_[i]; k < outer_[i + 1]; ++k) {
            size_type j = inner_[k];
            size_type dest = new_outer[j] + pos[j];
            new_inner[dest] = i;
            new_values[dest] = values_[k];
            pos[j]++;
        }
    }

    return fromCSR(cols_, rows_,
                   std::move(new_values), std::move(new_inner), std::move(new_outer));
}

template <Arithmetic T>
inline void SparseMatrix<T>::setTranspose() {
    *this = transpose();
}

//////////////////////////////////////////////////////////////////////////// Norm, trace, diag

template <Arithmetic T>
inline T SparseMatrix<T>::norm() const {
    T sum = T(0);
    for (const auto& v : values_)
        sum += v * v;
    return std::sqrt(sum);
}

template <Arithmetic T>
inline T SparseMatrix<T>::norm(const SparseMatrix& mat) {
    return mat.norm();
}

template <Arithmetic T>
inline T SparseMatrix<T>::trace() const {
    T result = T(0);
    size_type min_dim = std::min(rows_, cols_);
    for (size_type i = 0; i < min_dim; ++i) {
        auto start = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[i]);
        auto end = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[i + 1]);
        auto it = std::lower_bound(start, end, i);
        if (it != end && *it == i)
            result += values_[static_cast<size_type>(it - inner_.begin())];
    }
    return result;
}

template <Arithmetic T>
inline T SparseMatrix<T>::trace(const SparseMatrix& mat) {
    return mat.trace();
}

template <Arithmetic T>
inline std::vector<T> SparseMatrix<T>::diag() const {
    size_type min_dim = std::min(rows_, cols_);
    std::vector<T> result(min_dim, T(0));
    for (size_type i = 0; i < min_dim; ++i) {
        auto start = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[i]);
        auto end = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[i + 1]);
        auto it = std::lower_bound(start, end, i);
        if (it != end && *it == i)
            result[i] = values_[static_cast<size_type>(it - inner_.begin())];
    }
    return result;
}

template <Arithmetic T>
inline std::vector<T> SparseMatrix<T>::diag(const SparseMatrix& mat) {
    return mat.diag();
}

//////////////////////////////////////////////////////////////////////////// Row, col, submatrix access

template <Arithmetic T>
inline std::vector<T> SparseMatrix<T>::getRow(const iterator_type& rowI) const {
    if (rowI >= rows_)
        throw std::out_of_range("Index out of range");
    std::vector<T> result(cols_, T(0));
    for (size_type k = outer_[rowI]; k < outer_[rowI + 1]; ++k)
        result[inner_[k]] = values_[k];
    return result;
}

template <Arithmetic T>
inline std::vector<T> SparseMatrix<T>::getCol(const iterator_type& colI) const {
    if (colI >= cols_)
        throw std::out_of_range("Index out of range");
    std::vector<T> result(rows_, T(0));
    for (size_type i = 0; i < rows_; ++i) {
        auto start = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[i]);
        auto end = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[i + 1]);
        auto it = std::lower_bound(start, end, colI);
        if (it != end && *it == colI)
            result[i] = values_[static_cast<size_type>(it - inner_.begin())];
    }
    return result;
}

template <Arithmetic T>
inline void SparseMatrix<T>::getRowEntries(
    size_type row,
    std::vector<size_type>& cols,
    std::vector<T>& values,
    size_type start_col) const
{
    if (row >= rows_ || start_col > cols_)
        throw std::out_of_range("Index out of range");

    auto begin = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[row]);
    auto end = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[row + 1]);
    auto start = std::lower_bound(begin, end, start_col);
    auto pos = static_cast<size_type>(start - inner_.begin());
    auto finish = outer_[row + 1];

    cols.assign(inner_.begin() + static_cast<std::ptrdiff_t>(pos),
                inner_.begin() + static_cast<std::ptrdiff_t>(finish));
    values.assign(values_.begin() + static_cast<std::ptrdiff_t>(pos),
                  values_.begin() + static_cast<std::ptrdiff_t>(finish));
}

template <Arithmetic T>
inline void SparseMatrix<T>::replaceRow(
    size_type row,
    const std::vector<size_type>& cols,
    const std::vector<T>& values)
{
    if (row >= rows_)
        throw std::out_of_range("Index out of range");
    if (cols.size() != values.size())
        throw std::invalid_argument("replaceRow: column/value sizes must match");

    for (size_type i = 0; i < cols.size(); ++i) {
        if (cols[i] >= cols_)
            throw std::out_of_range("replaceRow: column index out of range");
        if (i > 0 && cols[i - 1] >= cols[i])
            throw std::invalid_argument("replaceRow: column indices must be strictly increasing");
    }

    const size_type old_begin = outer_[row];
    const size_type old_end = outer_[row + 1];
    const size_type old_size = old_end - old_begin;
    const size_type new_size = cols.size();

    values_.erase(values_.begin() + static_cast<std::ptrdiff_t>(old_begin),
                  values_.begin() + static_cast<std::ptrdiff_t>(old_end));
    inner_.erase(inner_.begin() + static_cast<std::ptrdiff_t>(old_begin),
                 inner_.begin() + static_cast<std::ptrdiff_t>(old_end));

    values_.insert(values_.begin() + static_cast<std::ptrdiff_t>(old_begin),
                   values.begin(), values.end());
    inner_.insert(inner_.begin() + static_cast<std::ptrdiff_t>(old_begin),
                  cols.begin(), cols.end());

    if (new_size == old_size)
        return;

    const bool grows = new_size > old_size;
    const size_type delta = grows ? (new_size - old_size) : (old_size - new_size);
    for (size_type i = row + 1; i <= rows_; ++i) {
        outer_[i] = grows ? (outer_[i] + delta) : (outer_[i] - delta);
    }
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::getSubmatrix(
    size_type start_row, size_type start_col,
    size_type num_rows, size_type num_cols) const
{
    if (start_row + num_rows > rows_ || start_col + num_cols > cols_)
        throw std::out_of_range("getSubmatrix: Index out of range");

    std::vector<T> new_values;
    std::vector<size_type> new_inner;
    std::vector<size_type> new_outer(num_rows + 1, 0);

    for (size_type i = 0; i < num_rows; ++i) {
        new_outer[i] = new_values.size();
        size_type src_row = start_row + i;
        for (size_type k = outer_[src_row]; k < outer_[src_row + 1]; ++k) {
            size_type j = inner_[k];
            if (j >= start_col && j < start_col + num_cols) {
                new_values.push_back(values_[k]);
                new_inner.push_back(j - start_col);
            }
        }
    }
    new_outer[num_rows] = new_values.size();

    return fromCSR(num_rows, num_cols,
                   std::move(new_values), std::move(new_inner), std::move(new_outer));
}

//////////////////////////////////////////////////////////////////////////// toDense

template <Arithmetic T>
inline Matrix<T> SparseMatrix<T>::toDense() const {
    Matrix<T> result(rows_, cols_);
    for (size_type i = 0; i < rows_; ++i) {
        for (size_type k = outer_[i]; k < outer_[i + 1]; ++k) {
            result(i, inner_[k]) = values_[k];
        }
    }
    return result;
}

//////////////////////////////////////////////////////////////////////////// Factory methods

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::identity(size_type size) {
    return identity(size, size);
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::identity(size_type rows, size_type cols) {
    size_type min_dim = std::min(rows, cols);

    std::vector<T> values(min_dim, T(1));
    std::vector<size_type> inner(min_dim);
    std::iota(inner.begin(), inner.end(), size_type(0));
    std::vector<size_type> outer(rows + 1);
    for (size_type i = 0; i <= rows; ++i)
        outer[i] = std::min(i, min_dim);

    return fromCSR(rows, cols, std::move(values), std::move(inner), std::move(outer));
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::zeroes(size_type size) {
    return SparseMatrix(size);
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::zeroes(size_type rows, size_type cols) {
    return SparseMatrix(rows, cols);
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::fromCSR(
    size_type rows, size_type cols,
    std::vector<T> values,
    std::vector<size_type> inner,
    std::vector<size_type> outer)
{
    SparseMatrix result;
    result.rows_ = rows;
    result.cols_ = cols;
    result.values_ = std::move(values);
    result.inner_ = std::move(inner);
    result.outer_ = std::move(outer);
    return result;
}

//////////////////////////////////////////////////////////////////////////// Sparse + Sparse (merge-based)

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator+(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size())
        throw std::invalid_argument("Matrices must have the same size for addition.");

    using sz = typename SparseMatrix<V>::size_type;
    sz rows = A.rows_size();
    sz cols = A.cols_size();

    const auto& av = A.valueData();
    const auto& ai = A.innerIndexData();
    const auto& ao = A.outerIndexData();
    const auto& bv = B.valueData();
    const auto& bi = B.innerIndexData();
    const auto& bo = B.outerIndexData();

    std::vector<V> values;
    std::vector<sz> inner;
    std::vector<sz> outer(rows + 1, 0);

    values.reserve(A.nonZeros() + B.nonZeros());
    inner.reserve(A.nonZeros() + B.nonZeros());

    for (sz row = 0; row < rows; ++row) {
        outer[row] = values.size();
        sz pa = ao[row], ea = ao[row + 1];
        sz pb = bo[row], eb = bo[row + 1];

        while (pa < ea && pb < eb) {
            if (ai[pa] < bi[pb]) {
                values.push_back(av[pa]);
                inner.push_back(ai[pa]);
                ++pa;
            } else if (ai[pa] > bi[pb]) {
                values.push_back(bv[pb]);
                inner.push_back(bi[pb]);
                ++pb;
            } else {
                V sum = av[pa] + bv[pb];
                if (sum != V(0)) {
                    values.push_back(sum);
                    inner.push_back(ai[pa]);
                }
                ++pa; ++pb;
            }
        }
        while (pa < ea) { values.push_back(av[pa]); inner.push_back(ai[pa]); ++pa; }
        while (pb < eb) { values.push_back(bv[pb]); inner.push_back(bi[pb]); ++pb; }
    }
    outer[rows] = values.size();

    return SparseMatrix<V>::fromCSR(rows, cols, std::move(values), std::move(inner), std::move(outer));
}

//////////////////////////////////////////////////////////////////////////// Sparse - Sparse (merge-based)

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator-(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size())
        throw std::invalid_argument("Matrices must have the same size for subtraction.");

    using sz = typename SparseMatrix<V>::size_type;
    sz rows = A.rows_size();
    sz cols = A.cols_size();

    const auto& av = A.valueData();
    const auto& ai = A.innerIndexData();
    const auto& ao = A.outerIndexData();
    const auto& bv = B.valueData();
    const auto& bi = B.innerIndexData();
    const auto& bo = B.outerIndexData();

    std::vector<V> values;
    std::vector<sz> inner;
    std::vector<sz> outer(rows + 1, 0);

    values.reserve(A.nonZeros() + B.nonZeros());
    inner.reserve(A.nonZeros() + B.nonZeros());

    for (sz row = 0; row < rows; ++row) {
        outer[row] = values.size();
        sz pa = ao[row], ea = ao[row + 1];
        sz pb = bo[row], eb = bo[row + 1];

        while (pa < ea && pb < eb) {
            if (ai[pa] < bi[pb]) {
                values.push_back(av[pa]);
                inner.push_back(ai[pa]);
                ++pa;
            } else if (ai[pa] > bi[pb]) {
                values.push_back(-bv[pb]);
                inner.push_back(bi[pb]);
                ++pb;
            } else {
                V diff = av[pa] - bv[pb];
                if (diff != V(0)) {
                    values.push_back(diff);
                    inner.push_back(ai[pa]);
                }
                ++pa; ++pb;
            }
        }
        while (pa < ea) { values.push_back(av[pa]); inner.push_back(ai[pa]); ++pa; }
        while (pb < eb) { values.push_back(-bv[pb]); inner.push_back(bi[pb]); ++pb; }
    }
    outer[rows] = values.size();

    return SparseMatrix<V>::fromCSR(rows, cols, std::move(values), std::move(inner), std::move(outer));
}

//////////////////////////////////////////////////////////////////////////// Sparse * Sparse (Gustavson's algorithm)

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator*(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    if (A.cols_size() != B.rows_size())
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication.");

    using sz = typename SparseMatrix<V>::size_type;
    sz m = A.rows_size();
    sz n = B.cols_size();

    std::vector<V> w(n, V(0));
    std::vector<sz> marker(n, SIZE_MAX);

    std::vector<V> values;
    std::vector<sz> inner;
    std::vector<sz> outer(m + 1, 0);

    for (sz i = 0; i < m; ++i) {
        outer[i] = values.size();
        std::vector<sz> col_indices;

        for (typename SparseMatrix<V>::InnerIterator itA(A, i); itA; ++itA) {
            sz k = itA.col();
            V a_ik = itA.value();

            for (typename SparseMatrix<V>::InnerIterator itB(B, k); itB; ++itB) {
                sz j = itB.col();

                if (marker[j] != i) {
                    marker[j] = i;
                    w[j] = a_ik * itB.value();
                    col_indices.push_back(j);
                } else {
                    w[j] += a_ik * itB.value();
                }
            }
        }

        std::sort(col_indices.begin(), col_indices.end());

        for (sz j : col_indices) {
            if (w[j] != V(0)) {
                values.push_back(w[j]);
                inner.push_back(j);
            }
        }
    }
    outer[m] = values.size();

    return SparseMatrix<V>::fromCSR(m, n, std::move(values), std::move(inner), std::move(outer));
}

//////////////////////////////////////////////////////////////////////////// Unary minus

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator-(const SparseMatrix<V>& A) {
    std::vector<V> values(A.valueData());
    for (auto& v : values)
        v = -v;
    return SparseMatrix<V>::fromCSR(
        A.rows_size(), A.cols_size(),
        std::move(values),
        std::vector<typename SparseMatrix<V>::size_type>(A.innerIndexData()),
        std::vector<typename SparseMatrix<V>::size_type>(A.outerIndexData()));
}

//////////////////////////////////////////////////////////////////////////// Scalar operations

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator*(const SparseMatrix<V>& A, const V& scalar) {
    std::vector<V> values(A.valueData());
    for (auto& v : values)
        v *= scalar;
    return SparseMatrix<V>::fromCSR(
        A.rows_size(), A.cols_size(),
        std::move(values),
        std::vector<typename SparseMatrix<V>::size_type>(A.innerIndexData()),
        std::vector<typename SparseMatrix<V>::size_type>(A.outerIndexData()));
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator*(const V& scalar, const SparseMatrix<V>& A) {
    return A * scalar;
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator/(const SparseMatrix<V>& A, const V& scalar) {
    if (scalar == V(0))
        throw std::runtime_error("scalar shouldn't be zero");
    std::vector<V> values(A.valueData());
    for (auto& v : values)
        v /= scalar;
    return SparseMatrix<V>::fromCSR(
        A.rows_size(), A.cols_size(),
        std::move(values),
        std::vector<typename SparseMatrix<V>::size_type>(A.innerIndexData()),
        std::vector<typename SparseMatrix<V>::size_type>(A.outerIndexData()));
}

//////////////////////////////////////////////////////////////////////////// Compound assignment

template <Arithmetic V>
inline constexpr SparseMatrix<V>& operator+=(SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    A = A + B;
    return A;
}

template <Arithmetic V>
inline constexpr SparseMatrix<V>& operator-=(SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    A = A - B;
    return A;
}

template <Arithmetic V>
inline constexpr SparseMatrix<V>& operator*=(SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    A = A * B;
    return A;
}

template <Arithmetic V>
inline constexpr SparseMatrix<V>& operator*=(SparseMatrix<V>& A, const V& scalar) {
    A = A * scalar;
    return A;
}

template <Arithmetic V>
inline constexpr SparseMatrix<V>& operator/=(SparseMatrix<V>& A, const V& scalar) {
    A = A / scalar;
    return A;
}

//////////////////////////////////////////////////////////////////////////// Comparison

template <Arithmetic V>
inline constexpr bool operator==(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size())
        return false;

    const auto& ao = A.outerIndexData();
    const auto& ai = A.innerIndexData();
    const auto& av = A.valueData();
    const auto& bo = B.outerIndexData();
    const auto& bi = B.innerIndexData();
    const auto& bv = B.valueData();

    for (typename SparseMatrix<V>::size_type row = 0; row < A.rows_size(); ++row) {
        auto pa = ao[row], ea = ao[row + 1];
        auto pb = bo[row], eb = bo[row + 1];

        while (pa < ea && pb < eb) {
            while (pa < ea && av[pa] == V(0)) ++pa;
            while (pb < eb && bv[pb] == V(0)) ++pb;
            if (pa >= ea || pb >= eb) break;

            if (ai[pa] != bi[pb] || av[pa] != bv[pb])
                return false;
            ++pa; ++pb;
        }
        while (pa < ea) { if (av[pa] != V(0)) return false; ++pa; }
        while (pb < eb) { if (bv[pb] != V(0)) return false; ++pb; }
    }
    return true;
}

template <Arithmetic V>
inline constexpr bool operator!=(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    return !(A == B);
}

//////////////////////////////////////////////////////////////////////////// Sparse * Dense -> Dense

template <Arithmetic V>
inline constexpr Matrix<V> operator*(const SparseMatrix<V>& A, const Matrix<V>& B) {
    if (A.cols_size() != B.rows_size())
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication.");

    Matrix<V> C(A.rows_size(), B.cols_size());

    for (typename SparseMatrix<V>::size_type i = 0; i < A.rows_size(); ++i) {
        for (typename SparseMatrix<V>::InnerIterator it(A, i); it; ++it) {
            V a_val = it.value();
            auto k = it.col();
            for (typename Matrix<V>::size_type j = 0; j < B.cols_size(); ++j) {
                C(i, j) += a_val * B(k, j);
            }
        }
    }

    return C;
}

//////////////////////////////////////////////////////////////////////////// Dense * Sparse -> Dense

template <Arithmetic V>
inline constexpr Matrix<V> operator*(const Matrix<V>& A, const SparseMatrix<V>& B) {
    if (A.cols_size() != B.rows_size())
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication.");

    Matrix<V> C(A.rows_size(), B.cols_size());

    for (typename SparseMatrix<V>::size_type k = 0; k < B.rows_size(); ++k) {
        for (typename SparseMatrix<V>::InnerIterator it(B, k); it; ++it) {
            V b_val = it.value();
            auto j = it.col();
            for (typename Matrix<V>::size_type i = 0; i < A.rows_size(); ++i) {
                C(i, j) += A(i, k) * b_val;
            }
        }
    }

    return C;
}

//////////////////////////////////////////////////////////////////////////// Stream output

template <Arithmetic V>
std::ostream& operator<<(std::ostream& os, const SparseMatrix<V>& mat) {
    os << "SparseMatrix(" << mat.rows_size() << "x" << mat.cols_size()
       << ", nnz=" << mat.nonZeros() << ")\n";
    for (typename SparseMatrix<V>::size_type i = 0; i < mat.rows_size(); ++i) {
        for (typename SparseMatrix<V>::InnerIterator it(mat, i); it; ++it) {
            os << "  (" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
        }
    }
    return os;
}

#endif // !MINIMIZEROPTIMIZER_HEADERS_SPARSEMATRIX_H_
