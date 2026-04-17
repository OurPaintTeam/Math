#ifndef MINIMIZEROPTIMIZER_HEADERS_SPARSEMATRIX_H_
#define MINIMIZEROPTIMIZER_HEADERS_SPARSEMATRIX_H_

#include "Matrix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

template <Arithmetic T>
struct Triplet {
    size_t row = 0;
    size_t col = 0;
    T value = T(0);

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
        InnerIterator(const SparseMatrix& mat, size_type col)
            : mat_(mat), col_(col),
              pos_(col < mat.cols_ ? mat.outer_[col] : 0),
              end_(col < mat.cols_ ? mat.outer_[col + 1] : 0) {}

        InnerIterator& operator++() {
            ++pos_;
            return *this;
        }

        explicit operator bool() const { return pos_ < end_; }

        T value() const { return mat_.values_[pos_]; }
        T& valueRef() { return const_cast<T&>(mat_.values_[pos_]); }
        size_type row() const { return mat_.inner_[pos_]; }
        size_type col() const { return col_; }
        size_type index() const { return mat_.inner_[pos_]; }

    private:
        const SparseMatrix& mat_;
        size_type col_;
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
    void makeRowMutable();
    const std::vector<size_type>& rowIndices(size_type row) const;
    const std::vector<T>& rowValues(size_type row) const;
    void swapRowBuffers(size_type row,
                        std::vector<size_type>& cols,
                        std::vector<T>& values);
    void compressRowMutable();

    SparseMatrix getSubmatrix(size_type start_row, size_type start_col,
                              size_type num_rows, size_type num_cols) const;

    Matrix<T> toDense() const;
    SparseMatrix compressed() const;

    static SparseMatrix identity(size_type size);
    static SparseMatrix identity(size_type rows, size_type cols);
    static SparseMatrix zeroes(size_type size);
    static SparseMatrix zeroes(size_type rows, size_type cols);

    static SparseMatrix fromCSC(size_type rows, size_type cols,
                                std::vector<T> values,
                                std::vector<size_type> inner,
                                std::vector<size_type> outer);
    static SparseMatrix fromCSR(size_type rows, size_type cols,
                                std::vector<T> values,
                                std::vector<size_type> inner,
                                std::vector<size_type> outer);

    [[nodiscard]] size_type rows_size() const noexcept { return rows_; }
    [[nodiscard]] size_type cols_size() const noexcept { return cols_; }
    [[nodiscard]] size_type nonZeros() const noexcept;
    [[nodiscard]] bool rowMutableMode() const noexcept { return row_mutable_; }

    [[nodiscard]] const std::vector<T>& valueData() const noexcept { return values_; }
    [[nodiscard]] const std::vector<size_type>& innerIndexData() const noexcept { return inner_; }
    [[nodiscard]] const std::vector<size_type>& outerIndexData() const noexcept { return outer_; }

private:
    [[nodiscard]] size_type rowMutableNonZeros() const noexcept;
    void validateIndices(size_type row, size_type col) const;
    [[nodiscard]] size_type lowerBoundPosition(size_type col, size_type row) const;

private:
    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<T> values_;
    std::vector<size_type> inner_;
    std::vector<size_type> outer_;
    bool row_mutable_ = false;
    std::vector<std::vector<size_type>> row_inner_;
    std::vector<std::vector<T>> row_values_;
};

template <Arithmetic T>
inline SparseMatrix<T>::SparseMatrix(size_type size)
    : rows_(size), cols_(size), outer_(size + 1, 0) {}

template <Arithmetic T>
inline SparseMatrix<T>::SparseMatrix(size_type rows, size_type cols)
    : rows_(rows), cols_(cols), outer_(cols + 1, 0) {}

template <Arithmetic T>
inline SparseMatrix<T>::SparseMatrix(const Matrix<T>& dense)
    : rows_(dense.rows_size()), cols_(dense.cols_size()), outer_(dense.cols_size() + 1, 0)
{
    for (size_type col = 0; col < cols_; ++col) {
        outer_[col] = values_.size();
        for (size_type row = 0; row < rows_; ++row) {
            const T value = dense(row, col);
            if (value != T(0)) {
                values_.push_back(value);
                inner_.push_back(row);
            }
        }
    }
    outer_[cols_] = values_.size();
}

template <Arithmetic T>
inline typename SparseMatrix<T>::size_type SparseMatrix<T>::rowMutableNonZeros() const noexcept {
    size_type nnz = 0;
    for (const auto& cols : row_inner_) {
        nnz += cols.size();
    }
    return nnz;
}

template <Arithmetic T>
inline typename SparseMatrix<T>::size_type SparseMatrix<T>::nonZeros() const noexcept {
    return row_mutable_ ? rowMutableNonZeros() : values_.size();
}

template <Arithmetic T>
inline void SparseMatrix<T>::validateIndices(size_type row, size_type col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Index out of range");
    }
}

template <Arithmetic T>
inline typename SparseMatrix<T>::size_type SparseMatrix<T>::lowerBoundPosition(
    size_type col,
    size_type row) const
{
    const auto begin = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[col]);
    const auto end = inner_.begin() + static_cast<std::ptrdiff_t>(outer_[col + 1]);
    return static_cast<size_type>(std::lower_bound(begin, end, row) - inner_.begin());
}

template <Arithmetic T>
inline T SparseMatrix<T>::operator()(iterator_type row, iterator_type col) const {
    validateIndices(row, col);

    if (row_mutable_) {
        const auto& row_cols = row_inner_[row];
        const auto& row_vals = row_values_[row];
        for (size_type k = 0; k < row_cols.size(); ++k) {
            if (row_cols[k] == col) {
                return row_vals[k];
            }
        }
        return T(0);
    }

    const size_type pos = lowerBoundPosition(col, row);
    if (pos < outer_[col + 1] && inner_[pos] == row) {
        return values_[pos];
    }
    return T(0);
}

template <Arithmetic T>
inline T SparseMatrix<T>::coeff(iterator_type row, iterator_type col) const {
    return (*this)(row, col);
}

template <Arithmetic T>
inline T& SparseMatrix<T>::coeffRef(iterator_type row, iterator_type col) {
    validateIndices(row, col);

    if (row_mutable_) {
        auto& row_cols = row_inner_[row];
        auto& row_vals = row_values_[row];
        for (size_type k = 0; k < row_cols.size(); ++k) {
            if (row_cols[k] == col) {
                return row_vals[k];
            }
        }
        row_cols.push_back(col);
        row_vals.push_back(T(0));
        return row_vals.back();
    }

    const size_type pos = lowerBoundPosition(col, row);
    if (pos < outer_[col + 1] && inner_[pos] == row) {
        return values_[pos];
    }

    values_.insert(values_.begin() + static_cast<std::ptrdiff_t>(pos), T(0));
    inner_.insert(inner_.begin() + static_cast<std::ptrdiff_t>(pos), row);
    for (size_type next_col = col + 1; next_col <= cols_; ++next_col) {
        ++outer_[next_col];
    }
    return values_[pos];
}

template <Arithmetic T>
inline void SparseMatrix<T>::insert(size_type row, size_type col, const T& value) {
    coeffRef(row, col) = value;
}

template <Arithmetic T>
inline void SparseMatrix<T>::reserve(size_type nnz) {
    values_.reserve(nnz);
    inner_.reserve(nnz);
}

template <Arithmetic T>
inline void SparseMatrix<T>::setZero() {
    values_.clear();
    inner_.clear();
    outer_.assign(cols_ + 1, 0);
    row_inner_.clear();
    row_values_.clear();
    row_mutable_ = false;
}

template <Arithmetic T>
template <typename InputIt>
inline void SparseMatrix<T>::setFromTriplets(InputIt first, InputIt last) {
    using TripletType = typename std::iterator_traits<InputIt>::value_type;
    std::vector<TripletType> triplets(first, last);

    std::sort(triplets.begin(), triplets.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.col < rhs.col || (lhs.col == rhs.col && lhs.row < rhs.row);
    });

    values_.clear();
    inner_.clear();
    outer_.assign(cols_ + 1, 0);
    row_inner_.clear();
    row_values_.clear();
    row_mutable_ = false;

    size_type cursor = 0;
    for (size_type col = 0; col < cols_; ++col) {
        outer_[col] = values_.size();
        while (cursor < triplets.size() && triplets[cursor].col == col) {
            if (triplets[cursor].row >= rows_) {
                throw std::out_of_range("Triplet row index out of range");
            }

            const size_type row = triplets[cursor].row;
            T value = T(0);
            while (cursor < triplets.size()
                   && triplets[cursor].col == col
                   && triplets[cursor].row == row) {
                value += triplets[cursor].value;
                ++cursor;
            }

            if (value != T(0)) {
                inner_.push_back(row);
                values_.push_back(value);
            }
        }
    }
    outer_[cols_] = values_.size();
}

template <Arithmetic T>
inline void SparseMatrix<T>::prune(const T& threshold) {
    if (row_mutable_) {
        for (size_type row = 0; row < rows_; ++row) {
            std::vector<size_type> kept_cols;
            std::vector<T> kept_vals;
            kept_cols.reserve(row_inner_[row].size());
            kept_vals.reserve(row_values_[row].size());
            for (size_type k = 0; k < row_inner_[row].size(); ++k) {
                if (std::abs(row_values_[row][k]) > threshold) {
                    kept_cols.push_back(row_inner_[row][k]);
                    kept_vals.push_back(row_values_[row][k]);
                }
            }
            row_inner_[row] = std::move(kept_cols);
            row_values_[row] = std::move(kept_vals);
        }
        return;
    }

    std::vector<T> new_values;
    std::vector<size_type> new_inner;
    std::vector<size_type> new_outer(cols_ + 1, 0);
    new_values.reserve(values_.size());
    new_inner.reserve(inner_.size());

    for (size_type col = 0; col < cols_; ++col) {
        new_outer[col] = new_values.size();
        for (size_type pos = outer_[col]; pos < outer_[col + 1]; ++pos) {
            if (std::abs(values_[pos]) > threshold) {
                new_values.push_back(values_[pos]);
                new_inner.push_back(inner_[pos]);
            }
        }
    }
    new_outer[cols_] = new_values.size();

    values_ = std::move(new_values);
    inner_ = std::move(new_inner);
    outer_ = std::move(new_outer);
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::compressed() const {
    if (!row_mutable_) {
        return *this;
    }

    SparseMatrix result(rows_, cols_);
    std::vector<Triplet<T>> triplets;
    triplets.reserve(rowMutableNonZeros());
    for (size_type row = 0; row < rows_; ++row) {
        for (size_type k = 0; k < row_inner_[row].size(); ++k) {
            const size_type col = row_inner_[row][k];
            if (col >= cols_) {
                throw std::out_of_range("row-mutable column index out of range");
            }
            const T value = row_values_[row][k];
            if (value != T(0)) {
                triplets.emplace_back(row, col, value);
            }
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::transpose() const {
    if (row_mutable_) {
        return compressed().transpose();
    }

    std::vector<size_type> counts(rows_, 0);
    for (const size_type row : inner_) {
        ++counts[row];
    }

    std::vector<size_type> new_outer(rows_ + 1, 0);
    for (size_type col = 0; col < rows_; ++col) {
        new_outer[col + 1] = new_outer[col] + counts[col];
    }

    std::vector<T> new_values(values_.size());
    std::vector<size_type> new_inner(inner_.size());
    std::vector<size_type> next = new_outer;

    for (size_type col = 0; col < cols_; ++col) {
        for (size_type pos = outer_[col]; pos < outer_[col + 1]; ++pos) {
            const size_type row = inner_[pos];
            const size_type dst = next[row]++;
            new_inner[dst] = col;
            new_values[dst] = values_[pos];
        }
    }

    return fromCSC(cols_, rows_, std::move(new_values), std::move(new_inner), std::move(new_outer));
}

template <Arithmetic T>
inline void SparseMatrix<T>::setTranspose() {
    *this = transpose();
}

template <Arithmetic T>
inline T SparseMatrix<T>::norm() const {
    T sum = T(0);
    if (row_mutable_) {
        for (const auto& row_vals : row_values_) {
            for (const T value : row_vals) {
                sum += value * value;
            }
        }
        return std::sqrt(sum);
    }

    for (const T value : values_) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

template <Arithmetic T>
inline T SparseMatrix<T>::norm(const SparseMatrix& mat) {
    return mat.norm();
}

template <Arithmetic T>
inline T SparseMatrix<T>::trace() const {
    if (row_mutable_) {
        return compressed().trace();
    }

    T result = T(0);
    const size_type min_dim = std::min(rows_, cols_);
    for (size_type diag = 0; diag < min_dim; ++diag) {
        const size_type pos = lowerBoundPosition(diag, diag);
        if (pos < outer_[diag + 1] && inner_[pos] == diag) {
            result += values_[pos];
        }
    }
    return result;
}

template <Arithmetic T>
inline T SparseMatrix<T>::trace(const SparseMatrix& mat) {
    return mat.trace();
}

template <Arithmetic T>
inline std::vector<T> SparseMatrix<T>::diag() const {
    if (row_mutable_) {
        return compressed().diag();
    }

    const size_type min_dim = std::min(rows_, cols_);
    std::vector<T> result(min_dim, T(0));
    for (size_type diag = 0; diag < min_dim; ++diag) {
        const size_type pos = lowerBoundPosition(diag, diag);
        if (pos < outer_[diag + 1] && inner_[pos] == diag) {
            result[diag] = values_[pos];
        }
    }
    return result;
}

template <Arithmetic T>
inline std::vector<T> SparseMatrix<T>::diag(const SparseMatrix& mat) {
    return mat.diag();
}

template <Arithmetic T>
inline std::vector<T> SparseMatrix<T>::getRow(const iterator_type& rowI) const {
    if (rowI >= rows_) {
        throw std::out_of_range("Index out of range");
    }

    std::vector<T> result(cols_, T(0));
    if (row_mutable_) {
        for (size_type k = 0; k < row_inner_[rowI].size(); ++k) {
            result[row_inner_[rowI][k]] += row_values_[rowI][k];
        }
        return result;
    }

    for (size_type col = 0; col < cols_; ++col) {
        const size_type pos = lowerBoundPosition(col, rowI);
        if (pos < outer_[col + 1] && inner_[pos] == rowI) {
            result[col] = values_[pos];
        }
    }
    return result;
}

template <Arithmetic T>
inline std::vector<T> SparseMatrix<T>::getCol(const iterator_type& colI) const {
    if (colI >= cols_) {
        throw std::out_of_range("Index out of range");
    }

    std::vector<T> result(rows_, T(0));
    if (row_mutable_) {
        return compressed().getCol(colI);
    }

    for (size_type pos = outer_[colI]; pos < outer_[colI + 1]; ++pos) {
        result[inner_[pos]] = values_[pos];
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
    if (row >= rows_ || start_col > cols_) {
        throw std::out_of_range("Index out of range");
    }

    cols.clear();
    values.clear();

    if (row_mutable_) {
        std::vector<std::pair<size_type, T>> pairs;
        for (size_type k = 0; k < row_inner_[row].size(); ++k) {
            if (row_inner_[row][k] >= start_col && row_values_[row][k] != T(0)) {
                pairs.emplace_back(row_inner_[row][k], row_values_[row][k]);
            }
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
        for (const auto& [col, value] : pairs) {
            if (!cols.empty() && cols.back() == col) {
                values.back() += value;
            } else if (value != T(0)) {
                cols.push_back(col);
                values.push_back(value);
            }
        }
        return;
    }

    for (size_type col = start_col; col < cols_; ++col) {
        const size_type pos = lowerBoundPosition(col, row);
        if (pos < outer_[col + 1] && inner_[pos] == row) {
            cols.push_back(col);
            values.push_back(values_[pos]);
        }
    }
}

template <Arithmetic T>
inline void SparseMatrix<T>::replaceRow(
    size_type row,
    const std::vector<size_type>& cols,
    const std::vector<T>& values)
{
    if (row >= rows_) {
        throw std::out_of_range("Index out of range");
    }
    if (cols.size() != values.size()) {
        throw std::invalid_argument("replaceRow: column/value sizes must match");
    }
    for (size_type i = 0; i < cols.size(); ++i) {
        if (cols[i] >= cols_) {
            throw std::out_of_range("replaceRow: column index out of range");
        }
        if (i > 0 && cols[i - 1] >= cols[i]) {
            throw std::invalid_argument("replaceRow: column indices must be strictly increasing");
        }
    }

    if (!row_mutable_) {
        makeRowMutable();
        row_inner_[row] = cols;
        row_values_[row] = values;
        compressRowMutable();
        return;
    }
    row_inner_[row] = cols;
    row_values_[row] = values;
}

template <Arithmetic T>
inline void SparseMatrix<T>::makeRowMutable() {
    if (row_mutable_) {
        return;
    }

    row_inner_.assign(rows_, {});
    row_values_.assign(rows_, {});
    for (size_type col = 0; col < cols_; ++col) {
        for (size_type pos = outer_[col]; pos < outer_[col + 1]; ++pos) {
            row_inner_[inner_[pos]].push_back(col);
            row_values_[inner_[pos]].push_back(values_[pos]);
        }
    }
    row_mutable_ = true;
}

template <Arithmetic T>
inline const std::vector<typename SparseMatrix<T>::size_type>& SparseMatrix<T>::rowIndices(size_type row) const {
    if (row >= rows_) {
        throw std::out_of_range("Index out of range");
    }
    if (!row_mutable_) {
        throw std::runtime_error("Call makeRowMutable() before rowIndices().");
    }
    return row_inner_[row];
}

template <Arithmetic T>
inline const std::vector<T>& SparseMatrix<T>::rowValues(size_type row) const {
    if (row >= rows_) {
        throw std::out_of_range("Index out of range");
    }
    if (!row_mutable_) {
        throw std::runtime_error("Call makeRowMutable() before rowValues().");
    }
    return row_values_[row];
}

template <Arithmetic T>
inline void SparseMatrix<T>::swapRowBuffers(
    size_type row,
    std::vector<size_type>& cols,
    std::vector<T>& values)
{
    if (row >= rows_) {
        throw std::out_of_range("Index out of range");
    }
    if (!row_mutable_) {
        throw std::runtime_error("Call makeRowMutable() before swapRowBuffers().");
    }
    if (cols.size() != values.size()) {
        throw std::invalid_argument("swapRowBuffers: column/value sizes must match");
    }

    row_inner_[row].swap(cols);
    row_values_[row].swap(values);
}

template <Arithmetic T>
inline void SparseMatrix<T>::compressRowMutable() {
    if (!row_mutable_) {
        return;
    }
    *this = compressed();
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::getSubmatrix(
    size_type start_row,
    size_type start_col,
    size_type num_rows,
    size_type num_cols) const
{
    if (start_row + num_rows > rows_ || start_col + num_cols > cols_) {
        throw std::out_of_range("getSubmatrix: Index out of range");
    }

    if (row_mutable_) {
        return compressed().getSubmatrix(start_row, start_col, num_rows, num_cols);
    }

    std::vector<T> new_values;
    std::vector<size_type> new_inner;
    std::vector<size_type> new_outer(num_cols + 1, 0);

    for (size_type local_col = 0; local_col < num_cols; ++local_col) {
        new_outer[local_col] = new_values.size();
        const size_type src_col = start_col + local_col;
        for (size_type pos = outer_[src_col]; pos < outer_[src_col + 1]; ++pos) {
            const size_type row = inner_[pos];
            if (row >= start_row && row < start_row + num_rows) {
                new_inner.push_back(row - start_row);
                new_values.push_back(values_[pos]);
            }
        }
    }
    new_outer[num_cols] = new_values.size();

    return fromCSC(num_rows, num_cols, std::move(new_values), std::move(new_inner), std::move(new_outer));
}

template <Arithmetic T>
inline Matrix<T> SparseMatrix<T>::toDense() const {
    Matrix<T> result(rows_, cols_);
    if (row_mutable_) {
        for (size_type row = 0; row < rows_; ++row) {
            for (size_type k = 0; k < row_inner_[row].size(); ++k) {
                result(row, row_inner_[row][k]) += row_values_[row][k];
            }
        }
        return result;
    }

    for (size_type col = 0; col < cols_; ++col) {
        for (size_type pos = outer_[col]; pos < outer_[col + 1]; ++pos) {
            result(inner_[pos], col) = values_[pos];
        }
    }
    return result;
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::identity(size_type size) {
    return identity(size, size);
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::identity(size_type rows, size_type cols) {
    const size_type min_dim = std::min(rows, cols);
    std::vector<T> values(min_dim, T(1));
    std::vector<size_type> inner(min_dim, 0);
    std::vector<size_type> outer(cols + 1, 0);

    for (size_type col = 0; col < cols; ++col) {
        outer[col] = std::min(col, min_dim);
        if (col < min_dim) {
            inner[col] = col;
        }
    }
    outer[cols] = min_dim;

    return fromCSC(rows, cols, std::move(values), std::move(inner), std::move(outer));
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
inline SparseMatrix<T> SparseMatrix<T>::fromCSC(
    size_type rows,
    size_type cols,
    std::vector<T> values,
    std::vector<size_type> inner,
    std::vector<size_type> outer)
{
    if (outer.size() != cols + 1) {
        throw std::invalid_argument("fromCSC: outer size must equal cols + 1");
    }
    if (values.size() != inner.size() || outer.back() != values.size()) {
        throw std::invalid_argument("fromCSC: inconsistent compressed buffers");
    }

    for (size_type col = 0; col < cols; ++col) {
        if (outer[col] > outer[col + 1]) {
            throw std::invalid_argument("fromCSC: outer offsets must be non-decreasing");
        }
        bool have_prev = false;
        size_type prev_row = 0;
        for (size_type pos = outer[col]; pos < outer[col + 1]; ++pos) {
            if (inner[pos] >= rows) {
                throw std::out_of_range("fromCSC: row index out of range");
            }
            if (have_prev && inner[pos] <= prev_row) {
                throw std::invalid_argument("fromCSC: column rows must be strictly increasing");
            }
            prev_row = inner[pos];
            have_prev = true;
        }
    }

    SparseMatrix result;
    result.rows_ = rows;
    result.cols_ = cols;
    result.values_ = std::move(values);
    result.inner_ = std::move(inner);
    result.outer_ = std::move(outer);
    return result;
}

template <Arithmetic T>
inline SparseMatrix<T> SparseMatrix<T>::fromCSR(
    size_type rows,
    size_type cols,
    std::vector<T> values,
    std::vector<size_type> inner,
    std::vector<size_type> outer)
{
    if (outer.size() != rows + 1) {
        throw std::invalid_argument("fromCSR: outer size must equal rows + 1");
    }
    if (values.size() != inner.size() || outer.back() != values.size()) {
        throw std::invalid_argument("fromCSR: inconsistent compressed buffers");
    }

    SparseMatrix result(rows, cols);
    std::vector<Triplet<T>> triplets;
    triplets.reserve(values.size());
    for (size_type row = 0; row < rows; ++row) {
        if (outer[row] > outer[row + 1]) {
            throw std::invalid_argument("fromCSR: outer offsets must be non-decreasing");
        }
        for (size_type pos = outer[row]; pos < outer[row + 1]; ++pos) {
            if (inner[pos] >= cols) {
                throw std::out_of_range("fromCSR: column index out of range");
            }
            if (values[pos] != T(0)) {
                triplets.emplace_back(row, inner[pos], values[pos]);
            }
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator+(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) {
        throw std::invalid_argument("Matrices must have the same size for addition.");
    }

    SparseMatrix<V> lhs_buffer;
    SparseMatrix<V> rhs_buffer;
    const SparseMatrix<V>* lhs = &A;
    const SparseMatrix<V>* rhs = &B;
    if (A.rowMutableMode()) {
        lhs_buffer = A.compressed();
        lhs = &lhs_buffer;
    }
    if (B.rowMutableMode()) {
        rhs_buffer = B.compressed();
        rhs = &rhs_buffer;
    }

    using sz = typename SparseMatrix<V>::size_type;
    std::vector<V> values;
    std::vector<sz> inner;
    std::vector<sz> outer(lhs->cols_size() + 1, 0);
    values.reserve(lhs->nonZeros() + rhs->nonZeros());
    inner.reserve(lhs->nonZeros() + rhs->nonZeros());

    for (sz col = 0; col < lhs->cols_size(); ++col) {
        outer[col] = values.size();
        sz pa = lhs->outerIndexData()[col];
        sz ea = lhs->outerIndexData()[col + 1];
        sz pb = rhs->outerIndexData()[col];
        sz eb = rhs->outerIndexData()[col + 1];

        while (pa < ea && pb < eb) {
            if (lhs->innerIndexData()[pa] < rhs->innerIndexData()[pb]) {
                inner.push_back(lhs->innerIndexData()[pa]);
                values.push_back(lhs->valueData()[pa]);
                ++pa;
            } else if (lhs->innerIndexData()[pa] > rhs->innerIndexData()[pb]) {
                inner.push_back(rhs->innerIndexData()[pb]);
                values.push_back(rhs->valueData()[pb]);
                ++pb;
            } else {
                const V sum = lhs->valueData()[pa] + rhs->valueData()[pb];
                if (sum != V(0)) {
                    inner.push_back(lhs->innerIndexData()[pa]);
                    values.push_back(sum);
                }
                ++pa;
                ++pb;
            }
        }

        while (pa < ea) {
            inner.push_back(lhs->innerIndexData()[pa]);
            values.push_back(lhs->valueData()[pa]);
            ++pa;
        }
        while (pb < eb) {
            inner.push_back(rhs->innerIndexData()[pb]);
            values.push_back(rhs->valueData()[pb]);
            ++pb;
        }
    }
    outer[lhs->cols_size()] = values.size();

    return SparseMatrix<V>::fromCSC(lhs->rows_size(), lhs->cols_size(), std::move(values), std::move(inner), std::move(outer));
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator-(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) {
        throw std::invalid_argument("Matrices must have the same size for subtraction.");
    }
    return A + (-B);
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator*(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    if (A.cols_size() != B.rows_size()) {
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication.");
    }

    SparseMatrix<V> lhs_buffer;
    SparseMatrix<V> rhs_buffer;
    const SparseMatrix<V>* lhs = &A;
    const SparseMatrix<V>* rhs = &B;
    if (A.rowMutableMode()) {
        lhs_buffer = A.compressed();
        lhs = &lhs_buffer;
    }
    if (B.rowMutableMode()) {
        rhs_buffer = B.compressed();
        rhs = &rhs_buffer;
    }

    using sz = typename SparseMatrix<V>::size_type;
    const sz m = lhs->rows_size();
    const sz n = rhs->cols_size();

    std::vector<V> accumulator(m, V(0));
    std::vector<sz> marker(m, 0);
    std::vector<V> values;
    std::vector<sz> inner;
    std::vector<sz> outer(n + 1, 0);
    size_t stamp = 1;

    for (sz col = 0; col < n; ++col) {
        if (stamp == 0) {
            std::fill(marker.begin(), marker.end(), 0);
            stamp = 1;
        }

        outer[col] = values.size();
        std::vector<sz> touched_rows;

        for (typename SparseMatrix<V>::InnerIterator itB(*rhs, col); itB; ++itB) {
            const sz k = itB.row();
            const V b_value = itB.value();
            for (typename SparseMatrix<V>::InnerIterator itA(*lhs, k); itA; ++itA) {
                const sz row = itA.row();
                if (marker[row] != stamp) {
                    marker[row] = stamp;
                    accumulator[row] = itA.value() * b_value;
                    touched_rows.push_back(row);
                } else {
                    accumulator[row] += itA.value() * b_value;
                }
            }
        }

        std::sort(touched_rows.begin(), touched_rows.end());
        for (const sz row : touched_rows) {
            if (accumulator[row] != V(0)) {
                inner.push_back(row);
                values.push_back(accumulator[row]);
            }
            accumulator[row] = V(0);
        }
        ++stamp;
    }
    outer[n] = values.size();

    return SparseMatrix<V>::fromCSC(m, n, std::move(values), std::move(inner), std::move(outer));
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator-(const SparseMatrix<V>& A) {
    SparseMatrix<V> buffer;
    const SparseMatrix<V>* matrix = &A;
    if (A.rowMutableMode()) {
        buffer = A.compressed();
        matrix = &buffer;
    }

    std::vector<V> values(matrix->valueData());
    for (V& value : values) {
        value = -value;
    }
    return SparseMatrix<V>::fromCSC(
        matrix->rows_size(),
        matrix->cols_size(),
        std::move(values),
        std::vector<typename SparseMatrix<V>::size_type>(matrix->innerIndexData()),
        std::vector<typename SparseMatrix<V>::size_type>(matrix->outerIndexData()));
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator*(const SparseMatrix<V>& A, const V& scalar) {
    SparseMatrix<V> buffer;
    const SparseMatrix<V>* matrix = &A;
    if (A.rowMutableMode()) {
        buffer = A.compressed();
        matrix = &buffer;
    }

    std::vector<V> values(matrix->valueData());
    for (V& value : values) {
        value *= scalar;
    }
    return SparseMatrix<V>::fromCSC(
        matrix->rows_size(),
        matrix->cols_size(),
        std::move(values),
        std::vector<typename SparseMatrix<V>::size_type>(matrix->innerIndexData()),
        std::vector<typename SparseMatrix<V>::size_type>(matrix->outerIndexData()));
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator*(const V& scalar, const SparseMatrix<V>& A) {
    return A * scalar;
}

template <Arithmetic V>
inline constexpr SparseMatrix<V> operator/(const SparseMatrix<V>& A, const V& scalar) {
    if (scalar == V(0)) {
        throw std::runtime_error("scalar shouldn't be zero");
    }
    return A * (V(1) / scalar);
}

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

template <Arithmetic V>
inline constexpr bool operator==(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    if (A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size()) {
        return false;
    }

    SparseMatrix<V> lhs_buffer;
    SparseMatrix<V> rhs_buffer;
    const SparseMatrix<V>* lhs = &A;
    const SparseMatrix<V>* rhs = &B;
    if (A.rowMutableMode()) {
        lhs_buffer = A.compressed();
        lhs = &lhs_buffer;
    }
    if (B.rowMutableMode()) {
        rhs_buffer = B.compressed();
        rhs = &rhs_buffer;
    }

    for (typename SparseMatrix<V>::size_type col = 0; col < lhs->cols_size(); ++col) {
        auto pa = lhs->outerIndexData()[col];
        auto ea = lhs->outerIndexData()[col + 1];
        auto pb = rhs->outerIndexData()[col];
        auto eb = rhs->outerIndexData()[col + 1];

        while (pa < ea && pb < eb) {
            while (pa < ea && lhs->valueData()[pa] == V(0)) {
                ++pa;
            }
            while (pb < eb && rhs->valueData()[pb] == V(0)) {
                ++pb;
            }
            if (pa >= ea || pb >= eb) {
                break;
            }
            if (lhs->innerIndexData()[pa] != rhs->innerIndexData()[pb]
                || lhs->valueData()[pa] != rhs->valueData()[pb]) {
                return false;
            }
            ++pa;
            ++pb;
        }

        while (pa < ea) {
            if (lhs->valueData()[pa] != V(0)) {
                return false;
            }
            ++pa;
        }
        while (pb < eb) {
            if (rhs->valueData()[pb] != V(0)) {
                return false;
            }
            ++pb;
        }
    }
    return true;
}

template <Arithmetic V>
inline constexpr bool operator!=(const SparseMatrix<V>& A, const SparseMatrix<V>& B) {
    return !(A == B);
}

template <Arithmetic V>
inline constexpr Matrix<V> operator*(const SparseMatrix<V>& A, const Matrix<V>& B) {
    if (A.cols_size() != B.rows_size()) {
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication.");
    }

    SparseMatrix<V> buffer;
    const SparseMatrix<V>* matrix = &A;
    if (A.rowMutableMode()) {
        buffer = A.compressed();
        matrix = &buffer;
    }

    Matrix<V> result(matrix->rows_size(), B.cols_size());
    for (typename SparseMatrix<V>::size_type k = 0; k < matrix->cols_size(); ++k) {
        for (typename SparseMatrix<V>::InnerIterator it(*matrix, k); it; ++it) {
            for (typename Matrix<V>::size_type col = 0; col < B.cols_size(); ++col) {
                result(it.row(), col) += it.value() * B(k, col);
            }
        }
    }
    return result;
}

template <Arithmetic V>
inline constexpr Matrix<V> operator*(const Matrix<V>& A, const SparseMatrix<V>& B) {
    if (A.cols_size() != B.rows_size()) {
        throw std::invalid_argument("Matrices must have compatible dimensions for multiplication.");
    }

    SparseMatrix<V> buffer;
    const SparseMatrix<V>* matrix = &B;
    if (B.rowMutableMode()) {
        buffer = B.compressed();
        matrix = &buffer;
    }

    Matrix<V> result(A.rows_size(), matrix->cols_size());
    for (typename SparseMatrix<V>::size_type col = 0; col < matrix->cols_size(); ++col) {
        for (typename SparseMatrix<V>::InnerIterator it(*matrix, col); it; ++it) {
            for (typename Matrix<V>::size_type row = 0; row < A.rows_size(); ++row) {
                result(row, col) += A(row, it.row()) * it.value();
            }
        }
    }
    return result;
}

template <Arithmetic V>
inline std::ostream& operator<<(std::ostream& os, const SparseMatrix<V>& mat) {
    SparseMatrix<V> buffer;
    const SparseMatrix<V>* matrix = &mat;
    if (mat.rowMutableMode()) {
        buffer = mat.compressed();
        matrix = &buffer;
    }

    os << "SparseMatrix(" << matrix->rows_size() << "x" << matrix->cols_size()
       << ", nnz=" << matrix->nonZeros() << ")\n";
    for (typename SparseMatrix<V>::size_type col = 0; col < matrix->cols_size(); ++col) {
        for (typename SparseMatrix<V>::InnerIterator it(*matrix, col); it; ++it) {
            os << "  (" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
        }
    }
    return os;
}

#endif // !MINIMIZEROPTIMIZER_HEADERS_SPARSEMATRIX_H_
