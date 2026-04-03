#include "SparseQR.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

double sumSquared(const std::vector<double>& values) {
    double sum = 0.0;
    for (double value : values) {
        sum += value * value;
    }
    return sum;
}

double sparseDot(
    const std::vector<size_t>& lhs_idx,
    const std::vector<double>& lhs_val,
    const std::vector<size_t>& rhs_idx,
    const std::vector<double>& rhs_val)
{
    double dot = 0.0;
    size_t lhs_pos = 0;
    size_t rhs_pos = 0;
    while (lhs_pos < lhs_idx.size() && rhs_pos < rhs_idx.size()) {
        if (lhs_idx[lhs_pos] < rhs_idx[rhs_pos]) {
            ++lhs_pos;
        } else if (lhs_idx[lhs_pos] > rhs_idx[rhs_pos]) {
            ++rhs_pos;
        } else {
            dot += lhs_val[lhs_pos] * rhs_val[rhs_pos];
            ++lhs_pos;
            ++rhs_pos;
        }
    }
    return dot;
}

void buildUpdatedRow(
    const std::vector<size_t>& row_idx,
    const std::vector<double>& row_val,
    const std::vector<size_t>& reflector_idx,
    const std::vector<double>& reflector_val,
    double scale,
    double drop_tolerance,
    std::vector<size_t>& out_idx,
    std::vector<double>& out_val)
{
    out_idx.clear();
    out_val.clear();
    out_idx.reserve(row_idx.size() + reflector_idx.size());
    out_val.reserve(row_val.size() + reflector_val.size());

    size_t row_pos = 0;
    size_t ref_pos = 0;
    while (row_pos < row_idx.size() || ref_pos < reflector_idx.size()) {
        size_t idx = 0;
        double value = 0.0;
        if (ref_pos == reflector_idx.size()
            || (row_pos < row_idx.size() && row_idx[row_pos] < reflector_idx[ref_pos])) {
            idx = row_idx[row_pos];
            value = row_val[row_pos];
            ++row_pos;
        } else if (row_pos == row_idx.size() || reflector_idx[ref_pos] < row_idx[row_pos]) {
            idx = reflector_idx[ref_pos];
            value = -scale * reflector_val[ref_pos];
            ++ref_pos;
        } else {
            idx = row_idx[row_pos];
            value = row_val[row_pos] - scale * reflector_val[ref_pos];
            ++row_pos;
            ++ref_pos;
        }

        if (std::abs(value) > drop_tolerance) {
            out_idx.push_back(idx);
            out_val.push_back(value);
        }
    }
}

void buildDiagonalizedRow(
    const std::vector<size_t>& row_idx,
    const std::vector<double>& row_val,
    size_t diagonal,
    double diagonal_value,
    double drop_tolerance,
    std::vector<size_t>& out_idx,
    std::vector<double>& out_val)
{
    out_idx.clear();
    out_val.clear();
    out_idx.reserve(row_idx.size());
    out_val.reserve(row_val.size());

    for (size_t pos = 0; pos < row_idx.size(); ++pos) {
        if (row_idx[pos] >= diagonal) {
            break;
        }
        if (std::abs(row_val[pos]) > drop_tolerance) {
            out_idx.push_back(row_idx[pos]);
            out_val.push_back(row_val[pos]);
        }
    }

    if (std::abs(diagonal_value) > drop_tolerance) {
        out_idx.push_back(diagonal);
        out_val.push_back(diagonal_value);
    }
}

SparseMatrix<> buildPermutedWorkspace(
    const SparseMatrix<>& input,
    const std::vector<size_t>& perm_inv)
{
    std::vector<Triplet<double>> triplets;
    triplets.reserve(input.nonZeros());
    for (size_t row = 0; row < input.rows_size(); ++row) {
        for (SparseMatrix<>::InnerIterator it(input, row); it; ++it) {
            triplets.emplace_back(perm_inv[it.col()], row, it.value());
        }
    }

    SparseMatrix<> workspace(input.cols_size(), input.rows_size());
    workspace.setFromTriplets(triplets.begin(), triplets.end());
    return workspace;
}

SparseMatrix<> buildSparseR(const SparseMatrix<>& workspace, size_t min_mn) {
    SparseMatrix<> result(min_mn, workspace.rows_size());
    if (min_mn == 0) {
        return result;
    }

    std::vector<Triplet<double>> triplets;
    triplets.reserve(workspace.nonZeros());
    for (size_t col = 0; col < workspace.rows_size(); ++col) {
        for (SparseMatrix<>::InnerIterator it(workspace, col); it; ++it) {
            const size_t row = it.col();
            if (row >= min_mn || row > col || it.value() == 0.0) {
                continue;
            }
            triplets.emplace_back(row, col, it.value());
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

} // namespace

SparseQR::SparseQR(const SparseMatrix<>& A) {
    if (A.rows_size() < 1 || A.cols_size() < 1) {
        throw std::runtime_error("Matrix should be: rows > 0 && cols > 0");
    }
    _A = A;
}

SparseQR::SparseQR(const SparseQR& other)
    : _A(other._A), _R_sparse(other._R_sparse), _m(other._m), _n(other._n), _min_mn(other._min_mn),
      _pivot_threshold(other._pivot_threshold), _factorization_threshold(other._factorization_threshold),
      _numerical_rank(other._numerical_rank), _leading_numerical_rank(other._leading_numerical_rank),
      _ref_ptr(other._ref_ptr), _ref_idx(other._ref_idx), _ref_val(other._ref_val),
      _ref_beta(other._ref_beta), _col_counts(other._col_counts), _zero_cols(other._zero_cols),
      _active_cols(other._active_cols), _perm(other._perm), _perm_inv(other._perm_inv),
      _use_default_threshold(other._use_default_threshold), _preprocessed(other._preprocessed),
      _analyzed(other._analyzed) {}

SparseQR::SparseQR(SparseQR&& other) noexcept
    : _A(std::move(other._A)), _R_sparse(std::move(other._R_sparse)), _m(other._m), _n(other._n),
      _min_mn(other._min_mn), _pivot_threshold(other._pivot_threshold),
      _factorization_threshold(other._factorization_threshold), _numerical_rank(other._numerical_rank),
      _leading_numerical_rank(other._leading_numerical_rank), _ref_ptr(std::move(other._ref_ptr)),
      _ref_idx(std::move(other._ref_idx)), _ref_val(std::move(other._ref_val)),
      _ref_beta(std::move(other._ref_beta)), _col_counts(std::move(other._col_counts)),
      _zero_cols(std::move(other._zero_cols)), _active_cols(std::move(other._active_cols)),
      _perm(std::move(other._perm)), _perm_inv(std::move(other._perm_inv)),
      _use_default_threshold(other._use_default_threshold), _preprocessed(other._preprocessed),
      _analyzed(other._analyzed) {}

SparseQR& SparseQR::operator=(const SparseQR& other) {
    SparseQR temp(other);
    std::swap(_A, temp._A);
    std::swap(_R_sparse, temp._R_sparse);
    std::swap(_m, temp._m);
    std::swap(_n, temp._n);
    std::swap(_min_mn, temp._min_mn);
    std::swap(_pivot_threshold, temp._pivot_threshold);
    std::swap(_factorization_threshold, temp._factorization_threshold);
    std::swap(_numerical_rank, temp._numerical_rank);
    std::swap(_leading_numerical_rank, temp._leading_numerical_rank);
    std::swap(_ref_ptr, temp._ref_ptr);
    std::swap(_ref_idx, temp._ref_idx);
    std::swap(_ref_val, temp._ref_val);
    std::swap(_ref_beta, temp._ref_beta);
    std::swap(_col_counts, temp._col_counts);
    std::swap(_zero_cols, temp._zero_cols);
    std::swap(_active_cols, temp._active_cols);
    std::swap(_perm, temp._perm);
    std::swap(_perm_inv, temp._perm_inv);
    std::swap(_use_default_threshold, temp._use_default_threshold);
    std::swap(_preprocessed, temp._preprocessed);
    std::swap(_analyzed, temp._analyzed);
    return *this;
}

SparseQR& SparseQR::operator=(SparseQR&& other) noexcept {
    SparseQR temp(std::move(other));
    std::swap(_A, temp._A);
    std::swap(_R_sparse, temp._R_sparse);
    std::swap(_m, temp._m);
    std::swap(_n, temp._n);
    std::swap(_min_mn, temp._min_mn);
    std::swap(_pivot_threshold, temp._pivot_threshold);
    std::swap(_factorization_threshold, temp._factorization_threshold);
    std::swap(_numerical_rank, temp._numerical_rank);
    std::swap(_leading_numerical_rank, temp._leading_numerical_rank);
    std::swap(_ref_ptr, temp._ref_ptr);
    std::swap(_ref_idx, temp._ref_idx);
    std::swap(_ref_val, temp._ref_val);
    std::swap(_ref_beta, temp._ref_beta);
    std::swap(_col_counts, temp._col_counts);
    std::swap(_zero_cols, temp._zero_cols);
    std::swap(_active_cols, temp._active_cols);
    std::swap(_perm, temp._perm);
    std::swap(_perm_inv, temp._perm_inv);
    std::swap(_use_default_threshold, temp._use_default_threshold);
    std::swap(_preprocessed, temp._preprocessed);
    std::swap(_analyzed, temp._analyzed);
    return *this;
}

bool operator==(const SparseQR& A, const SparseQR& B) {
    return A.A() == B.A() && A.Q() == B.Q() && A.R() == B.R();
}

bool operator!=(const SparseQR& A, const SparseQR& B) {
    return !(A == B);
}

double SparseQR::computeDefaultThreshold() const {
    const double eps = std::numeric_limits<double>::epsilon();
    std::vector<double> col_norm_sq(_n, 0.0);
    for (size_t i = 0; i < _m; ++i) {
        for (SparseMatrix<>::InnerIterator it(_A, i); it; ++it) {
            col_norm_sq[it.col()] += it.value() * it.value();
        }
    }

    double max_col_norm = 0.0;
    for (double sq_norm : col_norm_sq) {
        max_col_norm = std::max(max_col_norm, std::sqrt(sq_norm));
    }
    if (max_col_norm == 0.0) {
        max_col_norm = 1.0;
    }

    return 20.0 * static_cast<double>(_m + _n) * max_col_norm * eps;
}

double SparseQR::resolvePivotThreshold() const {
    return _use_default_threshold ? computeDefaultThreshold() : _pivot_threshold;
}

void SparseQR::preprocessProblem() {
    if (_preprocessed) {
        return;
    }
    _m = _A.rows_size();
    _n = _A.cols_size();
    _min_mn = std::min(_m, _n);
    _col_counts.assign(_n, 0);
    _zero_cols.clear();
    _active_cols.clear();
    for (size_t i = 0; i < _m; ++i) {
        for (SparseMatrix<>::InnerIterator it(_A, i); it; ++it) {
            ++_col_counts[it.col()];
        }
    }
    for (size_t j = 0; j < _n; ++j) {
        if (_col_counts[j] == 0) {
            _zero_cols.push_back(j);
        } else {
            _active_cols.push_back(j);
        }
    }
    _preprocessed = true;
}

void SparseQR::analyzeStructure() {
    preprocessProblem();
    if (_analyzed) {
        return;
    }
    _perm.clear();
    _perm.reserve(_n);
    for (size_t j : _zero_cols) {
        _perm.push_back(j);
    }
    std::vector<size_t> orderedActive(_active_cols);
    std::sort(orderedActive.begin(), orderedActive.end(), [this](size_t a, size_t b) {
        if (_col_counts[a] != _col_counts[b]) {
            return _col_counts[a] < _col_counts[b];
        }
        return a < b;
    });
    for (size_t j : orderedActive) {
        _perm.push_back(j);
    }
    _perm_inv.resize(_n);
    for (size_t j = 0; j < _n; ++j) {
        _perm_inv[_perm[j]] = j;
    }
    _analyzed = true;
}

SparseMatrix<> SparseQR::buildPermutedWorkspace() const {
    return ::buildPermutedWorkspace(_A, _perm_inv);
}

SparseMatrix<> SparseQR::buildSparseR(const SparseMatrix<>& workspace) const {
    return ::buildSparseR(workspace, _min_mn);
}

void SparseQR::updateColumnRowIndex(
    std::vector<std::vector<size_t>>& column_rows,
    size_t row,
    const std::vector<size_t>& old_idx,
    const std::vector<size_t>& new_idx) const
{
    size_t old_pos = 0;
    size_t new_pos = 0;
    while (old_pos < old_idx.size() || new_pos < new_idx.size()) {
        if (new_pos == new_idx.size()
            || (old_pos < old_idx.size() && old_idx[old_pos] < new_idx[new_pos])) {
            auto& rows = column_rows[old_idx[old_pos]];
            auto it = std::lower_bound(rows.begin(), rows.end(), row);
            if (it != rows.end() && *it == row) {
                rows.erase(it);
            }
            ++old_pos;
        } else if (old_pos == old_idx.size() || new_idx[new_pos] < old_idx[old_pos]) {
            auto& rows = column_rows[new_idx[new_pos]];
            auto it = std::lower_bound(rows.begin(), rows.end(), row);
            if (it == rows.end() || *it != row) {
                rows.insert(it, row);
            }
            ++new_pos;
        } else {
            ++old_pos;
            ++new_pos;
        }
    }
}

void SparseQR::collectAffectedRows(
    const std::vector<std::vector<size_t>>& column_rows,
    const std::vector<size_t>& support,
    size_t min_row,
    std::vector<size_t>& marks,
    size_t stamp,
    std::vector<size_t>& affected_rows) const
{
    affected_rows.clear();
    for (size_t col : support) {
        for (size_t row : column_rows[col]) {
            if (row <= min_row || marks[row] == stamp) {
                continue;
            }
            marks[row] = stamp;
            affected_rows.push_back(row);
        }
    }
    std::sort(affected_rows.begin(), affected_rows.end());
}

void SparseQR::factorizeNumeric() {
    if (!_analyzed) {
        throw std::runtime_error("Call analyze() before factorizeNumeric().");
    }

    SparseMatrix<> workspace = buildPermutedWorkspace();
    std::vector<std::vector<size_t>> column_rows(workspace.cols_size());
    for (size_t row = 0; row < workspace.rows_size(); ++row) {
        for (SparseMatrix<>::InnerIterator it(workspace, row); it; ++it) {
            column_rows[it.col()].push_back(row);
        }
    }
    _ref_ptr.assign(_min_mn + 1, 0);
    _ref_idx.clear();
    _ref_val.clear();
    _ref_beta.assign(_min_mn, 0.0);
    _factorization_threshold = resolvePivotThreshold();
    _numerical_rank = 0;
    _leading_numerical_rank = 0;

    const double drop_tolerance = 64.0 * std::numeric_limits<double>::epsilon();

    _ref_idx.reserve(workspace.nonZeros() + _min_mn);
    _ref_val.reserve(workspace.nonZeros() + _min_mn);

    std::vector<size_t> row_idx;
    std::vector<double> row_val;
    std::vector<size_t> reflector_idx;
    std::vector<double> reflector_val;
    std::vector<size_t> updated_idx;
    std::vector<double> updated_val;
    std::vector<size_t> affected_rows;
    std::vector<size_t> row_marks(_n, 0);
    size_t mark_stamp = 1;

    bool leading_block_is_full_rank = true;
    for (size_t k = 0; k < _min_mn; ++k) {
        workspace.getRowEntries(k, row_idx, row_val);
        auto tail_begin = std::lower_bound(row_idx.begin(), row_idx.end(), k);
        const size_t tail_offset = static_cast<size_t>(tail_begin - row_idx.begin());

        reflector_idx.assign(row_idx.begin() + static_cast<std::ptrdiff_t>(tail_offset), row_idx.end());
        reflector_val.assign(row_val.begin() + static_cast<std::ptrdiff_t>(tail_offset), row_val.end());

        const double sigma = sumSquared(reflector_val);
        const double norm = std::sqrt(sigma);
        const bool strong_pivot = norm > 0.0 && norm >= _factorization_threshold;
        if (!strong_pivot) {
            _ref_ptr[k + 1] = _ref_ptr[k];
            leading_block_is_full_rank = false;
            continue;
        }

        double old_lead = 0.0;
        if (!reflector_idx.empty() && reflector_idx.front() == k) {
            old_lead = reflector_val.front();
        }

        const double alpha = (old_lead >= 0.0) ? -norm : norm;
        if (!reflector_idx.empty() && reflector_idx.front() == k) {
            reflector_val.front() -= alpha;
        } else {
            reflector_idx.insert(reflector_idx.begin(), k);
            reflector_val.insert(reflector_val.begin(), -alpha);
        }

        const double vtv = sumSquared(reflector_val);
        _ref_beta[k] = 2.0 / vtv;

        _ref_idx.insert(_ref_idx.end(), reflector_idx.begin(), reflector_idx.end());
        _ref_val.insert(_ref_val.end(), reflector_val.begin(), reflector_val.end());
        _ref_ptr[k + 1] = _ref_idx.size();

        buildDiagonalizedRow(row_idx, row_val, k, alpha, drop_tolerance, updated_idx, updated_val);
        workspace.replaceRow(k, updated_idx, updated_val);
        updateColumnRowIndex(column_rows, k, row_idx, updated_idx);

        ++_numerical_rank;
        if (leading_block_is_full_rank) {
            ++_leading_numerical_rank;
        }

        collectAffectedRows(column_rows, reflector_idx, k, row_marks, mark_stamp, affected_rows);
        ++mark_stamp;

        for (size_t row = 0; row < affected_rows.size(); ++row) {
            const size_t j = affected_rows[row];
            workspace.getRowEntries(j, row_idx, row_val);

            const double dot = sparseDot(row_idx, row_val, reflector_idx, reflector_val);
            if (std::abs(dot) <= drop_tolerance) {
                continue;
            }

            const double scale = _ref_beta[k] * dot;
            buildUpdatedRow(
                row_idx, row_val,
                reflector_idx, reflector_val,
                scale, drop_tolerance,
                updated_idx, updated_val);
            workspace.replaceRow(j, updated_idx, updated_val);
            updateColumnRowIndex(column_rows, j, row_idx, updated_idx);
        }
    }

    _R_sparse = buildSparseR(workspace);
}

void SparseQR::analyze() {
    preprocessProblem();
    analyzeStructure();
}

void SparseQR::factorize() {
    if (!_analyzed) {
        throw std::runtime_error("Call analyze() before factorize().");
    }
    factorizeNumeric();
}

void SparseQR::qr() {
    preprocessProblem();
    analyzeStructure();
    factorizeNumeric();
}

void SparseQR::setPivotThreshold(double threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0) {
        throw std::invalid_argument("pivot threshold must be finite and non-negative.");
    }
    _use_default_threshold = false;
    _pivot_threshold = threshold;
}

Matrix<> SparseQR::solve(const Matrix<>& b, double damping) const {
    if (_R_sparse.rows_size() == 0 || _R_sparse.cols_size() == 0) {
        throw std::runtime_error("Call qr() before solve().");
    }
    if (damping < 0.0) {
        throw std::invalid_argument("damping must be non-negative.");
    }
    if (b.rows_size() != _A.rows_size()) {
        throw std::invalid_argument("Right-hand side rows must match A rows.");
    }
    if (_m < _n) {
        throw std::runtime_error(
            "solve() currently supports only overdetermined or square systems (m >= n); "
            "use pseudoInverse() explicitly for underdetermined problems.");
    }
    if (_numerical_rank != _n || _leading_numerical_rank != _numerical_rank) {
        throw std::runtime_error(
            "solve() currently supports only full column rank QR factors; "
            "use pseudoInverse() explicitly for rank-deficient problems.");
    }

    (void)damping;

    Matrix<> y = applyQt(b);
    Matrix<> yHead(_numerical_rank, b.cols_size());
    for (size_t i = 0; i < _numerical_rank; ++i) {
        for (size_t rhs = 0; rhs < b.cols_size(); ++rhs) {
            yHead(i, rhs) = y(i, rhs);
        }
    }

    Matrix<> z = solveUpperTriangular(yHead, _numerical_rank);
    Matrix<> x(_n, b.cols_size());
    for (size_t j = 0; j < _n; ++j) {
        const size_t orig = _perm[j];
        for (size_t rhs = 0; rhs < b.cols_size(); ++rhs) {
            if (j < _numerical_rank) {
                x(orig, rhs) = z(j, rhs);
            } else {
                x(orig, rhs) = 0.0;
            }
        }
    }
    return x;
}

Matrix<> SparseQR::pseudoInverse(double damping) const {
    if (_R_sparse.rows_size() == 0 || _R_sparse.cols_size() == 0) {
        throw std::runtime_error("Call qr() before pseudoInverse().");
    }
    if (damping < 0.0) {
        throw std::invalid_argument("damping must be non-negative.");
    }
    Matrix<> denseR = R();
    Matrix<> q = Q();
    if (denseR.rows_size() != q.cols_size()) {
        throw std::runtime_error("pseudoInverse: inconsistent Q and R dimensions.");
    }
    Matrix<> Rt = denseR.transpose();
    Matrix<> Qt = q.transpose();
    if (Rt.cols_size() != Qt.rows_size()) {
        throw std::runtime_error("pseudoInverse: internal multiplication dimensions are inconsistent.");
    }
    Matrix<> RtR = Rt * denseR;
    Matrix<> reg = Matrix<>::identity(RtR.rows_size(), RtR.cols_size()) * damping;
    Matrix<> inv = (RtR + reg).inverse();
    Matrix<> pinvPerm = inv * Rt * Qt;
    Matrix<> pinv(_A.cols_size(), pinvPerm.cols_size());
    for (size_t j = 0; j < _A.cols_size(); ++j) {
        const size_t orig = _perm[j];
        for (size_t col = 0; col < pinvPerm.cols_size(); ++col) {
            pinv(orig, col) = pinvPerm(j, col);
        }
    }
    return pinv;
}

size_t SparseQR::rank(double tol) const {
    if (_R_sparse.rows_size() == 0 || _R_sparse.cols_size() == 0) {
        throw std::runtime_error("Call qr() before rank().");
    }
    if (tol < 0.0) {
        return _numerical_rank;
    }

    const size_t min_dim = std::min(_R_sparse.rows_size(), _R_sparse.cols_size());
    size_t r = 0;
    for (size_t i = 0; i < min_dim; ++i) {
        if (std::abs(_R_sparse(i, i)) >= tol) {
            ++r;
        }
    }
    return r;
}

SparseMatrix<> SparseQR::A() const {
    return _A;
}

Matrix<> SparseQR::applyQ(const Matrix<>& B) const {
    if (B.rows_size() != _A.rows_size()) {
        throw std::invalid_argument("applyQ: matrix rows must match A rows.");
    }
    Matrix<> result(B);
    for (size_t kk = _ref_beta.size(); kk > 0; --kk) {
        const size_t k = kk - 1;
        const double beta = _ref_beta[k];
        if (beta == 0.0) continue;
        const size_t rstart = _ref_ptr[k];
        const size_t rend = _ref_ptr[k + 1];
        const size_t* ridx = _ref_idx.data() + rstart;
        const double* rval = _ref_val.data() + rstart;
        const size_t rlen = rend - rstart;
        for (size_t j = 0; j < result.cols_size(); ++j) {
            double dot = 0.0;
            for (size_t p = 0; p < rlen; ++p) {
                dot += rval[p] * result(ridx[p], j);
            }
            if (dot == 0.0) continue;
            const double s = beta * dot;
            for (size_t p = 0; p < rlen; ++p) {
                result(ridx[p], j) -= s * rval[p];
            }
        }
    }
    return result;
}

Matrix<> SparseQR::applyQt(const Matrix<>& B) const {
    if (B.rows_size() != _A.rows_size()) {
        throw std::invalid_argument("applyQt: matrix rows must match A rows.");
    }
    Matrix<> result(B);
    for (size_t k = 0; k < _ref_beta.size(); ++k) {
        const double beta = _ref_beta[k];
        if (beta == 0.0) continue;
        const size_t rstart = _ref_ptr[k];
        const size_t rend = _ref_ptr[k + 1];
        const size_t* ridx = _ref_idx.data() + rstart;
        const double* rval = _ref_val.data() + rstart;
        const size_t rlen = rend - rstart;
        for (size_t j = 0; j < result.cols_size(); ++j) {
            double dot = 0.0;
            for (size_t p = 0; p < rlen; ++p) {
                dot += rval[p] * result(ridx[p], j);
            }
            if (dot == 0.0) continue;
            const double s = beta * dot;
            for (size_t p = 0; p < rlen; ++p) {
                result(ridx[p], j) -= s * rval[p];
            }
        }
    }
    return result;
}

Matrix<> SparseQR::solveUpperTriangular(const Matrix<>& rhs, size_t effective_rank) const {
    if (_R_sparse.rows_size() == 0 || _R_sparse.cols_size() == 0) {
        throw std::runtime_error("Call qr() before solveUpperTriangular().");
    }
    if (effective_rank > std::min(_R_sparse.rows_size(), _R_sparse.cols_size())) {
        throw std::invalid_argument("solveUpperTriangular: effective rank exceeds the triangular factor size.");
    }
    if (rhs.rows_size() != effective_rank) {
        throw std::invalid_argument("solveUpperTriangular: right-hand side rows must match the effective rank.");
    }

    Matrix<> x(effective_rank, rhs.cols_size());
    for (size_t rhsCol = 0; rhsCol < rhs.cols_size(); ++rhsCol) {
        for (size_t ii = effective_rank; ii > 0; --ii) {
            const size_t i = ii - 1;
            double sum = rhs(i, rhsCol);
            double diagonal = 0.0;
            for (SparseMatrix<>::InnerIterator it(_R_sparse, i); it; ++it) {
                if (it.col() == i) {
                    diagonal = it.value();
                } else if (it.col() > i && it.col() < effective_rank) {
                    sum -= it.value() * x(it.col(), rhsCol);
                }
            }
            if (std::abs(diagonal) < _factorization_threshold) {
                throw std::runtime_error(
                    "solveUpperTriangular: encountered a pivot below the numerical rank threshold.");
            }
            x(i, rhsCol) = sum / diagonal;
        }
    }
    return x;
}

Matrix<> SparseQR::Q() const {
    if (_R_sparse.rows_size() == 0 || _R_sparse.cols_size() == 0) {
        return Matrix<>();
    }
    const size_t m = _A.rows_size();
    const size_t k = std::min(_A.rows_size(), _A.cols_size());
    Matrix<> identity(m, k);
    for (size_t i = 0; i < k; ++i) {
        identity(i, i) = 1.0;
    }
    return applyQ(identity);
}

Matrix<> SparseQR::R() const {
    return _R_sparse.toDense();
}
