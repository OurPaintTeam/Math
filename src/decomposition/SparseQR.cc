#include "SparseQR.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

size_t findRoot(size_t index, std::vector<size_t>& parent) {
    size_t p = parent[index];
    size_t gp = parent[p];
    while (gp != p) {
        parent[index] = gp;
        index = gp;
        p = parent[index];
        gp = parent[p];
    }
    return p;
}

void appendReachPath(
    size_t start,
    size_t column,
    const std::vector<size_t>& etree,
    std::vector<size_t>& marks,
    size_t stamp,
    std::vector<size_t>& reach)
{
    const size_t begin = reach.size();
    while (start < marks.size() && start < column && marks[start] != stamp) {
        reach.push_back(start);
        marks[start] = stamp;
        start = etree[start];
    }
    std::reverse(
        reach.begin() + static_cast<std::ptrdiff_t>(begin),
        reach.end());
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
      _workspace_outer(other._workspace_outer), _workspace_inner(other._workspace_inner),
      _workspace_val(other._workspace_val), _etree(other._etree), _first_row_elt(other._first_row_elt),
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
      _workspace_outer(std::move(other._workspace_outer)), _workspace_inner(std::move(other._workspace_inner)),
      _workspace_val(std::move(other._workspace_val)), _etree(std::move(other._etree)),
      _first_row_elt(std::move(other._first_row_elt)),
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
    std::swap(_workspace_outer, temp._workspace_outer);
    std::swap(_workspace_inner, temp._workspace_inner);
    std::swap(_workspace_val, temp._workspace_val);
    std::swap(_etree, temp._etree);
    std::swap(_first_row_elt, temp._first_row_elt);
    std::swap(_r_csc_outer, temp._r_csc_outer);
    std::swap(_r_csc_inner, temp._r_csc_inner);
    std::swap(_r_csc_val, temp._r_csc_val);
    std::swap(_r_diag, temp._r_diag);
    std::swap(_r_csr_outer_scratch, temp._r_csr_outer_scratch);
    std::swap(_r_csr_inner_scratch, temp._r_csr_inner_scratch);
    std::swap(_r_csr_values_scratch, temp._r_csr_values_scratch);
    std::swap(_r_csr_next_scratch, temp._r_csr_next_scratch);
    std::swap(_col_norms, temp._col_norms);
    std::swap(_col_map, temp._col_map);
    std::swap(_qt_workspace, temp._qt_workspace);
    std::swap(_enable_numeric_pivoting, temp._enable_numeric_pivoting);
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
    std::swap(_workspace_outer, temp._workspace_outer);
    std::swap(_workspace_inner, temp._workspace_inner);
    std::swap(_workspace_val, temp._workspace_val);
    std::swap(_etree, temp._etree);
    std::swap(_first_row_elt, temp._first_row_elt);
    std::swap(_r_csc_outer, temp._r_csc_outer);
    std::swap(_r_csc_inner, temp._r_csc_inner);
    std::swap(_r_csc_val, temp._r_csc_val);
    std::swap(_r_diag, temp._r_diag);
    std::swap(_r_csr_outer_scratch, temp._r_csr_outer_scratch);
    std::swap(_r_csr_inner_scratch, temp._r_csr_inner_scratch);
    std::swap(_r_csr_values_scratch, temp._r_csr_values_scratch);
    std::swap(_r_csr_next_scratch, temp._r_csr_next_scratch);
    std::swap(_col_norms, temp._col_norms);
    std::swap(_col_map, temp._col_map);
    std::swap(_qt_workspace, temp._qt_workspace);
    std::swap(_enable_numeric_pivoting, temp._enable_numeric_pivoting);
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

void SparseQR::buildPermutedWorkspace() {
    _workspace_outer.assign(_n + 1, 0);
    for (size_t row = 0; row < _m; ++row) {
        for (SparseMatrix<>::InnerIterator it(_A, row); it; ++it) {
            ++_workspace_outer[_perm_inv[it.col()] + 1];
        }
    }
    for (size_t col = 0; col < _n; ++col) {
        _workspace_outer[col + 1] += _workspace_outer[col];
    }

    _workspace_inner.assign(_workspace_outer.back(), 0);
    _workspace_val.assign(_workspace_outer.back(), 0.0);
    std::vector<size_t> next(_workspace_outer);
    for (size_t row = 0; row < _m; ++row) {
        for (SparseMatrix<>::InnerIterator it(_A, row); it; ++it) {
            const size_t col = _perm_inv[it.col()];
            const size_t pos = next[col]++;
            _workspace_inner[pos] = row;
            _workspace_val[pos] = it.value();
        }
    }
}

void SparseQR::buildColumnEliminationTree() {
    _first_row_elt.assign(_m, _n);
    for (size_t row = 0; row < _min_mn; ++row) {
        _first_row_elt[row] = row;
    }
    for (size_t col = 0; col < _n; ++col) {
        for (size_t p = _workspace_outer[col]; p < _workspace_outer[col + 1]; ++p) {
            const size_t row = _workspace_inner[p];
            _first_row_elt[row] = std::min(_first_row_elt[row], col);
        }
    }

    _etree.assign(_n, _n);
    std::vector<size_t> root(_n, 0);
    std::vector<size_t> parent(_n, 0);
    for (size_t col = 0; col < _n; ++col) {
        bool found_diag = col >= _m;
        parent[col] = col;
        size_t cset = col;
        root[cset] = col;
        _etree[col] = _n;

        size_t p = _workspace_outer[col];
        while (p < _workspace_outer[col + 1] || !found_diag) {
            size_t row = col;
            if (p < _workspace_outer[col + 1]) {
                row = _workspace_inner[p];
                ++p;
            } else {
                found_diag = true;
            }
            if (row == col) {
                found_diag = true;
            }

            const size_t first = _first_row_elt[row];
            if (first >= col) {
                continue;
            }

            const size_t rset = findRoot(first, parent);
            const size_t rroot = root[rset];
            if (rroot != col) {
                _etree[rroot] = col;
                parent[cset] = rset;
                cset = rset;
                root[cset] = col;
            }
        }
    }
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
    std::vector<size_t> ordered_active(_active_cols);
    std::sort(ordered_active.begin(), ordered_active.end(), [this](size_t a, size_t b) {
        if (_col_counts[a] != _col_counts[b]) {
            return _col_counts[a] < _col_counts[b];
        }
        return a < b;
    });
    for (size_t j : ordered_active) {
        _perm.push_back(j);
    }

    _perm_inv.resize(_n);
    for (size_t j = 0; j < _n; ++j) {
        _perm_inv[_perm[j]] = j;
    }

    buildPermutedWorkspace();
    buildColumnEliminationTree();
    _analyzed = true;
}

SparseMatrix<> SparseQR::buildSparseR(
    const std::vector<std::vector<size_t>>& column_rows,
    const std::vector<std::vector<double>>& column_values) const
{
    if (_min_mn == 0) {
        return SparseMatrix<>(0, _n);
    }

    std::vector<size_t> outer(_min_mn + 1, 0);
    for (size_t col = 0; col < _n; ++col) {
        for (size_t pos = 0; pos < column_rows[col].size(); ++pos) {
            const size_t row = column_rows[col][pos];
            if (row >= _min_mn || std::abs(column_values[col][pos]) == 0.0) {
                continue;
            }
            ++outer[row + 1];
        }
    }
    for (size_t row = 0; row < _min_mn; ++row) {
        outer[row + 1] += outer[row];
    }

    std::vector<size_t> inner(outer.back());
    std::vector<double> values(outer.back());
    std::vector<size_t> next(outer);
    for (size_t col = 0; col < _n; ++col) {
        for (size_t pos = 0; pos < column_rows[col].size(); ++pos) {
            const size_t row = column_rows[col][pos];
            const double value = column_values[col][pos];
            if (row >= _min_mn || value == 0.0) {
                continue;
            }
            const size_t out_pos = next[row]++;
            inner[out_pos] = col;
            values[out_pos] = value;
        }
    }

    return SparseMatrix<>::fromCSR(
        _min_mn,
        _n,
        std::move(values),
        std::move(inner),
        std::move(outer));
}

void SparseQR::buildSparseRFromCSC() {
    if (_min_mn == 0) {
        _R_sparse = SparseMatrix<>(0, _n);
        return;
    }

    _r_csr_outer_scratch.assign(_min_mn + 1, 0);
    for (size_t col = 0; col < _n; ++col) {
        for (size_t p = _r_csc_outer[col]; p < _r_csc_outer[col + 1]; ++p) {
            const size_t row = _r_csc_inner[p];
            if (row < _min_mn && _r_csc_val[p] != 0.0) {
                ++_r_csr_outer_scratch[row + 1];
            }
        }
    }
    for (size_t row = 0; row < _min_mn; ++row) {
        _r_csr_outer_scratch[row + 1] += _r_csr_outer_scratch[row];
    }

    const size_t nnz = _r_csr_outer_scratch.back();
    std::vector<size_t> inner(nnz);
    std::vector<double> values(nnz);
    _r_csr_next_scratch = _r_csr_outer_scratch;

    for (size_t col = 0; col < _n; ++col) {
        for (size_t p = _r_csc_outer[col]; p < _r_csc_outer[col + 1]; ++p) {
            const size_t row = _r_csc_inner[p];
            const double value = _r_csc_val[p];
            if (row >= _min_mn || value == 0.0) continue;
            const size_t out_pos = _r_csr_next_scratch[row]++;
            inner[out_pos] = col;
            values[out_pos] = value;
        }
    }

    _R_sparse = SparseMatrix<>::fromCSR(
        _min_mn,
        _n,
        std::move(values),
        std::move(inner),
        std::move(_r_csr_outer_scratch));
}

void SparseQR::factorizeNumeric() {
    if (!_analyzed) {
        throw std::runtime_error("Call analyze() before factorize().");
    }

    _ref_ptr.assign(_min_mn + 1, 0);
    _ref_idx.clear();
    _ref_val.clear();
    _ref_beta.assign(_min_mn, 0.0);
    _factorization_threshold = resolvePivotThreshold();
    _numerical_rank = 0;
    _leading_numerical_rank = 0;

    // Production-safe drop policy: tie filtering to factorization scale.
    const double drop_tolerance =
        _factorization_threshold * std::numeric_limits<double>::epsilon();

    _r_csc_outer.assign(_n + 1, 0);
    _r_csc_inner.clear();
    _r_csc_val.clear();
    _r_diag.assign(_min_mn, 0.0);

    const size_t nnz_est = _workspace_val.size() * 2 + _min_mn;
    _r_csc_inner.reserve(nnz_est);
    _r_csc_val.reserve(nnz_est);

    _factor_accumulator.assign(_m, 0.0);
    _factor_touched_marks.assign(_m, 0);
    _factor_marks.assign(_n, 0);
    _factor_touched_rows.clear();
    _factor_reach.clear();
    _factor_reflector_rows.clear();
    _factor_reflector_values.clear();

    _ref_idx.reserve(_workspace_val.size() + _min_mn);
    _ref_val.reserve(_workspace_val.size() + _min_mn);
    _factor_reach.reserve(_min_mn);
    _factor_touched_rows.reserve(_m);
    _factor_reflector_rows.reserve(_m);
    _factor_reflector_values.reserve(_m);

    bool leading_block_is_full_rank = true;
    size_t stamp = 1;

    for (size_t col = 0; col < _n; ++col, ++stamp) {
        _r_csc_outer[col] = _r_csc_inner.size();
        _factor_reach.clear();
        _factor_touched_rows.clear();
        const size_t reach_stop = (col < _min_mn) ? col : _min_mn;
        if (reach_stop < _factor_marks.size()) {
            _factor_marks[reach_stop] = stamp;
        }
        if (col < _min_mn) {
            _factor_touched_marks[col] = stamp;
            _factor_touched_rows.push_back(col);
            _factor_accumulator[col] = 0.0;
        }

        const size_t col_begin = _workspace_outer[col];
        const size_t col_end = _workspace_outer[col + 1];
        bool found_diag = col >= _m;

        size_t p = col_begin;
        while (p < col_end || !found_diag) {
            size_t row = col;
            double value = 0.0;
            if (p < col_end) {
                row = _workspace_inner[p];
                value = _workspace_val[p];
                ++p;
            } else {
                found_diag = true;
            }
            if (row == col) {
                found_diag = true;
            }
            if (_factor_touched_marks[row] != stamp) {
                _factor_touched_marks[row] = stamp;
                _factor_touched_rows.push_back(row);
                _factor_accumulator[row] = 0.0;
            }
            _factor_accumulator[row] = value;

            const size_t start = _first_row_elt[row];
            if (start < _min_mn && start < col) {
                appendReachPath(start, col, _etree, _factor_marks, stamp, _factor_reach);
            }
        }

        for (size_t ii = _factor_reach.size(); ii > 0; --ii) {
            const size_t prev = _factor_reach[ii - 1];
            const double beta = _ref_beta[prev];
            if (beta == 0.0) {
                continue;
            }

            const size_t ref_begin = _ref_ptr[prev];
            const size_t ref_end = _ref_ptr[prev + 1];
            double dot = 0.0;
            const size_t* ridx = _ref_idx.data() + ref_begin;
            const double* rval = _ref_val.data() + ref_begin;
            const size_t rlen = ref_end - ref_begin;
            for (size_t p = 0; p < rlen; ++p) {
                dot += rval[p] * _factor_accumulator[ridx[p]];
            }
            if (std::abs(dot) <= drop_tolerance) {
                continue;
            }

            const double scale = beta * dot;
            for (size_t p = 0; p < rlen; ++p) {
                const size_t row = ridx[p];
                if (_factor_touched_marks[row] != stamp) {
                    _factor_touched_marks[row] = stamp;
                    _factor_touched_rows.push_back(row);
                    _factor_accumulator[row] = 0.0;
                }
                _factor_accumulator[row] -= scale * rval[p];
            }
        }

        const size_t r_limit = (col < _min_mn) ? col : _min_mn;
        for (size_t row = 0; row < r_limit; ++row) {
            if (_factor_touched_marks[row] != stamp) continue;
            const double value = _factor_accumulator[row];
            if (std::abs(value) > drop_tolerance) {
                _r_csc_inner.push_back(row);
                _r_csc_val.push_back(value);
            }
        }

        if (col < _min_mn) {
            double leading_value = _factor_accumulator[col];
            double sqr_norm = 0.0;
            _factor_reflector_rows.clear();
            _factor_reflector_values.clear();
            _factor_reflector_rows.push_back(col);
            _factor_reflector_values.push_back(leading_value);
            for (size_t row : _factor_touched_rows) {
                if (row <= col) {
                    continue;
                }
                const double value = _factor_accumulator[row];
                if (std::abs(value) <= drop_tolerance) {
                    continue;
                }
                _factor_reflector_rows.push_back(row);
                _factor_reflector_values.push_back(value);
                sqr_norm += value * value;
            }
            const double norm = std::sqrt(leading_value * leading_value + sqr_norm);
            if (norm > 0.0) {
                const double alpha = (leading_value >= 0.0) ? -norm : norm;
                _factor_reflector_values[0] = leading_value - alpha;
                const double vtv = _factor_reflector_values[0] * _factor_reflector_values[0] + sqr_norm;
                const double beta = (vtv == 0.0) ? 0.0 : 2.0 / vtv;
                _ref_idx.insert(_ref_idx.end(), _factor_reflector_rows.begin(), _factor_reflector_rows.end());
                _ref_val.insert(_ref_val.end(), _factor_reflector_values.begin(), _factor_reflector_values.end());
                _ref_ptr[col + 1] = _ref_idx.size();
                _ref_beta[col] = beta;
                _r_diag[col] = alpha;
                if (std::abs(alpha) > drop_tolerance) {
                    _r_csc_inner.push_back(col);
                    _r_csc_val.push_back(alpha);
                }

                if (norm >= _factorization_threshold) {
                    ++_numerical_rank;
                    if (leading_block_is_full_rank) {
                        ++_leading_numerical_rank;
                    }
                } else {
                    leading_block_is_full_rank = false;
                }
            } else {
                _ref_ptr[col + 1] = _ref_ptr[col];
                leading_block_is_full_rank = false;
            }
        }

        for (size_t row : _factor_touched_rows) {
            _factor_accumulator[row] = 0.0;
        }
    }

    _r_csc_outer[_n] = _r_csc_inner.size();
    buildSparseRFromCSC();
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

void SparseQR::setNumericPivotingEnabled(bool enabled) {
    _enable_numeric_pivoting = enabled;
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

    const size_t rank = _numerical_rank;
    const size_t nrhs = b.cols_size();
    const size_t ld = nrhs;

    _qt_workspace.resize(_m * ld);
    for (size_t i = 0; i < _m; ++i) {
        for (size_t j = 0; j < nrhs; ++j) {
            _qt_workspace[i * ld + j] = b(i, j);
        }
    }
    applyQtDense(_qt_workspace.data(), nrhs, ld);

    std::vector<double> zbuf(rank * ld, 0.0);
    backsolveCSC(_qt_workspace.data(), zbuf.data(), rank, nrhs, ld, ld);

    Matrix<> x(_n, b.cols_size());
    for (size_t j = 0; j < _n; ++j) {
        const size_t orig = _perm[j];
        for (size_t rhs = 0; rhs < b.cols_size(); ++rhs) {
            if (j < _numerical_rank) {
                x(orig, rhs) = zbuf[j * ld + rhs];
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
    if (damping == 0.0 && _m >= _n && _numerical_rank == _n && _leading_numerical_rank == _numerical_rank) {
        Matrix<> identity(_m, _m);
        for (size_t i = 0; i < _m; ++i) {
            identity(i, i) = 1.0;
        }
        return solve(identity, 0.0);
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

const std::vector<size_t>& SparseQR::colsPermutation() const {
    return _perm;
}

Matrix<> SparseQR::applyQ(const Matrix<>& B) const {
    if (B.rows_size() != _A.rows_size()) {
        throw std::invalid_argument("applyQ: matrix rows must match A rows.");
    }
    Matrix<> result(B);
    for (size_t kk = _ref_beta.size(); kk > 0; --kk) {
        const size_t k = kk - 1;
        const double beta = _ref_beta[k];
        if (beta == 0.0) {
            continue;
        }
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
            if (dot == 0.0) {
                continue;
            }
            const double scale = beta * dot;
            for (size_t p = 0; p < rlen; ++p) {
                result(ridx[p], j) -= scale * rval[p];
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
        if (beta == 0.0) {
            continue;
        }
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
            if (dot == 0.0) {
                continue;
            }
            const double scale = beta * dot;
            for (size_t p = 0; p < rlen; ++p) {
                result(ridx[p], j) -= scale * rval[p];
            }
        }
    }
    return result;
}

void SparseQR::applyQtDense(double* data, size_t nrhs, size_t ld) const {
    for (size_t k = 0; k < _ref_beta.size(); ++k) {
        const double beta = _ref_beta[k];
        if (beta == 0.0) continue;
        const size_t rstart = _ref_ptr[k];
        const size_t rend = _ref_ptr[k + 1];
        const size_t* ridx = _ref_idx.data() + rstart;
        const double* rval = _ref_val.data() + rstart;
        const size_t rlen = rend - rstart;

        for (size_t j = 0; j < nrhs; ++j) {
            double dot = 0.0;
            for (size_t p = 0; p < rlen; ++p) {
                dot += rval[p] * data[ridx[p] * ld + j];
            }
            if (dot == 0.0) continue;
            const double scale = beta * dot;
            for (size_t p = 0; p < rlen; ++p) {
                data[ridx[p] * ld + j] -= scale * rval[p];
            }
        }
    }
}

void SparseQR::backsolveCSC(
    const double* rhs,
    double* x,
    size_t rank,
    size_t nrhs,
    size_t ld_rhs,
    size_t ld_x) const
{
    for (size_t j = 0; j < nrhs; ++j) {
        for (size_t i = 0; i < rank; ++i) {
            x[i * ld_x + j] = rhs[i * ld_rhs + j];
        }
    }

    for (size_t kk = rank; kk > 0; --kk) {
        const size_t k = kk - 1;
        const double diag = _r_diag[k];
        if (std::abs(diag) < _factorization_threshold) {
            throw std::runtime_error(
                "backsolveCSC: encountered a pivot below the numerical rank threshold.");
        }
        const double inv_diag = 1.0 / diag;
        for (size_t j = 0; j < nrhs; ++j) {
            x[k * ld_x + j] *= inv_diag;
        }
        for (size_t p = _r_csc_outer[k]; p < _r_csc_outer[k + 1]; ++p) {
            const size_t row = _r_csc_inner[p];
            if (row >= k || row >= rank) continue;
            const double val = _r_csc_val[p];
            for (size_t j = 0; j < nrhs; ++j) {
                x[row * ld_x + j] -= val * x[k * ld_x + j];
            }
        }
    }
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
    for (size_t rhs_col = 0; rhs_col < rhs.cols_size(); ++rhs_col) {
        for (size_t ii = effective_rank; ii > 0; --ii) {
            const size_t i = ii - 1;
            double sum = rhs(i, rhs_col);
            double diagonal = 0.0;
            for (SparseMatrix<>::InnerIterator it(_R_sparse, i); it; ++it) {
                if (it.col() == i) {
                    diagonal = it.value();
                } else if (it.col() > i && it.col() < effective_rank) {
                    sum -= it.value() * x(it.col(), rhs_col);
                }
            }
            if (std::abs(diagonal) < _factorization_threshold) {
                throw std::runtime_error(
                    "solveUpperTriangular: encountered a pivot below the numerical rank threshold.");
            }
            x(i, rhs_col) = sum / diagonal;
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
