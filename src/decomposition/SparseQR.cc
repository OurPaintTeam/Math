#include "SparseQR.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

SparseQR::SparseQR(const SparseMatrix<>& A) {
    if (A.rows_size() < 1 || A.cols_size() < 1) {
        throw std::runtime_error("Matrix should be: rows > 0 && cols > 0");
    }
    _A = A;
}

SparseQR::SparseQR(const SparseQR& other)
    : _A(other._A), _R(other._R), _m(other._m), _n(other._n), _min_mn(other._min_mn),
      _pivot_threshold(other._pivot_threshold), _factorization_threshold(other._factorization_threshold),
      _numerical_rank(other._numerical_rank), _leading_numerical_rank(other._leading_numerical_rank),
      _ref_ptr(other._ref_ptr), _ref_idx(other._ref_idx), _ref_val(other._ref_val),
      _ref_beta(other._ref_beta), _col_counts(other._col_counts), _zero_cols(other._zero_cols),
      _active_cols(other._active_cols), _perm(other._perm), _perm_inv(other._perm_inv),
      _use_default_threshold(other._use_default_threshold), _preprocessed(other._preprocessed),
      _analyzed(other._analyzed) {}

SparseQR::SparseQR(SparseQR&& other) noexcept
    : _A(std::move(other._A)), _R(std::move(other._R)), _m(other._m), _n(other._n), _min_mn(other._min_mn),
      _pivot_threshold(other._pivot_threshold), _factorization_threshold(other._factorization_threshold),
      _numerical_rank(other._numerical_rank), _leading_numerical_rank(other._leading_numerical_rank),
      _ref_ptr(std::move(other._ref_ptr)), _ref_idx(std::move(other._ref_idx)),
      _ref_val(std::move(other._ref_val)), _ref_beta(std::move(other._ref_beta)),
      _col_counts(std::move(other._col_counts)), _zero_cols(std::move(other._zero_cols)),
      _active_cols(std::move(other._active_cols)), _perm(std::move(other._perm)), _perm_inv(std::move(other._perm_inv)),
      _use_default_threshold(other._use_default_threshold), _preprocessed(other._preprocessed),
      _analyzed(other._analyzed) {}

SparseQR& SparseQR::operator=(const SparseQR& other) {
    SparseQR temp(other);
    std::swap(_A, temp._A);
    std::swap(_R, temp._R);
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
    std::swap(_R, temp._R);
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

void SparseQR::factorizeNumeric() {
    if (!_analyzed) {
        throw std::runtime_error("Call analyze() before factorizeNumeric().");
    }
    std::vector<double> data(_m * _n, 0.0);
    for (size_t i = 0; i < _m; ++i) {
        for (SparseMatrix<>::InnerIterator it(_A, i); it; ++it) {
            data[_perm_inv[it.col()] * _m + i] = it.value();
        }
    }
    _ref_ptr.assign(_min_mn + 1, 0);
    _ref_idx.clear();
    _ref_val.clear();
    _ref_beta.assign(_min_mn, 0.0);
    _factorization_threshold = resolvePivotThreshold();
    _numerical_rank = 0;
    _leading_numerical_rank = 0;

    _ref_idx.reserve(_m * 2);
    _ref_val.reserve(_m * 2);
    bool leading_block_is_full_rank = true;
    for (size_t k = 0; k < _min_mn; ++k) {
        double* __restrict__ ck = data.data() + k * _m;

        double sigma = 0.0;
        for (size_t i = k; i < _m; ++i) sigma += ck[i] * ck[i];

        const double norm = std::sqrt(sigma);
        const bool strong_pivot = norm > 0.0 && norm >= _factorization_threshold;
        if (!strong_pivot) {
            _ref_ptr[k + 1] = _ref_ptr[k];
            leading_block_is_full_rank = false;
            continue;
        }

        double alpha = (ck[k] >= 0.0) ? -norm : norm;
        double old_lead = ck[k];
        ck[k] -= alpha;

        double vtv = sigma + ck[k] * ck[k] - old_lead * old_lead;
        double beta = 2.0 / vtv;
        _ref_beta[k] = beta;

        const size_t rstart = _ref_idx.size();
        for (size_t i = k; i < _m; ++i) {
            if (ck[i] != 0.0) {
                _ref_idx.push_back(i);
                _ref_val.push_back(ck[i]);
            }
        }
        _ref_ptr[k + 1] = _ref_idx.size();
        const size_t rend = _ref_ptr[k + 1];
        const size_t rlen = rend - rstart;

        const size_t* __restrict__ ridx = _ref_idx.data() + rstart;
        const double* __restrict__ rval = _ref_val.data() + rstart;

        ck[k] = alpha;
        std::memset(ck + k + 1, 0, (_m - k - 1) * sizeof(double));
        ++_numerical_rank;
        if (leading_block_is_full_rank) {
            ++_leading_numerical_rank;
        }

        for (size_t j = k + 1; j < _n; ++j) {
            double* __restrict__ cj = data.data() + j * _m;

            double dot = 0.0;
            for (size_t p = 0; p < rlen; ++p) {
                dot += rval[p] * cj[ridx[p]];
            }

            if (dot == 0.0) continue;

            const double s = beta * dot;
            for (size_t p = 0; p < rlen; ++p) {
                cj[ridx[p]] -= s * rval[p];
            }
        }
    }

    _R = Matrix<>(_min_mn, _n);
    for (size_t j = 0; j < _n; ++j) {
        const double* col = data.data() + j * _m;
        const size_t lim = std::min(j + 1, _min_mn);
        for (size_t i = 0; i < lim; ++i) {
            _R(i, j) = col[i];
        }
    }
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
    if (_R.rows_size() == 0 || _R.cols_size() == 0) {
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

    (void)damping;  // QR solve uses the factors directly; damping remains a pseudoInverse() option.

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
    if (_R.rows_size() == 0 || _R.cols_size() == 0) {
        throw std::runtime_error("Call qr() before pseudoInverse().");
    }
    if (damping < 0.0) {
        throw std::invalid_argument("damping must be non-negative.");
    }
    Matrix<> Rt = _R.transpose();
    Matrix<> RtR = Rt * _R;
    Matrix<> reg = Matrix<>::identity(RtR.rows_size(), RtR.cols_size()) * damping;
    Matrix<> inv = (RtR + reg).inverse();
    Matrix<> pinvPerm = inv * Rt * Q().transpose();
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
    if (_R.rows_size() == 0 || _R.cols_size() == 0) {
        throw std::runtime_error("Call qr() before rank().");
    }
    if (tol < 0.0) {
        return _numerical_rank;
    }

    const size_t min_dim = std::min(_R.rows_size(), _R.cols_size());
    size_t r = 0;
    for (size_t i = 0; i < min_dim; ++i) {
        if (std::abs(_R(i, i)) >= tol) {
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
    if (_R.rows_size() == 0 || _R.cols_size() == 0) {
        throw std::runtime_error("Call qr() before solveUpperTriangular().");
    }
    if (effective_rank > std::min(_R.rows_size(), _R.cols_size())) {
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
            for (size_t j = i + 1; j < effective_rank; ++j) {
                sum -= _R(i, j) * x(j, rhsCol);
            }
            const double d = _R(i, i);
            if (std::abs(d) < _factorization_threshold) {
                throw std::runtime_error(
                    "solveUpperTriangular: encountered a pivot below the numerical rank threshold.");
            }
            x(i, rhsCol) = sum / d;
        }
    }
    return x;
}

Matrix<> SparseQR::Q() const {
    if (_R.rows_size() == 0 || _R.cols_size() == 0) {
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
    return _R;
}
