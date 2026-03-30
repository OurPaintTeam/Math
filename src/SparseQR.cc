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
    : _A(other._A), _R(other._R), _ref_ptr(other._ref_ptr), _ref_idx(other._ref_idx),
      _ref_val(other._ref_val), _ref_beta(other._ref_beta) {}

SparseQR::SparseQR(SparseQR&& other) noexcept
    : _A(std::move(other._A)), _R(std::move(other._R)), _ref_ptr(std::move(other._ref_ptr)),
      _ref_idx(std::move(other._ref_idx)), _ref_val(std::move(other._ref_val)),
      _ref_beta(std::move(other._ref_beta)) {}

SparseQR& SparseQR::operator=(const SparseQR& other) {
    SparseQR temp(other);
    std::swap(_A, temp._A);
    std::swap(_R, temp._R);
    std::swap(_ref_ptr, temp._ref_ptr);
    std::swap(_ref_idx, temp._ref_idx);
    std::swap(_ref_val, temp._ref_val);
    std::swap(_ref_beta, temp._ref_beta);
    return *this;
}

SparseQR& SparseQR::operator=(SparseQR&& other) noexcept {
    SparseQR temp(std::move(other));
    std::swap(_A, temp._A);
    std::swap(_R, temp._R);
    std::swap(_ref_ptr, temp._ref_ptr);
    std::swap(_ref_idx, temp._ref_idx);
    std::swap(_ref_val, temp._ref_val);
    std::swap(_ref_beta, temp._ref_beta);
    return *this;
}

bool operator==(const SparseQR& A, const SparseQR& B) {
    return A.A() == B.A() && A.Q() == B.Q() && A.R() == B.R();
}

bool operator!=(const SparseQR& A, const SparseQR& B) {
    return !(A == B);
}

void SparseQR::qr() {
    const size_t m = _A.rows_size();
    const size_t n = _A.cols_size();
    const size_t min_mn = std::min(m, n);

    std::vector<double> data(m * n, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (SparseMatrix<>::InnerIterator it(_A, i); it; ++it) {
            data[it.col() * m + i] = it.value();
        }
    }
    _ref_ptr.assign(min_mn + 1, 0);
    _ref_idx.clear();
    _ref_val.clear();
    _ref_beta.assign(min_mn, 0.0);

    _ref_idx.reserve(m * 2);
    _ref_val.reserve(m * 2);
    for (size_t k = 0; k < min_mn; ++k) {
        double* __restrict__ ck = data.data() + k * m;

        double sigma = 0.0;
        for (size_t i = k; i < m; ++i) sigma += ck[i] * ck[i];

        if (sigma < 1e-24) {
            _ref_ptr[k + 1] = _ref_ptr[k];
            continue;
        }

        double norm = std::sqrt(sigma);
        double alpha = (ck[k] >= 0.0) ? -norm : norm;
        double old_lead = ck[k];
        ck[k] -= alpha;

        double vtv = sigma + ck[k] * ck[k] - old_lead * old_lead;
        double beta = 2.0 / vtv;
        _ref_beta[k] = beta;

        const size_t rstart = _ref_idx.size();
        for (size_t i = k; i < m; ++i) {
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
        std::memset(ck + k + 1, 0, (m - k - 1) * sizeof(double));

        for (size_t j = k + 1; j < n; ++j) {
            double* __restrict__ cj = data.data() + j * m;

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

    _R = Matrix<>(min_mn, n);
    for (size_t j = 0; j < n; ++j) {
        const double* col = data.data() + j * m;
        const size_t lim = std::min(j + 1, min_mn);
        for (size_t i = 0; i < lim; ++i) {
            _R(i, j) = col[i];
        }
    }
}

Matrix<> SparseQR::solve(const Matrix<>& b, double damping) const {
    if (_R.rows_size() == 0 || _R.cols_size() == 0) {
        throw std::runtime_error("Call qr() before solve().");
    }
    if (b.rows_size() != _A.rows_size()) {
        throw std::invalid_argument("Right-hand side rows must match A rows.");
    }
    const size_t m = _A.rows_size();
    const size_t n = _A.cols_size();
    const size_t nrhs = b.cols_size();
    Matrix<> y = applyQt(b);
    if (m >= n) {
        Matrix<> yHead(n, nrhs);
        for (size_t i = 0; i < n; ++i) {
            for (size_t rhs = 0; rhs < nrhs; ++rhs) {
                yHead(i, rhs) = y(i, rhs);
            }
        }
        Matrix<> x(n, nrhs);
        const double diag_tol = std::max(damping, 1e-12);
        for (size_t rhs = 0; rhs < nrhs; ++rhs) {
            for (size_t ii = n; ii > 0; --ii) {
                const size_t i = ii - 1;
                double sum = yHead(i, rhs);
                for (size_t j = i + 1; j < n; ++j) {
                    sum -= _R(i, j) * x(j, rhs);
                }
                double d = _R(i, i);
                if (std::abs(d) < diag_tol) {
                    if (d >= 0.0) d += diag_tol;
                    else d -= diag_tol;
                }
                x(i, rhs) = sum / d;
            }
        }
        return x;
    }
    return pseudoInverse(damping) * b;
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
    return inv * Rt * Q().transpose();
}

size_t SparseQR::rank(double tol) const {
    if (_R.rows_size() == 0 || _R.cols_size() == 0) {
        throw std::runtime_error("Call qr() before rank().");
    }
    const size_t min_dim = std::min(_R.rows_size(), _R.cols_size());
    double max_diag = 0.0;
    for (size_t i = 0; i < min_dim; ++i) {
        max_diag = std::max(max_diag, std::abs(_R(i, i)));
    }
    if (max_diag == 0.0) {
        return 0;
    }
    double threshold = tol;
    if (threshold < 0.0) {
        const double eps = std::numeric_limits<double>::epsilon();
        threshold = eps * std::max(_A.rows_size(), _A.cols_size()) * max_diag;
    }
    size_t r = 0;
    for (size_t i = 0; i < min_dim; ++i) {
        if (std::abs(_R(i, i)) > threshold) {
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
