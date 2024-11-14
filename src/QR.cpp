#include "QR.h"

QR::QR(const Matrix<>& _A) {
    if (_A.rows_size() < 1 || _A.cols_size() < 1) {
        throw std::runtime_error("Matrix should be: rows > 0 && cols > 0");
    }
    this->_A = _A;
}

QR::QR(const QR &other) : _A(other._A), _Q(other._Q), _R(other._R) {}

void QR::qr() {
    qrIMGS();
}

QR::QR(QR &&other) noexcept : _A(std::move(other._A)), _Q(std::move(other._Q)), _R(std::move(other._R)) {}

QR &QR::operator=(const QR &other) {
    QR temp(other);
    std::swap(_A, temp._A);
    std::swap(_Q, temp._Q);
    std::swap(_R, temp._R);
    return *this;
}

QR& QR::operator=(QR&& other) noexcept {
    QR temp(std::move(other));
    std::swap(_A, temp._A);
    std::swap(_Q, temp._Q);
    std::swap(_R, temp._R);
    return *this;
}

bool operator==(const QR &A, const QR &B) {
    return A.A() == B.A() && A.Q() == B.Q() && A.R() == B.R();
}

bool operator!=(const QR &A, const QR &B) {
    return !(A == B);
}

void QR::qrCGS() {
    size_t m = _A.rows_size();
    size_t n = _A.cols_size();
    size_t min_mn = std::min(m, n);

    _Q = Matrix<>(m, min_mn);
    _R = Matrix<>(min_mn, n);

    for (size_t i = 0; i < n; ++i) {
        std::vector<double> v_i = _A.getCol(i);
        std::vector<double> u_i = v_i;

        for (size_t j = 0; j < min_mn && j < i; ++j) {
            std::vector<double> e_j = _Q.getCol(j);
            double proj_scalar = 0.0;

            for (size_t k = 0; k < m; ++k) {
                proj_scalar += v_i[k] * e_j[k];
            }

            _R(j, i) = proj_scalar;

            for (size_t k = 0; k < m; ++k) {
                u_i[k] -= proj_scalar * e_j[k];
            }
        }

        if (i < min_mn) {
            double normVec = 0.0;
            for (size_t k = 0; k < m; ++k) {
                normVec += u_i[k] * u_i[k];
            }
            normVec = sqrt(normVec);

            if (normVec > 1e-10) {
                _R(i, i) = normVec;

                std::vector<double> e_i(m);
                for (size_t k = 0; k < m; ++k) {
                    e_i[k] = u_i[k] / normVec;
                }

                _Q.setCol(e_i, i);
            }
            else {
                _R(i, i) = 0.0;
            }
        }
    }
}

void QR::qrMGS() {
    size_t m = _A.rows_size();
    size_t n = _A.cols_size();
    size_t min_mn = std::min(m, n);

    _Q = Matrix<>(m, min_mn); // Orthogonal matrix
    _R = Matrix<>(min_mn, n); // Upper triangular matrix

    std::vector<std::vector<double>> V(n); // Vectors to be orthogonalized

    // Initialize V with the columns of A
    for (size_t i = 0; i < n; ++i) {
        V[i] = _A.getCol(i);
    }

    const double epsilon = 1e-10;

    for (size_t j = 0; j < min_mn; ++j) {
        // Compute R(j, j) = ||V[j]|| (2-norm)
        double normVec = 0.0;
        for (size_t k = 0; k < m; ++k) {
            normVec += V[j][k] * V[j][k];
        }
        normVec = sqrt(normVec);

        _R(j, j) = normVec;

        if (normVec > epsilon) {
            // Normalize Q(:, j) = V[j] / R(j, j)
            std::vector<double> q_j(m);
            for (size_t k = 0; k < m; ++k) {
                q_j[k] = V[j][k] / normVec;
            }
            _Q.setCol(q_j, j);

            // Update the remaining vectors in V
            for (size_t k = j + 1; k < n; ++k) {
                // R(j, k) = Q(:, j)^T * V[k]
                double dotProduct = 0.0;
                for (size_t l = 0; l < m; ++l) {
                    dotProduct += q_j[l] * V[k][l];
                }
                _R(j, k) = dotProduct;

                // V[k] = V[k] - R(j, k) * Q(:, j)
                for (size_t l = 0; l < m; ++l) {
                    V[k][l] -= dotProduct * q_j[l];
                }
            }
        }
        else {
            std::vector<double> zero_col(m, 0.0);
            _Q.setCol(zero_col, j);

            for (size_t k = j + 1; k < n; ++k) {
                _R(j, k) = 0.0;
            }
        }
    }
}

void QR::qrIMGS() {
    size_t m = _A.rows_size();
    size_t n = _A.cols_size();
    size_t min_mn = std::min(m, n);

    // init Q and R
    _Q = Matrix<>(m, min_mn);
    _R = Matrix<>(min_mn, n);

    // init V with cols A
    std::vector<std::vector<double>> V(n, std::vector<double>(m));
    for (size_t i = 0; i < n; ++i) {
        V[i] = _A.getCol(i);
    }

    const double epsilon = 1e-10; // Orthogonality accuracy
    const int max_iterations = 10;
    int iteration = 0;
    bool is_orthogonal = false;

    while (iteration < max_iterations && !is_orthogonal) {
        _Q = Matrix<>(m, min_mn);
        _R = Matrix<>(min_mn, n);

        // Using MGS
        for (size_t j = 0; j < min_mn; ++j) {
            // norm V[j]
            double normVec = 0.0;
            for (size_t k = 0; k < m; ++k) {
                normVec += V[j][k] * V[j][k];
            }
            normVec = sqrt(normVec);

            _R(j, j) = normVec;

            if (normVec > epsilon) {
                // normalize Q col
                std::vector<double> q_j(m);
                for (size_t k = 0; k < m; ++k) {
                    q_j[k] = V[j][k] / normVec;
                }
                _Q.setCol(q_j, j);

                // Orthogonalize other vectors
                for (size_t k = j + 1; k < n; ++k) {
                    // Compute R(j, k) = Q(:, j)^T * V[k]
                    double dotProduct = 0.0;
                    for (size_t l = 0; l < m; ++l) {
                        dotProduct += q_j[l] * V[k][l];
                    }
                    _R(j, k) = dotProduct;

                    // Update V[k] = V[k] - R(j, k) * Q(:, j)
                    for (size_t l = 0; l < m; ++l) {
                        V[k][l] -= dotProduct * q_j[l];
                    }
                }
            }
            else {
                // If norm too small, col be zero
                std::vector<double> zero_col(m, 0.0);
                _Q.setCol(zero_col, j);
                for (size_t k = j + 1; k < n; ++k) {
                    _R(j, k) = 0.0;
                }
            }
        }

        // Check Orthogonality Q matrix
        is_orthogonal = true;
        for (size_t i = 0; i < min_mn && is_orthogonal; ++i) {
            for (size_t j = i + 1; j < min_mn && is_orthogonal; ++j) {
                double dot = 0.0;
                std::vector<double> col_i = _Q.getCol(i);
                std::vector<double> col_j = _Q.getCol(j);
                for (size_t k = 0; k < m; ++k) {
                    dot += col_i[k] * col_j[k];
                }
                if (std::abs(dot) > epsilon) {
                    is_orthogonal = false;
                }
            }
        }

        iteration++;
    }

    if (!is_orthogonal) {
        throw std::runtime_error("Warning: Iterative Gram-Schmidt did not achieve desired orthogonality after ");
    }
}

void QR::qrBGS() {

}

void QR::qrRGS() {

}

void QR::qrCGSP() {

}

void QR::qrHouseholder() {

}

void QR::qrGivens() {

}

Matrix<> QR::solve(const Matrix<>& b) const // TODO: write test
{
    // Compute Q^T * b
    Matrix<> Qt_b = Q().transpose() * b;

    // Solve Rx = Qt_b for x using back-substitution
    Matrix<> x = (R().inverse(), Qt_b);
    return x;
}

Matrix<> QR::pseudoInverse() const // TODO: write test
{
    // A^{+} = R^{-1} * Q^T
    Matrix<> R = (_R + Matrix<>::identity(_R.rows_size()) * 1e-8);
    /*
    std::cout << "R: " << std::endl;
    for (size_t i = 0; i < R.rows_size(); ++i) {
        for (size_t j = 0; j < R.cols_size(); ++j) {
            std::cout << _R(i, j) << " ";
        }
        std::cout << std::endl;
    }
    */
    Matrix<> R_inv = R.inverse();
    return R_inv * _Q.transpose();
}

Matrix<> QR::A() const {
    return _A;
}

Matrix<> QR::Q() const {
    return _Q;
}

Matrix<> QR::R() const {
    return _R;
}