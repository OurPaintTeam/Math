#ifndef QR_H
#define QR_H

#include <cmath>
#include <vector>
#include "Matrix.h"

// A = QR
// A^{+} = Q^{-1}R^{T}

class QR {
private:
    Matrix<> _A;
    Matrix<> _Q;
    Matrix<> _R;

public:
    QR(const Matrix<>& _A);

    QR(const QR& other);
    QR(QR&& other);

    QR& operator=(const QR& other);
    QR& operator=(QR&& other) noexcept;

    friend bool operator==(const QR& A, const QR& B);
    friend bool operator!=(const QR& A, const QR& B);

    void qr();

    // Gram - Schmidt
    void qrGS();

    // modified Gram - Schmidt
    void qrMGS();

    // Householder
    void qrHouseholder();

    // givens rotation
    void qrGivens();


    Matrix<> A() const { return _A; }
    Matrix<> Q() const { return _Q; }
    Matrix<> R() const { return _R; }
};

inline QR::QR(const Matrix<>& _A) : _A(_A), _Q(), _R() {
}

inline void QR::qr() {
    qrGS();
}

inline void QR::qrGS()
{
    size_t m = _A.rows_size();
    size_t n = _A.cols_size();

    _Q = Matrix<>(m, n);
    _R = Matrix<>(n, n);

    for (size_t i = 0; i < n; ++i) {
        std::vector<double> v_i = _A.getCol(i);
        std::vector<double> u_i = v_i;

        for (size_t j = 0; j < i; ++j) {
            std::vector<double> e_j = _Q.getCol(j);
            double proj_scalar = 0;

            for (size_t k = 0; k < m; ++k) {
                proj_scalar += v_i[k] * e_j[k];
            }

            _R(j, i) = proj_scalar;

            for (size_t k = 0; k < m; ++k) {
                u_i[k] -= proj_scalar * e_j[k];
            }
        }

        // norm
        double normVec = 0;
        for (size_t k = 0; k < m; ++k) {
            normVec += u_i[k] * u_i[k];
        }
        normVec = sqrt(normVec);

        _R(i, i) = normVec;

        std::vector<double> e_i(m);
        for (size_t k = 0; k < m; ++k) {
            e_i[k] = u_i[k] / normVec;
        }

        _Q.setCol(e_i, i);
    }
}

#endif // QR_H