#ifndef QR_H
#define QR_H

#include <cmath>
#include <vector>
#include <utility>
#include "Matrix.h"

// A = QR
// A^{+} = Q^{-1}R^{T}

class QR {
private:
	Matrix<> _A;
	Matrix<> _Q;
	Matrix<> _R;

public:
	QR(const Matrix<> &_A);

	QR(const QR& other);
	QR(QR&& other) noexcept;

	QR& operator=(const QR& other);
	QR& operator=(QR&& other) noexcept;

	friend bool operator==(const QR& A, const QR& B);
	friend bool operator!=(const QR& A, const QR& B);

	void qr();

	// Classical Gram-Schmidt,
	void qrCGS();

	// Modified Gram - Schmidt
	void qrMGS();

    // Iterative Gram-Schmidt
    void qrIGS();

    // Block Gram-Schmidt
    void qrBGS();

    // Reordered Gram-Schmidt
    void qrRGS();

    // Classical Gram-Schmidt with Pivoting
    void qrCGSP();

	// Householder
	void qrHouseholder();

	// Givens rotation
	void qrGivens();

	Matrix<> A() const;
	Matrix<> Q() const;
	Matrix<> R() const;

    Matrix<> solve(const Matrix<>& b) const;

    Matrix<> pseudoInverse() const;
};

#endif // QR_H