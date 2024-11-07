#include <gtest/gtest.h>

#include "Matrix.h"
#include "../headers/decomposition/QR.h"

TEST(QRTests, qrGS_part1) {
    Matrix<> A = {
            {1, 2},
            {3, 4},
            {5, 6}
    };
    QR qr(A);
    qr.qrGS();

    // VolframAlpha result
    Matrix<> Q = {
            { 0.169031, 0.897085},
            { 0.507093, 0.276026},
            { 0.845154, -0.345033}
    };

    double eps = 0.00001;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_TRUE(qr.Q()(i, j) > Q(i, j) - eps && qr.Q()(i, j) < Q(i, j) + eps);
        }
    }

    // VolframAlpha result
    Matrix<> R = {
            { 5.91608, 7.43736},
            { 0.0, 0.828079}
    };

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_TRUE(qr.R()(i, j) > R(i, j) - eps && qr.R()(i, j) < R(i, j) + eps);
        }
    }
}

TEST(QRTests, qrGS_part2) {
    Matrix<> A = {
            {1, 2, 7},
            {3, 4, 8},
            {5, 6, 5}
    };

    QR qr(A);
    qr.qrGS();

    // VolframAlpha result
    Matrix<> Q = {
            {0.169031, 0.897085, -0.408248},
            {0.507093, 0.276026, 0.816497},
            {0.845154, -0.345033, -0.408248}
    };

    double eps = 0.00001;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_TRUE(qr.Q()(i, j) > Q(i, j) - eps && qr.Q()(i, j) < Q(i, j) + eps);
        }
    }

    // VolframAlpha result
    Matrix<> R = {
            {5.91608, 7.43736, 9.46573},
            {0, 0.828079, 6.76264},
            {0, 0, 1.63299}
    };

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_TRUE(qr.R()(i, j) > R(i, j) - eps && qr.R()(i, j) < R(i, j) + eps);
        }
    }
}