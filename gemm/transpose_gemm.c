#include "gemm.h"

void transpose_gemm(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) {
    matrix_type BT[N][N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[j][i] = B[i][j];
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * BT[j][k];
            }
        }
    }
}