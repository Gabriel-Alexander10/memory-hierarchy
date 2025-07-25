#include "gemm.h"

void naive_gemm(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) {
    #if MMORDER == 0  // ijk
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    #elif MMORDER == 1  // ikj
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] = (k == 0) ? 0 : C[i][j];
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    #elif MMORDER == 2  // jik
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    #elif MMORDER == 3  // jki
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < N; i++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    #elif MMORDER == 4  // kij
        for (int k = 0; k < N; k++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    #elif MMORDER == 5  // kji
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    #endif
}