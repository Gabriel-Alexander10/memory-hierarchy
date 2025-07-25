#include "gemm.h"

void vector_gemm(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k += 4) {
                C[i][j] += A[i][k]   * B[k][j]   + 
                           A[i][k+1] * B[k+1][j] + 
                           A[i][k+2] * B[k+2][j] + 
                           A[i][k+3] * B[k+3][j];
            }
        }
    }
}