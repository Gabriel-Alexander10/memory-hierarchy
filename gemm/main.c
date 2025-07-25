#include "gemm.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(NAIVE)
void gemm(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) {
    naive_gemm(A, B, C);
}
#elif defined(VECTOR)
void gemm(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) { vector_gemm(A, B, C); }
#elif defined(TRANSPOSE)
void gemm(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) { transpose_gemm(A, B, C); }
#elif defined(BLOCKED)
void gemm(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) { blocked_gemm(A, B, C); }
#elif defined(GOTO)
void gemm(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) { gotoblas(A, B, C); }
#endif


int main() {
    matrix_type (*A)[N] = malloc(N * N * sizeof(matrix_type));
    matrix_type (*B)[N] = malloc(N * N * sizeof(matrix_type));
    matrix_type (*C)[N] = malloc(N * N * sizeof(matrix_type));

    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (i == j) ? 1 : 0;
            B[i][j] = i * N + j;
            C[i][j] = 0;
        }
    }

    gemm(A, B, C);
    free(A); free(B); free(C);

    return 0;
}
