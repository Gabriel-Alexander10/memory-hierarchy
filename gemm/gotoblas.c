#include <stdio.h>
#include "gemm.h"

#define L3_size 64 * 1024 / 4
#define L2_size 16 * 1024 / 4
#define L1_size 4 * 1024 / 4
#define R 4

// Micro-kernel: C[mr x nr] += A[mr x kc] * B[kc x nr]
void micro_kernel(
    matrix_type *A,
    matrix_type *B,
    matrix_type *C,
    int ldc, int lda, int ldb,
    int kc, int nr, int mr
) {
    for (int j = 0; j < nr; j++) {
        for (int i = 0; i < mr; i++) {
            double sum = C[i * ldc + j];

            for (int p = 0; p < kc; p++) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }

            C[i * ldc + j] = sum;
        }
    }
}

void gotoblas_inner_loop(
    matrix_type A[N][N], 
    matrix_type B[N][N], 
    matrix_type C[N][N],
    int c_row, int c_col,
    int a_row, int a_col,
    int b_row, int b_col,
    int mc, int kc, int nr, int mr
) {
    // C [mr x nr] blocks
    for (int i = 0; i < mc; i += mr) {
        micro_kernel(
            &A[a_row + i][a_col],
            &B[b_row][b_col],
            &C[c_row + i][c_col],
            N, N, N,
            kc, nr, mr
        );
    }
}

void gotoblas(matrix_type A[N][N], matrix_type B[N][N], matrix_type C[N][N]) {
    int kc = L3_size / N; // L3 = RAM (?)
    if (kc > N) kc = N;
    int mc = L2_size / kc;
    if (mc > N) mc = N;

    int nr = L1_size / kc;
    if (nr > N) nr = N;

    int mr = R / nr;
    if (mr == 0) mr = 1;

    for (int a_col = 0; a_col < N; a_col += kc) {
        for (int a_row = 0; a_row < N; a_row += mc) {
            for (int b_col = 0; b_col < N; b_col += nr) {
                // - A block: [a_row, a_col] (size mc x kc)
                // - B block: [a_col, b_col] (size kc x nr) 

                gotoblas_inner_loop(
                    A, B, C,
                    a_row, b_col,  // C block
                    a_row, a_col,  // A block
                    a_col, b_col,  // B block
                    mc, kc, nr, mr
                );
            }
        }
    }
}
