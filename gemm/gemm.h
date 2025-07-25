#ifndef GEMM_H
#define GEMM_H

#ifndef N
#define N 64
#endif
typedef int matrix_type;

void gemm(matrix_type[N][N], matrix_type[N][N], matrix_type[N][N]);

void naive_gemm(matrix_type[N][N], matrix_type[N][N], matrix_type[N][N]);
void vector_gemm(matrix_type[N][N], matrix_type[N][N], matrix_type[N][N]);
void transpose_gemm(matrix_type[N][N], matrix_type[N][N], matrix_type[N][N]);
void blocked_gemm(matrix_type[N][N], matrix_type[N][N], matrix_type[N][N]);
void gotoblas(matrix_type[N][N], matrix_type[N][N], matrix_type[N][N]);

#endif