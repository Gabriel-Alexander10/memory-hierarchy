#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>


/* To save you time, we are including all 6 variants of the loop ordering
   as separate functions and then calling them using function pointers.
   The reason for having separate functions that are nearly identical is
   to avoid counting any extraneous processing towards the computation
   time.  This includes I/O accesses (printf) and conditionals (if/switch).
   I/O accesses are slow and conditional/branching statements could
   unfairly bias results (lower cases in switches must run through more
   case statements on each iteration).
*/
void multMat0( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is ijk loop order. */
    for( i = 0; i < n; i++ )
        for( j = 0; j < n; j++ )
            for( k = 0; k < n; k++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat1( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is ikj loop order. */
    for( i = 0; i < n; i++ )
        for( k = 0; k < n; k++ )
            for( j = 0; j < n; j++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat2( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is jik loop order. */
    for( j = 0; j < n; j++ )
        for( i = 0; i < n; i++ )
            for( k = 0; k < n; k++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat3( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is jki loop order. */
    for( j = 0; j < n; j++ )
        for( k = 0; k < n; k++ )
            for( i = 0; i < n; i++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat4( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is kij loop order. */
    for( k = 0; k < n; k++ )
        for( i = 0; i < n; i++ )
            for( j = 0; j < n; j++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat5( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is kji loop order. */
    for( k = 0; k < n; k++ )
        for( j = 0; j < n; j++ )
            for( i = 0; i < n; i++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

#ifndef NMAX
#define NMAX 128
#endif

#ifndef MMORDER
#define MMORDER 0
#endif

double drand48(void)
{
  unsigned long cycles;
  asm volatile ("rdcycle %0" : "=r" (cycles));
  return (double)cycles;
}

int main( int argc, char **argv ) {
    int i;
    float *A = malloc(NMAX*NMAX * sizeof(float));
    float *B = malloc(NMAX*NMAX * sizeof(float));
    float *C = malloc(NMAX*NMAX * sizeof(float));

    void (*orderings[])(int,float *,float *,float *) =
        {&multMat0,&multMat1,&multMat2,&multMat3,&multMat4,&multMat5};
    char *names[] = {"ijk","ikj","jik","jki","kij","kji"};

    for( i = 0; i < NMAX*NMAX; i++ ) A[i] = drand48()*2-1;
    for( i = 0; i < NMAX*NMAX; i++ ) B[i] = drand48()*2-1;
    for( i = 0; i < NMAX*NMAX; i++ ) C[i] = drand48()*2-1;

    (*orderings[MMORDER])( NMAX, A, B, C );

    printf("%s done\n", names[MMORDER]);

    free(A); free(B); free(C);
    return 0;
}
