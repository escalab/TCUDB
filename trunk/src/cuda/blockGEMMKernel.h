#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef BLOCK_HALF
void msplitm(char transa, char transb, unsigned long m, unsigned long n, unsigned long k, 
    float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);
#else
void msplitm(char transa, char transb, unsigned long m, unsigned long n, unsigned long k, 
    float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);
#endif 
