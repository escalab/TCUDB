/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
/*#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384*/
#define MATRIX_M 16
#define MATRIX_N 16
#define MATRIX_K 16
//#define MATRIX_K 10240
//#define MATRIX_N 10240
//#define MATRIX_K 10240
/*#define MATRIX_M 2048
#define MATRIX_N 2048
#define MATRIX_K 2048*/



// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

int large_mat1[256], large_mat2[256];
char large_mat1_txt[15] = "large_mat1.txt";
char large_mat2_txt[15] = "large_mat2.txt";

__host__ void read_txt(char * infile, int * matrix) {
    FILE* fp = fopen(infile, "r");
    if (fp == NULL) {
        printf("File does not exist.");
        return;
    }

    for (int i = 0; i < 256; i++) {
        fscanf(fp, "%d", &matrix[i]);
    }

}

// TODO:convert to fp16 and fp32
__host__ void convertIntToFp16 (half *out, int *in) {
//__global__ void convertIntToFp16 (half *out, int *in, int n) {
   /* 
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }*/
    for (int i = 0; i < 256; i++)
        out[i] = (half)in[i];
}

__host__ void convertIntToFp32 (float *out, int *in) {
//__global__ void convertIntToFp32 (float *out, int *in, int n) {
    /*
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }*/
    for(int i = 0; i < 256; i++)
        out[i] = (float)in[i];
}

__host__ void generate_data_fp16(half *a, half *b) {
  
    for (int i = 0; i < WMMA_M; i++) {
      for (int j = 0; j < WMMA_K; j++) {
        a[i * WMMA_K + j] = (half)0;
      }
    }

    a[0] = (half)(7);
    a[2] = (half)(1);
    a[18] = (half)(-2);
    a[32] = (half)(-10);
    a[33] = (half)(4);

    for (int i = 0; i < WMMA_N; i++) {
      for (int j = 0; j < WMMA_K; j++) {
        b[i * WMMA_K + j] = (half)0;
      }
    }

    b[0] = (half)(4);
    b[1] = (half)(-5);
    b[16] = (half)(11);
    b[18] = (half)(1);
    b[32] = (half)(-3);
    b[33] = (half)(6);

}

__host__ void generate_data_fp32(float *a, float *b) {
  
    for (int i = 0; i < WMMA_M; i++) {
      for (int j = 0; j < WMMA_K; j++) {
        a[i * WMMA_K + j] = (float)0;
      }
    }

    a[0] = (float)(7);
    a[2] = (float)(1);
    a[18] = (float)(-2);
    a[32] = (float)(-10);
    a[33] = (float)(4);

    for (int i = 0; i < WMMA_N; i++) {
      for (int j = 0; j < WMMA_K; j++) {
        b[i * WMMA_K + j] = (float)0;
      }
    }

    b[0] = (float)(4);
    b[1] = (float)(-5);
    b[16] = (float)(11);
    b[18] = (float)(1);
    b[32] = (float)(-3);
    b[33] = (float)(6);

}
// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

int main(int argc, char* argv[]) {
   half *a_h = NULL;
   half *b_h = NULL;
   float *a_32h = NULL;
   float *b_32h = NULL;
   float *c_h = NULL;

   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas;
   float *c_wmma;
   float *c_sgemm;
   float *c_cublas_gemmEx;

   float *c_host_cublas;
   float *c_host_cublasCublasGemmEx;
   float *c_host_wmma;
   float *c_host_sgemm;

   float alpha = 2.0f;
   float beta = 2.0f;
   
   a_h = (half*)malloc(sizeof(half) * MATRIX_M * MATRIX_K);
   b_h = (half*)malloc(sizeof(half) * MATRIX_K * MATRIX_N);
   c_h = (float*)malloc(sizeof(float) * MATRIX_M * MATRIX_N);
   a_32h = (float*)malloc(sizeof(float) * MATRIX_M * MATRIX_K);
   b_32h = (float*)malloc(sizeof(float) * MATRIX_K * MATRIX_N);

   curandGenerator_t gen;
   cublasHandle_t cublasHandle;
   cublasHandle_t cublasHandle_default;
   
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaEvent_t startcublasEX;
   cudaEvent_t stopcublasEX;

   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   cudaEvent_t startcublasCublasGemmEx;
   cudaEvent_t stopcublasCublasGemmEx;
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));

   cudaErrCheck(cudaEventCreate(&startcublasEX));
   cudaErrCheck(cudaEventCreate(&stopcublasEX));

   cudaErrCheck(cudaEventCreate(&startcublasCublasGemmEx));
   cudaErrCheck(cudaEventCreate(&stopcublasCublasGemmEx));
   
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   cublasErrCheck(cublasCreate(&cublasHandle_default));
   
   // Use tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
   cublasErrCheck(cublasSetMathMode(cublasHandle_default, CUBLAS_DEFAULT_MATH));
   
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas_gemmEx, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_sgemm, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_cublasCublasGemmEx = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_sgemm = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL)); // usgined long long

   //FIXME: manually change to certain number for comparison for now
   read_txt(large_mat1_txt, large_mat1);
   read_txt(large_mat2_txt, large_mat2);

   // convert to corresponding type
   //convertIntToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_h, large_mat1, MATRIX_M * MATRIX_K);
   //convertIntToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_h, large_mat2, MATRIX_K * MATRIX_N);
   convertIntToFp16(a_h, large_mat1);
   convertIntToFp16(b_h, large_mat2);

   //generate_data_fp16(a_h, b_h);
   //generate_data_fp32(a_32h, b_32h);
   //convertIntToFp32 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_32h, large_mat1, MATRIX_M * MATRIX_K);
   //convertIntToFp32 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_32h, large_mat2, MATRIX_K * MATRIX_N);
   convertIntToFp32(a_32h, large_mat1);
   convertIntToFp32(b_32h, large_mat2);

   cudaErrCheck(cudaMemcpy(a_fp16, a_h, sizeof(half) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_fp16, b_h, sizeof(half) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(a_fp32, a_32h, sizeof(float) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_fp32, b_32h, sizeof(float) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));

   // curand will create data directly on device -- don't need to cudaMemcpy host->device
   //curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   //curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   //convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   //convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
   
   curandErrCheck(curandDestroyGenerator(gen));
   
   cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(c_cublas_gemmEx, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));



   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   printf("Running with sgemm...\n"); // cuBLAS, single precision FP32
   cudaErrCheck(cudaEventRecord(startcublas));
   cublasSgemm(cublasHandle_default, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, a_fp32, MATRIX_M, b_fp32, MATRIX_N, &beta, c_sgemm, MATRIX_K);
//   wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   cudaErrCheck(cudaEventRecord(stopcublas));
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);


   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   cudaErrCheck(cudaEventRecord(stopWMMA));


   // Now using cuBLAS but not tensor
   printf("Running with cuBLAS (GemmEX) on GPUs...\n"); // general matmul extension with FP16 on GPUs
   cudaErrCheck(cudaEventRecord(startcublasCublasGemmEx));
   cublasErrCheck(cublasGemmEx(cublasHandle_default, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas_gemmEx, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT));
   cudaErrCheck(cudaEventRecord(stopcublasCublasGemmEx));

//   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
   
   // Now using cuBLAS with TCU (cublasHandle), FP16
   printf("Running with cuBLAS (GemmEx) on TCUs...\n");
   cudaErrCheck(cudaEventRecord(startcublasEX));
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
   cudaErrCheck(cudaEventRecord(stopcublasEX));

   // Error checking
   cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c_host_cublasCublasGemmEx, c_cublas_gemmEx, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c_host_sgemm, c_cublas_gemmEx, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

   // only compare result need to cudaMemcpy to copy data back to CPU
   printf("\nChecking results with cublas (cublasGemmEx)...\n");
   int errors_default = 0;
   for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
      float v1 = c_host_sgemm[i];
      float v2 = c_host_cublasCublasGemmEx[i];
//      if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-5) {
      if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-3) {
         errors_default++;
         if (errors_default < 10) printf("%f %f\n", v1, v2);
      }
   }
   
   printf("\nChecking results with tensor cores...\n");
   // 0.01% relative tolerance. 1e-5 absolute tolerance.
   int errors = 0;
   for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
      float v1 = c_host_sgemm[i];
      float v2 = c_host_cublas[i];
      if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-3) {
         errors++;
         if (errors < 10) printf("%f %f\n", v1, v2);
      }
   }

   
   if (errors_default > 0) {
      printf("WMMA does not agree with cuBLAS default! %d errors!\n", errors);
   }
   if (errors > 0) {
      printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
   }

//   else {
   {
      printf("Results verified: cublas and WMMA agree.\n\n");
      float wmmaTime;
      float cublasTime;
      cudaErrCheck(cudaEventSynchronize(stopWMMA));
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      printf("wmma took %fms\n", wmmaTime);
      printf("cublas (FP32) took %fms\n", cublasTime); // FP32
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublasCublasGemmEx, stopcublasCublasGemmEx));
      printf("cublas cublasGemmEx (FP16 on GPUs) took %fms\n", cublasTime); // FP16
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublasEX, stopcublasEX));
      printf("cublas tensor cores (FP16 on TCUs) took %fms\n", cublasTime); // FP16

      printf("\nFor a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was written as a demo only!\n\n");
   }
   
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));

   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_cublas));
   cudaErrCheck(cudaFree(c_wmma));
   
   free(c_host_cublas);
   free(c_host_wmma);
   free(a_h);
   free(b_h);
   free(c_h);
   free(a_32h);
   free(b_32h);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}

