/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include "../include/common.h"
#include "../include/tcuJoin.h"
#include "../include/gpuCudaLib.h"
#include "scanImpl.cu"
#include <cuda_fp16.h>
#include <curand.h>
#include <mma.h>
#include <cublas_v2.h>
#include "../include/uthash.h"

using namespace nvcuda;

// For wmma API, these must be multiples fo 16
//#define MATRIX_M 16
//#define MATRIX_N 16
//#define MATRIX_K 16

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Define some error checking macros.

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

typedef struct gb_hash {
    int left_col_idx;                    /* key */
    struct gb_hash *right_col_idx;
    int val;
    UT_hash_handle hh;         /* makes this structure hashable */
} gb_hash_t;

/*
#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}
*/

/* Map the table entires into matrix for tensor core to use 
 * Assum both matrix have the same dimension and the value in INT type for now, e.g., both 16x16 dim
 * To support multiple types, this function need to be modified
 */
__host__ void static fill_matrix(struct joinNode *jNode, int * matrix1, int * matrix2, int width,
        int attr_num1, int attr_num2, int attr_type1, int attr_type2) {
    int *mat1_i, *mat1_j, *mat1_val;
    int *mat2_i, *mat2_j, *mat2_val;

    int leftTupleNum = jNode->leftTable->tupleNum;
    int rightTupleNum = jNode->rightTable->tupleNum;
 
    mat1_i = (int*)malloc(sizeof(int) * leftTupleNum); 
    mat1_j = (int*)malloc(sizeof(int) * leftTupleNum); 
    mat1_val = (int*)malloc(sizeof(int) * leftTupleNum); 
   
    mat2_i = (int*)malloc(sizeof(int) * rightTupleNum); 
    mat2_j = (int*)malloc(sizeof(int) * rightTupleNum); 
    mat2_val = (int*)malloc(sizeof(int) * rightTupleNum); 

    int i, j; 
    for (i = 0; i < attr_num1; i++) {
        int left_col_idx = jNode->leftTable->attrIndex[i];
        int k = 0; // k is row-index of the table (tupleNum index)
        
        for (j = 0; j < leftTupleNum * attr_type1; j+=attr_type1) {
            int *temp;
            temp = (int*)(&jNode->leftTable->content[i][j]);
            
            if (left_col_idx == 0) { // match to schema's i
                mat1_i[k] = *temp;
            }
            else if (left_col_idx == 1) { // match to schema's j
                mat1_j[k] = *temp;
            }
            else { // match to schema's val
                // read 4 bytes at once because the type is int
                mat1_val[k] = *temp;
            }
            k++;
        }
    }

    
    for (i = 0; i < attr_num2; i++) {
        int right_col_idx = jNode->rightTable->attrIndex[i];
        int k = 0;
        
        for (j = 0; j < rightTupleNum * attr_type2; j+=attr_type2) {
            int *temp;
            temp = (int*)(&jNode->rightTable->content[i][j]);
            
            if (right_col_idx == 0) {
                mat2_i[k] = *temp;
            }
            else if (right_col_idx == 1) {
                mat2_j[k] = *temp;
            }
            else {
                mat2_val[k] = *temp;
            }
            k++;
        }
    }

    // map index to array[width * i + j] = val
    int m;
    for (m = 0; m < leftTupleNum; m++) {
        matrix1[width * mat1_i[m] + mat1_j[m]] = mat1_val[m];
    }
    
    for (m = 0; m < rightTupleNum; m++) {
        matrix2[width * mat2_i[m] + mat2_j[m]] = mat2_val[m];
    }

    free(mat1_i);
    free(mat1_j);
    free(mat1_val);
    free(mat2_i);
    free(mat2_j);
    free(mat2_val);
}

__host__ void static verify_result(float * matrix, int width) { // correct!

    int i;
    for (i = 0; i < width*width; i++) {
        //printf("%d\t", matrix[i]);
        printf("%.2f\t", matrix[i]);
        if ((i+1) % width == 0)
          printf("\n");  
    }

}
/* Printf matrix for debugging */
//__host__ void static print_matrix(int * matrix, int width) {
//__host__ void static print_matrix(float * matrix, int width) { // correct!
__host__ void static print_matrix(half * matrix, int width) { // correct!

    int i;
    for (i = 0; i < width*width; i++) {
        //printf("%d\t", matrix[i]);
        printf("%.2f\t", __half2float(matrix[i]));
        if ((i+1) % width == 0)
          printf("\n");  
    }

}

__host__ void static diff_mat(float *mat1, half *mat2, int width) {
    int i;
    for (i = 0; i < width*width; i++) {
        //printf("%d\t", matrix[i]);
        printf("%.2f\t", mat1[i]-__half2float(mat2[i]));
        if ((i+1) % width == 0)
          printf("\n");  
    }
} 

/* Convert matrix from int to half type */
__global__ void static convertFp32ToFp16(half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
        //out[idx] = in[idx];
    }
}


/* Convert matrix from int to float type */
__host__ void static convertIntToFp32(float *out, int *in, int width) {
    int i;
    for (i = 0; i < width * width; i++) {
        out[i] = (float)in[i]; 
    }
}

/* Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
 *  1) Matrices are packed in memory.
 *  2) M, N and K are multiples of 16.
 *  3) Neither A nor B are transposed.
 */
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

/*
 * tcuJoinn using NVIDIA's WMMA lib to perform matrix multiplication can aggregation..
 *
 * Prerequisites:
 *  1. the data to be joined can be fit into GPU device memory.
 *  2. dimension table is not compressed
 *  
 * Input:
 *  jNode: contains information about the two joined tables.
 *  pp: records statistics such as kernel execution time
 *
 * Output:
 *  A new table node
 */
struct tableNode * tcuJoin(struct joinNode *jNode, struct statistic *pp, int *matrix_dim, struct groupByNode *gbNode){
    // get some atrributes after jNode matched tuples
    int *gpuGbIndex = NULL, gpuTupleNum, gpuGbColNum;
    int gbConstant = 0;

    struct tableNode *res = (struct tableNode *) malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);
    res->tupleSize = gbNode->tupleSize;
    res->totalAttr = gbNode->outputAttrNum;
    res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrType);
    res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrSize);
    res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr); 
    CHECK_POINTER(res->attrTotalSize);
    res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->dataPos);
    res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->dataFormat);
    res->content = (char **) malloc(sizeof(char **) * res->totalAttr);
    CHECK_POINTER(res->content);

    for(int i=0;i<res->totalAttr;i++){
        res->attrType[i] = gbNode->attrType[i];
        res->attrSize[i] = gbNode->attrSize[i];
        res->dataFormat[i] = UNCOMPRESSED;
    }

    gpuTupleNum = gbNode->table->tupleNum;
    gpuGbColNum = gbNode->groupByColNum;

    // groupByIndex == -1 means query doesn't contain group by keyword
    if(gpuGbColNum == 1 && gbNode->groupByIndex[0] == -1){
        gbConstant = 1;
    }

    // need to extract data from agg_cal_cons->calMathExp


    // TODO: implement group by ranking using uthash 
    /*
    printf("groupByColNum: %d\n", gbNode->groupByColNum);
    // currently only support 2 groupBy index
    int leftGbIdx = gbNode->groupByIndex[0];
    int rightGbIdx = gbNode->groupByIndex[1];
    printf("left groupBy index: %d\n", leftGbIdx);
    printf("right groupBy index: %d\n", rightGbIdx);

    gb_hash_t *gb = NULL;

    gb_hash_t *item1, *item2, *tmp1, *tmp2;
    // test 
    gb_hash_t *i = (gb_hash_t*)malloc(sizeof(gb_hash_t));
    i->left_col_idx = 0;
    i->right_col_idx = NULL;
    i->val = 24;
    HASH_ADD_INT(gb, left_col_idx, i);
    gb_hash_t *s = (gb_hash_t*)malloc(sizeof(gb_hash_t));
    s->left_col_idx = 87;
    s->right_col_idx = NULL;
    s->val = 13;
    HASH_ADD_INT(i->right_col_idx, left_col_idx, s);

    HASH_ITER(hh, gb, item1, tmp1) {
        HASH_ITER(hh, item1->right_col_idx, item2, tmp2) {
            printf("$items{%d}{%d} = %d\n", item1->left_col_idx, item2->left_col_idx, item2->val);
        }
    }
    */

    int MATRIX_M, MATRIX_N, MATRIX_K;
    MATRIX_M = MATRIX_N = MATRIX_K = *matrix_dim;
    
    struct timespec tcu_start, tcu_end;
    struct timespec init_start, init_end;
    clock_gettime(CLOCK_REALTIME, &tcu_start);
    clock_gettime(CLOCK_REALTIME, &init_start);

    int *matrix1;
    int *matrix2;

    half *mat1_fp16;
    half *mat2_fp16;
    float *mat1_fp32;
    float *mat2_fp32;

    // on GPU device
    half *mat1_dev;
    half *mat2_dev;
    float *mat1_dev_fp32;
    float *mat2_dev_fp32;
    float *c;

    float *c_wmma;
    //float *c_cublas;
    //float *c_sgemm; // single precision

    // for error checking
    float *c_host_wmma;
    //float *c_host_cublas;
    //float *c_host_sgemm;

    // wmma parameters
    // C = alpha*A*B + beta*C  
    float alpha = 1.0f;
    float beta = 0.0f;

    // For WMMA
    dim3 gridDim;
    dim3 blockDim;
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    matrix1 = (int*)calloc(MATRIX_M*MATRIX_K, sizeof(int));
    matrix2 = (int*)calloc(MATRIX_K*MATRIX_N, sizeof(int));
    //matrix1 = (int*)malloc(sizeof(int)*MATRIX_M*MATRIX_K);
    //matrix2 = (int*)malloc(sizeof(int)*MATRIX_K*MATRIX_N);
    //memset(matrix1, 0, sizeof(int)*MATRIX_M*MATRIX_K);
    //memset(matrix2, 0, sizeof(int)*MATRIX_K*MATRIX_N);

    mat1_fp16 = (half*)malloc(sizeof(half) * MATRIX_M * MATRIX_K);
    mat2_fp16 = (half*)malloc(sizeof(half) * MATRIX_K * MATRIX_N);
    mat1_fp32 = (float*)malloc(sizeof(float) * MATRIX_M * MATRIX_K);
    mat2_fp32 = (float*)malloc(sizeof(float) * MATRIX_K * MATRIX_N);

    //curandGenerator_t gen;
    //cublasHandle_t cublasHandle;         // tcu
    //cublasHandle_t cublasHandle_default; // cublas default

    cudaEvent_t startWMMA;
    cudaEvent_t stopWMMA;
    //cudaEvent_t startcublasEX;
    //cudaEvent_t stopcublasEX;
    //cudaEvent_t startcublas; // for sgemm
    //cudaEvent_t stopcublas;

    cudaErrCheck(cudaEventCreate(&startWMMA));
    cudaErrCheck(cudaEventCreate(&stopWMMA)); 
    //cudaErrCheck(cudaEventCreate(&startcublasEX));
    //cudaErrCheck(cudaEventCreate(&stopcublasEX));
    //cudaErrCheck(cudaEventCreate(&startcublas));
    //cudaErrCheck(cudaEventCreate(&stopcublas));

    // use tensor core or cublas
    //cublasErrCheck(cublasCreate(&cublasHandle));
    //cublasErrCheck(cublasCreate(&cublasHandle_default));

    // enable tensor core
    //cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
    //cublasErrCheck(cublasSetMathMode(cublasHandle_default, CUBLAS_DEFAULT_MATH));

    c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    //c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    //c_host_sgemm = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    clock_gettime(CLOCK_REALTIME, &init_end);

    // fill matrices from jNode by mapping inputs into 1D array
    struct timespec fill_start, fill_end, convert_start, convert_end;
    struct timespec cuMemcpy_start, cuMemcpy_end;
    clock_gettime(CLOCK_REALTIME, &fill_start);
    fill_matrix(jNode, matrix1, matrix2, MATRIX_M, 
            jNode->leftTable->totalAttr, jNode->rightTable->totalAttr, jNode->leftTable->attrType[0], jNode->rightTable->attrType[0]);
    clock_gettime(CLOCK_REALTIME, &fill_end);

    clock_gettime(CLOCK_REALTIME, &convert_start);
    // convert to half type for wmma API
    convertIntToFp32(mat1_fp32, matrix1, MATRIX_M);
    convertIntToFp32(mat2_fp32, matrix2, MATRIX_N);
    clock_gettime(CLOCK_REALTIME, &convert_end);

    // print matrix for debugging
    //printf("Matrix 1:\n");
    //print_matrix(mat1_fp32, MATRIX_M);
    //printf("Matrix 2:\n");
    //print_matrix(mat2_fp32, MATRIX_M);
    

    // copy data to device for computation
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_start);
    cudaErrCheck(cudaMalloc((void**)&mat1_dev, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&mat2_dev, MATRIX_K * MATRIX_N * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&mat1_dev_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&mat2_dev_fp32, MATRIX_K * MATRIX_N * sizeof(float)));

    cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));

    cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));
    //cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
    //cudaErrCheck(cudaMalloc((void**)&c_sgemm, MATRIX_M * MATRIX_N * sizeof(float)));

    cudaErrCheck(cudaMemcpy(mat1_dev_fp32, mat1_fp32, sizeof(float) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(mat2_dev_fp32, mat2_fp32, sizeof(float) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));
    convertFp32ToFp16<<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (mat1_dev, mat1_dev_fp32, MATRIX_M * MATRIX_K);
    convertFp32ToFp16<<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (mat2_dev, mat2_dev_fp32, MATRIX_K * MATRIX_N);
    cudaErrCheck(cudaMemcpy(mat1_fp16, mat1_dev, sizeof(half) * MATRIX_M * MATRIX_K, cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(mat2_fp16, mat2_dev, sizeof(half) * MATRIX_K * MATRIX_N, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);

    printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

    printf("Running with wmma...\n");
    cudaErrCheck(cudaEventRecord(startWMMA));
    wmma_example <<< gridDim, blockDim >>> (mat1_dev, mat2_dev, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta); 
    cudaErrCheck(cudaEventRecord(stopWMMA));
    
    /*
    printf("Running with sgemm...\n");
    cudaErrCheck(cudaEventRecord(startcublas));
    cublasSgemm(cublasHandle_default, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, mat1_dev_fp32, MATRIX_M, mat2_dev_fp32, MATRIX_N, &beta, c_sgemm, MATRIX_K);
    cudaErrCheck(cudaEventRecord(stopcublas));
    */

    /*
    printf("Running with cuBLAS on TCUs...\n");
    //cudaErrCheck(cudaEventRecord(startcublasEX));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha,
                mat1_dev, CUDA_R_16F, MATRIX_M,
                mat2_dev, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
    //cudaErrCheck(cudaEventRecord(stopcublasEX));
    */

    //struct timespec chkRes_start, chkRes_end;
    //clock_gettime(CLOCK_REALTIME,&chkRes_start);
    // Copy result back to the host for error checking
    cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

    //verify_result(c_host_wmma, MATRIX_M);

    //cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    //cudaErrCheck(cudaMemcpy(c_host_sgemm, c_sgemm, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

    // print error checking, cublasGemmEx and cublas
    //printf("\nChecking results with tensor cores...\n");

    // 0.01% relative tolerance. 1e-5 absolute tolerance.
    /*
    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = c_host_sgemm[i];
        float v2 = c_host_cublas[i];

        // TODO: abs diff failed due to precision loss
        // current fix: range value less than 2^10 (IEEE half type)
        if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-3) {
            errors++;
            if (errors < 10) printf("%.1f %.1f diff:%.1f\n", v1, v2, abs(v1-v2));
        }
    }

    if (errors > 0) {
        printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
    }
    */

    //clock_gettime(CLOCK_REALTIME,&chkRes_end);

    // print time
    float wmmaTime;
    //float cublasTime;

    cudaErrCheck(cudaEventSynchronize(stopWMMA));
    //cudaErrCheck(cudaEventSynchronize(stopcublasEX));
    cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    //cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));

    printf("wmma took %fms\n", wmmaTime);
    //printf("cublas (FP32) took %fms\n", cublasTime);
    //cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublasEX, stopcublasEX));
    //printf("cublas tensor cores (FP16) took %fms\n", cublasTime);

    // free those data structures
    cudaErrCheck(cudaEventDestroy(startWMMA));
    cudaErrCheck(cudaEventDestroy(stopWMMA));
    //cudaErrCheck(cudaEventDestroy(startcublasEX));
    //cudaErrCheck(cudaEventDestroy(stopcublasEX));
    //cudaErrCheck(cudaEventDestroy(startcublas));
    //cudaErrCheck(cudaEventDestroy(stopcublas));

    cudaErrCheck(cudaFree(mat1_dev));
    cudaErrCheck(cudaFree(mat2_dev));
    cudaErrCheck(cudaFree(c));
    cudaErrCheck(cudaFree(c_wmma));
    //cudaErrCheck(cudaFree(c_cublas));
    //cudaErrCheck(cudaFree(c_sgemm));

    free(matrix1);
    free(matrix2);
    free(mat1_fp16);
    free(mat2_fp16);
    free(mat1_fp32);
    free(mat2_fp32);
    free(c_host_wmma);

    //free(c_host_cublas);
    //free(c_host_sgemm);

    clock_gettime(CLOCK_REALTIME, &tcu_end);
    double tcu_fill = (fill_end.tv_sec -  fill_start.tv_sec)* BILLION + fill_end.tv_nsec - fill_start.tv_nsec;
    double tcu_convert = (convert_end.tv_sec -  convert_start.tv_sec)* BILLION + convert_end.tv_nsec - convert_start.tv_nsec;
    double tcu_elapse = (tcu_end.tv_sec -  tcu_start.tv_sec)* BILLION + tcu_end.tv_nsec - tcu_start.tv_nsec;
    double init_elapse = (init_end.tv_sec -  init_start.tv_sec)* BILLION + init_end.tv_nsec - init_start.tv_nsec;
    double cuMemcpy_elapse = (cuMemcpy_end.tv_sec -  cuMemcpy_start.tv_sec)* BILLION + cuMemcpy_end.tv_nsec - cuMemcpy_start.tv_nsec;
    //double chkRes_elapse = (chkRes_end.tv_sec -  chkRes_start.tv_sec)* BILLION + chkRes_end.tv_nsec - chkRes_start.tv_nsec;
    
    printf("Time to initialize: %lf\n", init_elapse/(1000*1000));
    printf("Time to fill matrices: %lf\n", tcu_fill/(1000*1000));
    printf("Time to convert data type: %lf\n", tcu_convert/(1000*1000));
    printf("Time for cudaMemcpy: %lf\n", cuMemcpy_elapse/(1000*1000));
    //printf("Time to check result: %lf\n", chkRes_elapse/(1000*1000));
    printf("NVIDIA lib overall MatMul_Agg Time: %lf\n", tcu_elapse/(1000*1000));

    return 0; // non-void function

}
