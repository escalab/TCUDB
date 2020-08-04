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
#include <math.h>
#ifdef DEBUG
#include "../include/cuPrintf.cu"
#include "../include/cuPrintf.cuh"
#endif

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

__host__ void static verify_result(float * matrix, int height, int width) {
    int i;
    for (i = 0; i < height*width; i++) {
    //for (i = width*15; i < width*16; i++) {
        //printf("%d\t", matrix[i]);
        printf("%.0f\t", matrix[i]);
        if ((i+1) % width == 0)
            printf("\n\n");  
    }

}

/* Transpose the matrix on CPU */
__host__ void transpose(float *in, float *out, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            out[j*row+i] = in[i*col+j];
        }
    }
}

__host__ int sum_matrix(float *mat, int height, int width) {
    int i, sum = 0;
    for (i = 0; i < height*width; i++)
        sum += mat[i];
    return sum;
}

/* Find the nearest multiple of N, check the width of matrix or tupleNum to form the matrices for MM */
__host__ int nearestMultipleN(int inNum, int n) {
    if (!n)
        return inNum;
    int remain = inNum % n;
    if (!remain)
        return inNum;
    return (inNum + n - remain);
}

/*
 *  If the query only need to return the count of join result.
 *  result t = mat1*mat.T
 *  count = t.size - sum(t) -- how many non-zero in t
 */
__host__ void static tcu_match(struct joinNode *jNode, int width,
         float *A, float *B, int attr_type1, int attr_type2) {

    int A_tupleNum = jNode->leftTable->tupleNum;
    int B_tupleNum = jNode->rightTable->tupleNum;

    // create first matrix
    int i, colContIdx; // index of column content
    colContIdx = 0;
    for (i = 0; i < A_tupleNum; i++) {
        int *colCont;   // content of column
        colCont = (int*)(&jNode->leftTable->content[jNode->leftKeyIndex][colContIdx]);
        colContIdx += attr_type1; // 4 because of INT type
        A[i*width+(*colCont)] = 1; // mark as 1 if appear in the matrix
    }

    // create second matrix
    colContIdx = 0;
    for (i = 0; i < B_tupleNum; i++) {
        int *colCont;
        colCont = (int*)(&jNode->rightTable->content[jNode->rightKeyIndex][colContIdx]);
        colContIdx += attr_type1;
        B[i*width+(*colCont)] = 1;
    }

    // transpose second matrix
    //transpose(B, B_T, B_tupleNum, width);

    // perform MM & return count on device
}

/* Map the table entires into matrix for tensor core to use 
 * Assum both matrix have the same dimension and the value in INT type for now, e.g., both 16x16 dim
 * To support multiple types, this function need to be modified
 */

// micro benchmark for simple matrix multiplication query
__host__ void static micro_mm(struct joinNode *jNode, float * matrix1, float * matrix2, int width,
        int attr_num1, int attr_num2, int attr_type1, int attr_type2) {
    int *mat1_i, *mat1_j, *mat1_val; // row index, col index, value
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
    // prepare two matrices (1-D array format) for WMMA
    int m;
    for (m = 0; m < leftTupleNum; m++) {
        matrix1[width * mat1_i[m] + mat1_j[m]] = (float)mat1_val[m];
        //printf("%.2f\t", matrix1[width * mat1_i[m] + mat1_j[m]]);
    }
    //printf("\n");

    //printf("rightTupleNum: %d\n", rightTupleNum);
    for (m = 0; m < rightTupleNum; m++) {
        //printf("mat2 val: %d\tn: %d\n", mat2_val[n], n);
        matrix2[width * mat2_i[m] + mat2_j[m]] = (float)mat2_val[m];
        //printf("%.2f\t", matrix2[width * mat2_i[m] + mat2_j[m]]);
    }
    //printf("\n");

    free(mat1_i);
    free(mat1_j);
    free(mat1_val);
    free(mat2_i);
    free(mat2_j);
    free(mat2_val);
}

/* Print matrix content in device memory */
#ifdef DEBUG
__global__ void static verify_gpuResult(half * matrix, int width) {
    int i;
    for (i = 0; i < width*width; i++) {
        //printf("%d\t", matrix[i]);
        cuPrintf("%.1f\t", __half2float(matrix[i]));
        if ((i+1) % width == 0)
          cuPrintf("\n");  
    }

}
#endif

/* Convert input data from half to float type */
__global__ void static convertFp16ToFp32(float *out, half *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
        /*
        if (out[idx])  {
            cuPrintf("idx: %d\t%.1f\n", idx, __half2float(out[idx])); 
        }
        */
        //out[idx] = in[idx];
    }
}

/* Convert matrix from int to half type */
__global__ void static convertFp32ToFp16(half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
        /*
        if (out[idx])  {
            cuPrintf("idx: %d\t%.1f\n", idx, __half2float(out[idx])); 
        }
        */
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

/* Check whether the tupleNum is multiple of 16 because the WMMA requires the width of matrix be multiple of 16 */
__host__ int static findMatWidth(int tupleNum) {
    if (tupleNum <= 256)
        return 16;
    else {
        int tmp = ceil(sqrt(tupleNum));
        return (int)(ceil(tmp/(float)16)*16);
    }
}

__device__ static float getVal(char **content, struct mathExp exp, int pos) {
    float res;
    if (exp.opType == CONS)
        res = exp.opValue;
    else {
        int index = exp.opValue;
        res = ((int *)(content[index]))[pos];
    }

    return res;
}

// since WMMA perform C = alpha*A*B+beta*C, here we just fill operator MULTIPLY
__device__ static void fillMathExp(char **content, struct mathExp exp, int pos, float * A, float * B) {

    if (exp.op == MULTIPLY) {
        if (((struct mathExp*)exp.exp)[0].op == NOOP)
            A[pos] = getVal(content, ((struct mathExp*)exp.exp)[0], pos);
        if (((struct mathExp*)exp.exp)[1].op == NOOP)
            B[pos] = getVal(content, ((struct mathExp*)exp.exp)[1], pos);
    }
        
    return;
}

/* set the first column of the matrix to be 1.0 */
__host__ static void set_mask(float *mask, int height, int width) {
    for (int i = 0; i < height*width; i+=width) {
        mask[i] = 1.0;
    }
}

/* set the first row of the matrix to be 1.0 */
__host__ static void set_mask2(float *mask, int height, int width) {
    
    for (int i = 0; i < width; i++) {
        mask[i] = 1.0;
    }
    
}

__global__ static void agg_cal_cons(char ** content, int colNum, struct groupByExp* exp, long tupleNum, float * A, float * B) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=index;i<tupleNum;i+=stride){
        for(int j=0;j<colNum;j++){
            int func = exp[j].func;
            // for now, we only care about SUM
            if (func == SUM) {
                // 1. fill data into two matrices
                //transform_data(content, exp[j].exp, i, A, B);
                fillMathExp(content, exp[j].exp, i, A, B);
                // TODO: how to maintain the order for threads
                // maybe the order does not important if we can get relative ranking

                // 2. copy data into device using cudaMemcpy (if directly assign in device memory, can avoid this step)

            } else if (func == AVG) {
                // not the main point now
            }
        }
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
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
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
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);

        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_row_major);
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
struct tableNode * tcuJoin(struct joinNode *jNode, struct statistic *pp, int *matrix_dim){
#ifdef DEBUG
    cudaPrintfInit();
#endif
    int leftTupleNum = jNode->leftTable->tupleNum;
    int rightTupleNum = jNode->rightTable->tupleNum;

    // parse user input dimension from command line
    int MATRIX_M, MATRIX_N, MATRIX_K;
    MATRIX_M = nearestMultipleN(leftTupleNum, 16);
    MATRIX_N = nearestMultipleN(rightTupleNum, 16);
    // TODO: for CUBLAS_HALF, MATRIX_K should be other values
    MATRIX_K = *matrix_dim; // user input, matrix width
#ifdef DEBUG
    printf("left  tuple #: %d\n", leftTupleNum);
    printf("right tuple #: %d\n", rightTupleNum);
    printf("MATRIX_M: %d\n", MATRIX_M);
    printf("MATRIX_N: %d\n", MATRIX_N);
    printf("MATRIX_K: %d\n", MATRIX_K);
#endif

    struct timespec tcu_start, tcu_end;
    struct timespec init_start, init_end;
    struct timespec fill_start, fill_end;
    struct timespec convert_start, convert_end;
    struct timespec cuMemcpy_start, cuMemcpy_end;
    clock_gettime(CLOCK_REALTIME, &tcu_start);
    clock_gettime(CLOCK_REALTIME, &init_start);

#ifdef WMMA_INT4
    unsigned char *h_int_A, *h_int_B; // host int4 array
    unsigned char *d_int_A, *d_int_B; // device int4 array
    unsigned char *c_int_wmma, *c_host_int_wmma;
    int alpha = 1;
    int beta = 0;
#else
    float *h_fp32_A, *h_fp32_B; // host float32 array
    float *d_fp32_A, *d_fp32_B; // device float32 array
    half *d_fp16_A, *d_fp16_B;
    float *c_wmma, *c_wmma_sum1, *c_wmma_sum2, *c_host_wmma;
    float *d_fp32_mask, *h_fp32_mask;
    float *d_fp32_mask2, *h_fp32_mask2;
    half *d_fp16_mask;
    half *d_fp16_mask2;
    float alpha = 1.0f;
    float beta = 0.0f;
#endif

#ifdef CUBLAS_HALF
    float *c_cublas, *c_host_cublas;
    curandGenerator_t gen;
    // use tensor core or cublas
    cublasHandle_t cublasHandle; // cublas tcu
    cudaEvent_t startcublasEX;
    cudaEvent_t stopcublasEX;

    cublasErrCheck(cublasCreate(&cublasHandle));
    // enable tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
#elif CUBLAS
    float *c_sgemm, *c_host_sgemm;
    curandGenerator_t gen;
    cublasHandle_t cublasHandle_default; // cublas default
    cudaEvent_t startcublas; // for sgemm (FP32)
    cudaEvent_t stopcublas;

    cublasErrCheck(cublasCreate(&cublasHandle_default));
    cublasErrCheck(cublasSetMathMode(cublasHandle_default, CUBLAS_DEFAULT_MATH));
#else
    cudaEvent_t startWMMA;
    cudaEvent_t stopWMMA;
    cudaErrCheck(cudaEventCreate(&startWMMA));
    cudaErrCheck(cudaEventCreate(&stopWMMA)); 

    dim3 gridDim;
    dim3 blockDim;
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
#endif

#ifdef WMMA_INT4
    c_host_int_wmma = (int*)calloc(MATRIX_M*MATRIX_N, sizeof(int));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_int_wmma, MATRIX_M * MATRIX_N * sizeof(int)));
#elif CUBLAS_HALF
    c_host_cublas = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
#elif CUBLAS
    c_host_sgemm =  (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_sgemm, MATRIX_M * MATRIX_N * sizeof(float)));
#else
    c_host_wmma = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_wmma_sum1, MATRIX_M * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_wmma_sum2, MATRIX_M * MATRIX_N * sizeof(float)));
#endif
   
#ifdef WMMA_INT4
    h_int_A = (unsigned char*)calloc(MATRIX_M*MATRIX_K, sizeof(unsigned char));
    h_int_B = (unsigned char*)calloc(MATRIX_K*MATRIX_N, sizeof(unsigned char));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_int_A, MATRIX_M * MATRIX_K * sizeof(unsigned char)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_int_B, MATRIX_K * MATRIX_N * sizeof(unsigned char)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_int_wmma, MATRIX_M * MATRIX_N * sizeof(int)));

#else
    h_fp32_A = (float*)calloc(MATRIX_M*MATRIX_K, sizeof(float));
    h_fp32_B = (float*)calloc(MATRIX_K*MATRIX_N, sizeof(float));
    h_fp32_mask = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    h_fp32_mask2 = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_A, MATRIX_M * MATRIX_K * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_B, MATRIX_K * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_mask, MATRIX_M * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_mask2, MATRIX_M * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_fp16_A, MATRIX_M * MATRIX_K * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_fp16_B, MATRIX_K * MATRIX_N * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_mask, MATRIX_M * MATRIX_N * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_mask2, MATRIX_M * MATRIX_N * sizeof(half)));

    set_mask(h_fp32_mask, MATRIX_M, MATRIX_N);
    set_mask2(h_fp32_mask2, MATRIX_M, MATRIX_N);

    //printf("mask2:\n");
    //verify_result(h_fp32_mask2, MATRIX_M, MATRIX_K);
#endif    

    clock_gettime(CLOCK_REALTIME, &init_end);

    clock_gettime(CLOCK_REALTIME, &fill_start); 
#ifdef WMMA_INT4

#elif WMMA_HALF    
    tcu_match(jNode, MATRIX_K, h_fp32_A, h_fp32_B, jNode->leftTable->attrType[0], jNode->rightTable->attrType[0]);
    /*    
    micro_mm(jNode, h_fp32_A, h_fp32_B, MATRIX_M,
            jNode->leftTable->totalAttr, jNode->rightTable->totalAttr, jNode->leftTable->attrType[0], jNode->rightTable->attrType[0]);

    */
#else
#endif
    clock_gettime(CLOCK_REALTIME, &fill_end);

    clock_gettime(CLOCK_REALTIME, &cuMemcpy_start);
#ifdef WMMA_INT4
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_int_A, h_int_A, sizeof(unsigned char) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_int_B, h_int_B, sizeof(unsigned char) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));
#else
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_A, h_fp32_A, sizeof(float) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_B, h_fp32_B, sizeof(float) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_mask, h_fp32_mask, sizeof(float) * MATRIX_M * MATRIX_N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_mask2, h_fp32_mask2, sizeof(float) * MATRIX_M * MATRIX_N, cudaMemcpyHostToDevice));
#endif
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);

    clock_gettime(CLOCK_REALTIME, &convert_start); // if float->half
#ifdef WMMA_INT4
#else
    convertFp32ToFp16<<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_fp16_A, d_fp32_A, MATRIX_M * MATRIX_K);
    convertFp32ToFp16<<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (d_fp16_B, d_fp32_B, MATRIX_K * MATRIX_N);
    convertFp32ToFp16<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp16_mask, d_fp32_mask, MATRIX_M * MATRIX_N);
    convertFp32ToFp16<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp16_mask2, d_fp32_mask2, MATRIX_M * MATRIX_N);
#endif
    clock_gettime(CLOCK_REALTIME, &convert_end);
#ifdef WMMA_HALF 
    printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

    printf("Running with wmma...\n");
    cudaErrCheck(cudaEventRecord(startWMMA));
    wmma_example <<< gridDim, blockDim >>> (d_fp16_A, d_fp16_B, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta); 
    
    half *c_wmma_reduction1, *c_wmma_reduction2;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&c_wmma_reduction1, MATRIX_M * MATRIX_N * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&c_wmma_reduction2, MATRIX_M * MATRIX_N * sizeof(half)));
    convertFp32ToFp16<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_wmma_reduction1, c_wmma, MATRIX_M * MATRIX_N);

    wmma_example <<< gridDim, blockDim >>> (d_fp16_mask2, c_wmma_reduction1, c_wmma_sum1, MATRIX_M, MATRIX_N, MATRIX_M, alpha, beta); 
    convertFp32ToFp16<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_wmma_reduction2, c_wmma_sum1, MATRIX_M * MATRIX_N);

    wmma_example <<< gridDim, blockDim >>> (d_fp16_mask2, c_wmma_reduction2, c_wmma_sum2, MATRIX_M, MATRIX_N, MATRIX_M, alpha, beta); 
    
    cudaErrCheck(cudaEventRecord(stopWMMA));
#elif CUBLAS_HALF
    printf("Running with cuBLAS on TCUs...\n");
    cudaErrCheck(cudaEventRecord(startcublasEX));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha,
                d_fp16_A, CUDA_R_16F, MATRIX_M,
                d_fp16_B, CUDA_R_16F, MATRIX_N,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_K,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
    cudaErrCheck(cudaEventRecord(stopcublasEX));
#elif CUBLAS
    printf("Running with sgemm...\n");
    cudaErrCheck(cudaEventRecord(startcublas));
    cublasSgemm(cublasHandle_default, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, d_fp32_A, MATRIX_M, d_fp32_B, MATRIX_N, &beta, c_sgemm, MATRIX_K);
    cudaErrCheck(cudaEventRecord(stopcublas));
#endif    

#ifdef WMMA_HALF
    struct timespec tmp_start, tmp_end;
    clock_gettime(CLOCK_REALTIME, &tmp_start);
    cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma_sum2, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

    //printf("c_host_wmma:\n");
    //verify_result(c_host_wmma, MATRIX_M, MATRIX_N);

    printf("Number of join results (MM reduction): %.0f\n", c_host_wmma[0]);
    printf("Number of join results (CPU count): %d\n", sum_matrix(c_host_wmma, MATRIX_M, MATRIX_N));
    clock_gettime(CLOCK_REALTIME, &tmp_end);
#elif CUBLAS_HALF
    cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
#elif CUBLAS
    cudaErrCheck(cudaMemcpy(c_host_sgemm, c_sgemm, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
#endif

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
#ifdef CUBLAS_HALF
    float cublasEXTime;

    cudaErrCheck(cudaEventSynchronize(stopcublasEX));
    cudaErrCheck(cudaEventElapsedTime(&cublasEXTime, startcublasEX, stopcublasEX));
    printf("cublasEX tensor cores (FP16) took %fms\n", cublasEXTime);

    cudaErrCheck(cudaEventDestroy(startcublasEX));
    cudaErrCheck(cudaEventDestroy(stopcublasEX));
    free(c_host_cublas);
    cudaErrCheck(cudaFree(c_cublas));
#elif CUBLAS
    float cublasTime;

    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    printf("cublas sgemm (FP32) took %fms\n", cublasTime);

    cudaErrCheck(cudaEventDestroy(startcublas));
    cudaErrCheck(cudaEventDestroy(stopcublas));
    free(c_host_sgemm);
    cudaErrCheck(cudaFree(c_sgemm));
#else
    float wmmaTime;

    cudaErrCheck(cudaEventSynchronize(stopWMMA));
    cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    printf("wmma took %fms\n", wmmaTime);

    // free those data structures
    cudaErrCheck(cudaEventDestroy(startWMMA));
    cudaErrCheck(cudaEventDestroy(stopWMMA));
#endif

#ifdef WMMA_HALF
    free(c_host_wmma);
    cudaErrCheck(cudaFree(c_wmma));
#endif

#ifdef WMMA_INT4
    free(h_int_A);
    free(h_int_B);
    free(c_host_int_wmma);
    cudaErrCheck(cudaFree(d_int_A));
    cudaErrCheck(cudaFree(d_int_B));
    cudaErrCheck(cudaFree(c_int_wmma));
#else
    free(h_fp32_A);
    free(h_fp32_B);
    free(h_fp32_mask2);
    cudaErrCheck(cudaFree(d_fp32_A));
    cudaErrCheck(cudaFree(d_fp16_A));
    cudaErrCheck(cudaFree(d_fp32_B));
    cudaErrCheck(cudaFree(d_fp16_B));
    cudaErrCheck(cudaFree(d_fp32_mask2));
    cudaErrCheck(cudaFree(d_fp16_mask2));
#endif
    clock_gettime(CLOCK_REALTIME, &tcu_end);
    double tcu_fill = (fill_end.tv_sec -  fill_start.tv_sec)* BILLION + fill_end.tv_nsec - fill_start.tv_nsec;
    double tcu_convert = (convert_end.tv_sec -  convert_start.tv_sec)* BILLION + convert_end.tv_nsec - convert_start.tv_nsec;
    double tcu_elapse = (tcu_end.tv_sec -  tcu_start.tv_sec)* BILLION + tcu_end.tv_nsec - tcu_start.tv_nsec;
    double init_elapse = (init_end.tv_sec -  init_start.tv_sec)* BILLION + init_end.tv_nsec - init_start.tv_nsec;
    double cuMemcpy_elapse = (cuMemcpy_end.tv_sec -  cuMemcpy_start.tv_sec)* BILLION + cuMemcpy_end.tv_nsec - cuMemcpy_start.tv_nsec;
    double tmp_elapse = (tmp_end.tv_sec -  tmp_start.tv_sec)* BILLION + tmp_end.tv_nsec - tmp_start.tv_nsec;
    
    printf("Initialization: %lf(ms)\n", init_elapse/(1000*1000));
    printf("Matrices filling: %lf(ms)\n", tcu_fill/(1000*1000));
    printf("Data type convertion: %lf(ms)\n", tcu_convert/(1000*1000));
    printf("cudaMemcpy: %lf(ms)\n", cuMemcpy_elapse/(1000*1000));
    printf("MMA end-to-end: %lf(ms)\n", tcu_elapse/(1000*1000));
    printf("Result verification: %lf(ms)\n", tmp_elapse/(1000*1000));
#ifdef DEBUG
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif
    return 0; // non-void function

}
