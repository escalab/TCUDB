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
//#ifdef DEBUG
//#include "../include/cuPrintf.cu"
//#include "../include/cuPrintf.cuh"
//#endif

using namespace nvcuda;

#define MAX_THREADS 1024 // For NVIDIA Turing Architecture

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#if defined(CUBLAS) || defined(CUBLAS_HALF)
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}
#endif

void* cublasCreateThread(void *x)
{
    cublasHandle_t* cublasHandle = (cublasHandle_t *)x;
    cublasErrCheck(cublasCreate(cublasHandle));
    cublasErrCheck(cublasSetMathMode(*cublasHandle, CUBLAS_TENSOR_OP_MATH));
    return NULL;
}

__global__ static void count_op(float *red_sum, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i > length) return;
    if (red_sum[i] != 0)
        return;
}

__global__ static void gb_count(float *red_sum, int length, int *cnt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= length) return;
    if (red_sum[i] != 0)
        atomicAdd(cnt, 1);

}

/* Fill the actual float value for PageRank calculation. 
   Pagerank.ranking and Outdegree.degree */
__global__ void pagerank(char *columnIdx, char *columnVal, int matWidth, half *mat, size_t tupleNum, int attrTypeSize, int attrType, float pagerank_cons) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < tupleNum) {
        int stripe = i * attrTypeSize;
        int *id    = (int*)&columnIdx[stripe];

        if (attrType == INT) {
            int *val = (int*)&columnVal[stripe];
            mat[i*matWidth + (*id)] = __float2half((float)1/(*val));
            //cuPrintf("mat[%d]\t%d\n", i*matWidth + (*id), *val);
        } else if (attrType == FLOAT) {
            float *val   = (float*)&columnVal[stripe];
            
            mat[i*matWidth + (*id)] = __float2half((*val)*pagerank_cons);
            //cuPrintf("mat[%d]\t%.8f\n", i*matWidth + (*id), *val);
        }
    }
}

/* 
 *  Fill 1.0 on the index of unique value in the matrix;
 *  fill 0.0, otherwise. 
 */
__global__ void static gpu_fill(char *column, int matWidth, half *matA, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    //int value = (int)column[index]; // char -> int will lose 3 bytes
    int *value   = (int*)&column[index];
    matA[i*matWidth + (*value)] = __float2half(1.0f);
}

/* Fill matrix with data value. */
__global__ void static gpu_fill_data(char *join_column, char *data_column, int matWidth_k, half *matA, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];
    int *data_value = (int*)&data_column[index];
    matA[i * matWidth_k + (*join_value)] = __float2half((float)(*data_value));
}

__global__ void static gpu_fill_gb(char *join_column, char *data_column, int matWidth_k, half *matA, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];
    int *data_value = (int*)&data_column[index];
    matA[(*data_value) * matWidth_k + (*join_value)] = __float2half(1.0f);
}

__global__ void static gpu_fill_data_transpose(char *join_column, char *data_column, int matWidth_n, half *matB, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];
    int *data_value = (int*)&data_column[index];
    matB[(*join_value) * matWidth_n + i] = __float2half((float)(*data_value));
}

/* Fill matrix with ones according to groupBy column in transpose format. */
__global__ void static gpu_fill_gb_transpose(char *join_column, char *data_column, int matWidth_n, half *matB, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];
    int *data_value = (int*)&data_column[index];
    matB[(*join_value) * matWidth_n + (*data_value)] = __float2half(1.0f);
}

/*
 * Fill ones matrix in transpose matrix format.
 */
__global__ void static gpu_fill_transpose(char *column, int matWidth, half *matB, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *value   = (int*)&column[index];
    int pos = (*value)*tupleNum+i;
    matB[pos] = __float2half(1.0f);
}

/* Fill matrix in dense format for matrix multiplication */
__global__ void static microbenchmark(char *mat_i, char *mat_j, char *mat_val, int matWidth, half *mat, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= tupleNum) return;

    int index = i * attrType;
    int *row  = (int*)&mat_i[index]; 
    int *col  = (int*)&mat_j[index]; 
    int *val  = (int*)&mat_val[index];
    mat[(*row)*matWidth+(*col)] = __int2half_rn(*val);
}

__global__ void static microbenchmark_transpose(char *mat_i, char *mat_j, char *mat_val, int matWidth, half *mat, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= tupleNum) return;

    int index = i * attrType;
    int *row  = (int*)&mat_i[index]; 
    int *col  = (int*)&mat_j[index]; 
    int *val  = (int*)&mat_val[index];
    mat[(*col)*matWidth+(*row)] = __int2half_rn(*val);
}

__global__ void static outdegree_fill(char *column_val, half *mat, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= tupleNum) return;

    int index = i * attrType;
    //int *colIndex   = (int*)&column_idx[index];
    int *val        = (int*)&column_val[index];
    //printf("idx: %d\tval: %d\n", i*matWidth + (*colIndex), (*val));
    mat[(*val)] = __hadd(mat[(*val)], __int2half_rn(1));
}

#ifdef CUBLAS_HALF
__global__ void gpu_transpose(half *odata, const half *idata, int row, int col) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index % col;
    int y = index / col;

    if (x < col && y < row) {
        odata[x*row + y] = idata[y*col + x];
    }
}
#elif CUBLAS
__global__ void gpu_transpose(float *odata, const float *idata, int row, int col) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index % col;
    int y = index / col;

    if (x < col && y < row) {
        odata[x*row + y] = idata[y*col + x];
    }
}
#endif

__global__ void static pageRankAdd(float *mat, int n, float pageRankAlpha, int numNodes) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        if (mat[idx] > 1e-6)
        //if (__hgt(mat[idx], __float2half(1e-6))) // precision loss
            mat[idx] += (float)(1-pageRankAlpha)/numNodes;
            //mat[idx] += __float2half((1-pageRankAlpha)/numNodes);
    }
}

/* Convert input data from half to float type */
__global__ void static convertFp16ToFp32(float *out, half *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

/* Convert input data from half to float type */
__global__ void static convertFp32ToFp16(half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

/* Convert input data from char to half type */
__global__ void static convertCharToFp16(half *out, char *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __int2half_rn((int)in[idx]);
    }
}

__global__ void groupByCount(float *data, int n, int *gbCount) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] > 0.000001) {
        //if (data[idx] > 0.001) {
            atomicAdd(gbCount, 1);
        }
    }
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

__host__ static void setVector(float *vec, int n) {
    for (int i = 0; i < n; i++)
        vec[i] = 1.0;
}

__host__ static void setRed(short *red, int n) {
    for (int i = 0; i < n; i++)
        red[i] = (short)1;
}

/* Get column index from aggregate function for later data copy. */
__host__ static void getValIndex(struct joinNode *jNode, struct groupByNode *gb, int *lValIndex, int *rValIndex, int &lgbIndex, int &rgbIndex, int &dataColIndex) {

    for (int i = 0; i < jNode->leftOutputAttrNum; i++) {
        for (int j = 0; j < gb->numFuncExpCol; j++) {
            if (jNode->leftPos[i] == gb->funcExpColIndex[j]) {
                lValIndex[i] = jNode->leftOutputIndex[i];

                if (dataColIndex == -1)
                    dataColIndex = jNode->leftOutputIndex[i];
            }
            if (jNode->leftPos[i] == gb->groupByIndex[0]) {
                lgbIndex = 1;
            }
        }
    } 
    
    for (int i = 0; i < jNode->rightOutputAttrNum; i++) {
        for (int j = 0; j < gb->numFuncExpCol; j++) {
            if (jNode->rightPos[i] == gb->funcExpColIndex[j]) {
                rValIndex[i] = jNode->rightOutputIndex[i];

                if (dataColIndex == -1)
                    dataColIndex = jNode->rightOutputIndex[i];
            }
            if (jNode->rightPos[i] == gb->groupByIndex[0]) {
                rgbIndex = 1;
            }
        }
    } 
}

/* Match the first groupBy attribute, return 0 (left), 1 (right)*/
__host__ static int getGbLeftRight(struct joinNode *jNode, struct groupByNode *gb, int &gbConstant, int &gbLeftRight) {
    if (gbConstant == 1) return -1;
    
    for (int i = 0; i < jNode->leftOutputAttrNum; i++) {
        if (jNode->leftPos[i] == gb->groupByIndex[0]) {
            return 0;
        }
    } 
    
    for (int i = 0; i < jNode->rightOutputAttrNum; i++) {
        if (jNode->rightPos[i] == gb->groupByIndex[0]) {
            return 1;
        }
    } 
    return -1;
}

/* Mimic the max() in relational database. */
__host__ int getMaxVal(char *column, size_t tupleNum, int attrType) {
    int localMax = 0;

    for (int i = 0; i < tupleNum; i++) {
        int *val = (int*)&column[i*attrType];
        if (localMax < *val) {
            localMax = *val;
        }
    }
    return localMax;
}

/* Need to copy values to device */
__global__ void getMaxValGPU(char *column, size_t tupleNum, int attrType, int *maxVal) {
    __shared__ int sharedMax;

    if (threadIdx.x == 0) {
        sharedMax = 0;
    }
    __syncthreads();

    int localMax = 0;
    for (int i = threadIdx.x; i < tupleNum; i += blockDim.x) {
        int index = i * attrType;
        int *value   = (int*)&column[index];

        if (localMax < abs(*value)) {
            localMax = abs(*value);
        }
    }

    atomicMax(&sharedMax, localMax);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        *maxVal = sharedMax;
    }
}

/*
 * tcuJoinn using NVIDIA's cuBLAS lib to perform matrix multiplication and aggregation.
 *
 * Prerequisites:
 *  1. the data to be joined can be fit into GPU device memory.
 *  2. dimension table is not compressed
 *  3. user know the matrix dimension (#uniq values)
 *  
 * Input:
 *  jNode: contains information about the two joined tables.
 *  pp: records statistics such as kernel execution time
 *  matrix_dim: matrix width (number of unique values)
 *  gb: contains groupby information
 *
 * Output:
 *  Number of join counts and groupBy count if query contains groupBy keyword.
 *
 * Assumptions:
 *
 * 1. Two joined table schemas are the same for the simplicity of query parser.
 * 2. For all demo cases, all column types are INT, only PageRank queries 
 *    contain constant variable such as alpha and number of nodes.
 * 3. To support complex customized queries, code_gen.py modification is required.
 */
struct tableNode * tcuJoin(struct joinNode *jNode, struct statistic *pp, 
        int *matrix_dim, struct groupByNode *gb)
{

    struct timespec tcu_start, tcu_end;
    struct timespec init_start, init_end;
    struct timespec fill_start, fill_end;
    struct timespec maskRED_start, maskRED_end;
    struct timespec pagerankVerify_start, pagerankVerify_end;
    struct timespec cuMemcpy_start, cuMemcpy_end;

    struct tableNode * res = NULL;
    int leftTupleNum = jNode->leftTable->tupleNum;
    int rightTupleNum = jNode->rightTable->tupleNum;
    uint64_t MATRIX_M, MATRIX_N, MATRIX_K; // avoid overflow

    MATRIX_K = *matrix_dim; // user input, matrix width(#unique values)
#ifdef MICRO
    //Note: In our matrix multiplication cases, square matrix(M=N=K) are used
    MATRIX_M = MATRIX_K;
    MATRIX_N = MATRIX_K;
#else
    MATRIX_M = (uint64_t)leftTupleNum;
    MATRIX_N = (uint64_t)rightTupleNum;
#endif // end of MICRO
    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
    long primaryKeySize = jNode->rightTable->attrTotalSize[jNode->rightKeyIndex];
    
    float pageRankAlpha;

    int gbConstant = 0;   // 0: has groupBy, 1: no groupBy keyword
    int gbLeftRight = -1; // 0: gb by left, 1: gb by right
    int gbMatWidth = 0;   // size of dom(gb_column.val)

    if (gb->groupByColNum == 1 && gb->groupByIndex[0] == -1) {
        gbConstant = 1;
    }
    
    // update MATRIX_M or MATRIX_N given groupBy keyword
    if (gbConstant != 1) { // contains groupBy keyword
        char *gb_column;
        // linear scan to find the max value of groupBy column 
        gbLeftRight = getGbLeftRight(jNode, gb, gbConstant, gbLeftRight);
        if (gbLeftRight == 0) {
            gb_column = jNode->leftTable->content[gb->groupByIndex[0]];

            gbMatWidth = getMaxVal(gb_column, jNode->leftTable->tupleNum, jNode->leftOutputAttrType[0]) + 1;
            printf("matA gbMatWidth: %d\n", gbMatWidth);
        } else if (gbLeftRight == 1) {
            gb_column = jNode->rightTable->content[gb->groupByIndex[0]];
            gbMatWidth = getMaxVal(gb_column, jNode->rightTable->tupleNum, jNode->rightOutputAttrType[0]) + 1;
            printf("matB gbMatWidth: %d\n", gbMatWidth);
            // update
            MATRIX_N = gbMatWidth;
        } else {
            printf("No matched column found.\n");
        }
    }

    // TODO: determine which table to copy value column (left/right or both)
    // column index are leftOutputIndex[0] or rightOutputIndex[0]
    printf("numFuncExpCol: %d\n", gb->numFuncExpCol); // determine number of data column to be copied

    int *lValIndex, *rValIndex;
    int dataColIndex = -1;
    int lgbIndex = -1, rgbIndex = -1;
    lValIndex = (int *)malloc(sizeof(int) * jNode->leftOutputAttrNum);
    rValIndex = (int *)malloc(sizeof(int) * jNode->rightOutputAttrNum);
    memset(lValIndex, -1, sizeof(int) * jNode->leftOutputAttrNum);
    memset(rValIndex, -1, sizeof(int) * jNode->rightOutputAttrNum);

    // get data value index from gbNode
    getValIndex(jNode, gb, lValIndex, rValIndex, lgbIndex, rgbIndex, dataColIndex);
    printf("lValIndex[0]: %d\n", lValIndex[0]); // data copy from lValIndex
    printf("rValIndex[0]: %d\n", rValIndex[0]);
    printf("lgbIndex: %d\n", lgbIndex); // data copy from lValIndex
    printf("rgbIndex: %d\n", rgbIndex);
    printf("dataColIndex: %d\n", dataColIndex);


#ifdef PAGERANK
    //printf("func: %d\n", gb->gbExp[0].func);
    //printf("PageRank constant: %.3f\n", ((struct mathExp *)((struct mathExp *) gb->gbExp[0].exp.exp)[0].exp)[0].consValue);
    pageRankAlpha = ((struct mathExp *)((struct mathExp *) gb->gbExp[0].exp.exp)[0].exp)[0].consValue;
    //printf("(1-alpha)/#node: %.6f\n", ((struct mathExp *) gb->gbExp[0].exp.exp)[1].consValue);
#endif
    

//#ifdef DEBUG
    //cudaPrintfInit();
//#endif
    clock_gettime(CLOCK_REALTIME, &tcu_start);
    clock_gettime(CLOCK_REALTIME, &init_start);

/*
    printf("Left Tuple #: %d\n", leftTupleNum);
    printf("Right Tuple #: %d\n", rightTupleNum);
    printf("MATRIX_M: %lu\n", MATRIX_M);
    printf("MATRIX_N: %lu\n", MATRIX_N);
    printf("MATRIX_K: %lu\n", MATRIX_K);
*/
#ifdef PAGERANK
    //printf("PageRank Alpha: %.3f\n", pageRankAlpha);
    //printf("(1-alpha)/#node: %.6f\n", (1-pageRankAlpha)/MATRIX_K);
#endif


#if defined(CUBLAS_HALF) || defined(CUBLAS)
    //struct timespec debug_start, debug_end; // cublasCreate has init overhead
    struct timespec count_start, count_end;
    //struct timespec transpose_start, transpose_end;
#endif

    // read row data from tbl
    char *gpu_fact, *gpu_dim;         // joined column index
    char *gpu_fact_j, *gpu_dim_j;     // another index for dense table
    char *gpu_fact_val, *gpu_dim_val; // value
    char *gpu_ldata, *gpu_rdata;      // data columns of left/right tables
    char *d_redMat;
    half *d_redMatFp16;

    float alpha = 1.0f;
    float beta = 0.0f;
#ifdef CUBLAS_HALF
    half *d_fp16_A, *d_fp16_B, *d_fp16_BT;
//    half *d_fp16_A, *d_fp16_BT;
    float *c_cublas;
    half *c_fp16_cublas;

//    char *gpu_fact, *gpu_dim;         // raw data idx
//    char *gpu_fact_val, *gpu_dim_val; // raw data val
//    float alpha = 1.0f;
//    float beta = 0.0f;
    half alpha_fp16 = __float2half(1.0f);
    half beta_fp16 = __float2half(1.0f);
    float *c_host_cublas;

#ifdef PAGERANK
    //char *factID, *dimID; // use previous gpu_fact/gpu_dim
    char *factVal, *dimVal;
#endif

    struct timespec gbCount_start, gbCount_end;
    // TODO: move this into RED after decouple
    float *h_red, *d_red;
#ifdef RED
//    float *h_red, *d_red;
    float *h_red2, *d_red2;
#endif

    cublasHandle_t cublasHandle;
    cudaEvent_t startcublasEX;
    cudaEvent_t stopcublasEX;

    cudaErrCheck(cudaEventCreate(&startcublasEX));
    cudaErrCheck(cudaEventCreate(&stopcublasEX));
    //clock_gettime(CLOCK_REALTIME, &debug_start);
    cublasErrCheck(cublasCreate(&cublasHandle));
    //clock_gettime(CLOCK_REALTIME, &debug_end);
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
//    clock_gettime(CLOCK_REALTIME, &debug_start);
    //cublasErrCheck(cublasCreate(&cublasHandle));
//    clock_gettime(CLOCK_REALTIME, &debug_end);
    //cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

#elif CUBLAS // SGEMM
    float *h_fp32_A, *h_fp32_B;             // host float32 array
    float *d_fp32_A, *d_fp32_B, *d_fp32_BT; // device float32 array
//    char *gpu_fact, *gpu_dim; // raw data index
//    char *gpu_fact_val, *gpu_dim_val; // raw data val
    float *c_sgemm, *c_host_sgemm;
//    float alpha = 1.0f;
//    float beta = 0.0f;

    cublasHandle_t cublasHandle_default;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;

    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));
    //clock_gettime(CLOCK_REALTIME, &debug_start);
    cublasErrCheck(cublasCreate(&cublasHandle_default));
    //clock_gettime(CLOCK_REALTIME, &debug_end);
    cublasErrCheck(cublasSetMathMode(cublasHandle_default,CUBLAS_DEFAULT_MATH));
#endif

// allocate device memory for inputs
#ifdef CUBLAS_HALF
    
//    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
//    long primaryKeySize = jNode->rightTable->attrTotalSize[jNode->rightKeyIndex];

    //printf("gpu_fact size: %d\tgpu_dim size: %d\n", foreignKeySize, primaryKeySize);
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));

    if (lValIndex[0] != -1 || lgbIndex != -1) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_ldata,foreignKeySize));
        printf("cudaMalloc left_data column\n");
    }

    if (rValIndex[0] != -1 || rgbIndex != -1) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_rdata,primaryKeySize));
        printf("cudaMalloc right_data column\n");
    }
#ifdef MICRO
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact_j,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim_j,primaryKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact_val,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim_val,primaryKeySize));
#endif

#ifdef PAGERANK // only for pagerank dataset
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&factVal,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&dimVal,primaryKeySize));
#endif

#ifdef OUTDEGREE
    // TODO:create square matrix for PageRank Q1 output
    // any row in the resulting matrix is the answer
    // B mat filling by counting src node -- B should be MATRIX_N * 1
    c_host_cublas = (float*)calloc(MATRIX_M*MATRIX_M, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_cublas, (uint64_t)MATRIX_M * (uint64_t)MATRIX_M * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_A, (uint64_t)MATRIX_M * 1 * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_fp16_A, 1, (uint64_t)MATRIX_M * 1 * sizeof(half)));
    // same dimension as A mat
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_B, (uint64_t)MATRIX_M * 1 * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_fp16_B, 0, (uint64_t)MATRIX_M * 1 * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact_val,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim_val,primaryKeySize));

#else
    c_host_cublas = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    //TODO: seems need to move cudaMalloc into if-condition to dynamically adjust size
    printf("cudaMalloc here\n");
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_cublas,(uint64_t)MATRIX_M*(uint64_t)MATRIX_N*sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_A,(uint64_t)MATRIX_M*(uint64_t)MATRIX_K*sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_B,(uint64_t)MATRIX_N*(uint64_t)MATRIX_K*sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_BT,(uint64_t)MATRIX_K*(uint64_t)MATRIX_N*sizeof(half)));
#endif //end of OUTDEGREE

    // TODO: move this into RED after decouple
    h_red = (float*)calloc(MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_red, MATRIX_N * sizeof(float)));
#ifdef RED
//    h_red = (float*)calloc(MATRIX_N, sizeof(float));
//    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_red, MATRIX_N * sizeof(float)));
    h_red2 = (float*)calloc(MATRIX_M, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_red2, MATRIX_M * sizeof(float)));
#endif

#elif CUBLAS

#ifdef MICRO
    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
    long primaryKeySize = sizeof(int) * jNode->rightTable->tupleNum;

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact_j,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim_j,primaryKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact_val,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim_val,primaryKeySize));
#endif // MICRO

    h_fp32_A =     (float*)calloc(MATRIX_M*MATRIX_K, sizeof(float));
    h_fp32_B =     (float*)calloc(MATRIX_N*MATRIX_K, sizeof(float));
    c_host_sgemm = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_A, MATRIX_M * MATRIX_K * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_B, MATRIX_N * MATRIX_K * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_BT, MATRIX_K * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_sgemm, MATRIX_M * MATRIX_N * sizeof(float)));
#endif // end of initialization
    clock_gettime(CLOCK_REALTIME, &init_end);

#ifdef CUBLAS_HALF
// call different matrix filling methods according to dataset
#ifdef MICRO
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_start);

    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));
    // ystree.py gen_column_index generates index for select_list first
    // joined attr with -1 index which means the last index
    // other attr indices follow the sequence without certain getter function to access
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact_j,jNode->leftTable->content[0], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim_j,jNode->rightTable->content[0], primaryKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact_val,jNode->leftTable->content[1], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim_val,jNode->rightTable->content[1], primaryKeySize,cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);

    clock_gettime(CLOCK_REALTIME, &fill_start); 
    microbenchmark<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
            gpu_fact_j,
            gpu_fact_val,
            MATRIX_K,
            d_fp16_A,
            leftTupleNum,
            jNode->leftTable->attrType[jNode->leftKeyIndex]);
    cudaErrCheck(cudaFree(gpu_fact));
    cudaErrCheck(cudaFree(gpu_fact_j));
    cudaErrCheck(cudaFree(gpu_fact_val));
    microbenchmark_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_dim,
            gpu_dim_j,
            gpu_dim_val,
            MATRIX_K,
            d_fp16_BT,
            rightTupleNum,
            jNode->rightTable->attrType[jNode->rightKeyIndex]);
    cudaErrCheck(cudaFree(gpu_dim));
    cudaErrCheck(cudaFree(gpu_dim_j));
    cudaErrCheck(cudaFree(gpu_dim_val));

    clock_gettime(CLOCK_REALTIME, &fill_end); 
#elif OUTDEGREE // PageRank Q1

    clock_gettime(CLOCK_REALTIME, &cuMemcpy_start);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact_val,jNode->leftTable->content[1], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim_val,jNode->rightTable->content[1], primaryKeySize,cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);

    clock_gettime(CLOCK_REALTIME, &fill_start); 
    outdegree_fill<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact_val,
            d_fp16_A,
            leftTupleNum,
            jNode->leftTable->attrType[jNode->leftKeyIndex]);
    outdegree_fill<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_dim_val,
            d_fp16_B,
            rightTupleNum,
            jNode->rightTable->attrType[jNode->rightKeyIndex]);
    clock_gettime(CLOCK_REALTIME, &fill_end); 
// end of OUTDEGREE to cudaMemcpy and filling matrix

#else //  MM for join count

    // cudaMemcpyHostToDevice raw data->char *column
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_start);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));
    if (lValIndex[0] != -1 || lgbIndex != -1) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_ldata,jNode->leftTable->content[dataColIndex], foreignKeySize,cudaMemcpyHostToDevice));
        printf("cudaMemcpy gpu_ldata\n");
    }
    if (rValIndex[0] != -1 || rgbIndex != -1) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_rdata,jNode->rightTable->content[dataColIndex], primaryKeySize,cudaMemcpyHostToDevice));
        printf("cudaMemcpy gpu_rdata\n");
    }

    /*for (int i = 0; i < 5; i ++) {
        int *value   = (int*)&jNode->leftTable->content[0][i*4];
        printf("%d\n", *value);
    }*/


#ifdef PAGERANK  // pagerank requires additional float value instead of filling 0/1
    int factCol = jNode->leftOutputIndex[jNode->leftOutputAttrNum-1];
    int dimCol = jNode->rightOutputIndex[jNode->rightOutputAttrNum-1];
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(factVal,jNode->leftTable->content[factCol], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(dimVal,jNode->rightTable->content[dimCol], primaryKeySize,cudaMemcpyHostToDevice));
#endif // end of PageRank
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);

    clock_gettime(CLOCK_REALTIME, &fill_start);  // filling time (except for MICRO, OUTDEGREE)
#ifdef PAGERANK // specifically design for PageRank

    if (gb->gbExp[0].func == SUM) { // 20, defined in common.h
        pagerank<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
                factVal,
                MATRIX_K,
                d_fp16_A,
                leftTupleNum,
                jNode->leftTable->attrType[jNode->leftKeyIndex],
                jNode->leftTable->attrType[factCol],
                pageRankAlpha); 
        cudaErrCheck(cudaFree(gpu_fact));
        cudaErrCheck(cudaFree(factVal));

        pagerank<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_dim,
                dimVal,
                MATRIX_K,
//                d_fp16_B,
                d_fp16_BT,
                rightTupleNum,
                jNode->rightTable->attrType[jNode->rightKeyIndex],
                jNode->rightTable->attrType[dimCol],
                1.0); 
        cudaErrCheck(cudaFree(gpu_dim));
        cudaErrCheck(cudaFree(dimVal));

        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(c_cublas,0,(uint64_t)MATRIX_M*(uint64_t)MATRIX_N*sizeof(float)));
    }
#else // query ask for join counts except for OUTDEGREE, MICRO, PAGERANK
    //TODO: call corresponding filling method (check SQL pattern)
    if (gb->gbExp[gb->aggFuncIndex].func == SUM) {
        //printf("Query contains SUM\n");

        if (gb->numFuncExpCol == 1) { // Q3
            // judge whether to pass left or right data column
           // printf("rValIndex[0]\n", rValIndex[0]);
           // printf("lValIndex[0]\n", lValIndex[0]);
            if (rValIndex[0] == -1) // pass left 
            {
               // getMaxValGPU(char *column, size_t tupleNum, int attrType, int *maxVal);
                // gpu_fill_data (left), gpu_fill_transpose (right)
                gpu_fill_data<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
                    gpu_ldata,    
                    MATRIX_K,
                    d_fp16_A,
                    leftTupleNum,
                    jNode->leftTable->attrType[jNode->leftKeyIndex]);

                cudaErrCheck(cudaFree(gpu_fact));
                cudaErrCheck(cudaFree(gpu_ldata));
                // TODO: right fill with ones_gb
                
                gpu_fill_gb_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                        gpu_dim,
                        gpu_rdata,
                        MATRIX_N,
                        d_fp16_BT,
                        rightTupleNum,
                        jNode->rightTable->attrType[jNode->rightKeyIndex]);

                cudaErrCheck(cudaFree(gpu_dim));
                cudaErrCheck(cudaFree(gpu_rdata));

                /*
                gpu_fill_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_dim,
                    MATRIX_K,
                    d_fp16_BT,
                    rightTupleNum,
                    jNode->rightTable->attrType[jNode->rightKeyIndex]);

                cudaErrCheck(cudaFree(gpu_dim));
                */
            } 
            else if (lValIndex[0] == -1)// pass right 
            {
                // matA -> gbMatWidth x MATRIX_K
                gpu_fill_gb<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                        gpu_fact,
                        gpu_ldata,    
                        MATRIX_K,
                        d_fp16_A,
                        leftTupleNum,
                        jNode->leftTable->attrType[jNode->leftKeyIndex]);
                cudaErrCheck(cudaFree(gpu_fact));
                cudaErrCheck(cudaFree(gpu_ldata));

                // MATRIX_K x MATRIX_N
                gpu_fill_data_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                        gpu_dim,
                        gpu_rdata,
                        MATRIX_N,
                        d_fp16_BT,
                        rightTupleNum,
                        jNode->rightTable->attrType[jNode->rightKeyIndex]);
                cudaErrCheck(cudaFree(gpu_dim));
                cudaErrCheck(cudaFree(gpu_rdata));

            }
        }
        else if (gb->numFuncExpCol == 2) { // Q4, gb->numFuncExpCol == 2

        }
        


    }

    /* If has groupBy, after MM, then compute gbCount */
/*    if (gbConstant != 1) { // print gbCount
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMat, 1 * MATRIX_M * sizeof(char)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMatFp16, 1 * MATRIX_M * sizeof(half)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_redMat, 1, MATRIX_M * sizeof(char)));
        convertCharToFp16 <<< (MATRIX_M + 255) / 256, 256 >>> (d_redMatFp16, 
                d_redMat, MATRIX_M);
        
        // compute groupBy count by performing reduction

    }*/

    /*
     Q3
     Need to determine take left/right data column and groupBy which table? => gbLeftRight
     0: left, 1: right
     1 -- either lValIndex or rValIndex is -1, one as actual value, the other as 1
     */
    /*
    if (gb->numFuncExpCol == 1) {
        // judge whether to pass left or right data column
        if (rValIndex[0] == -1) // pass left 
        {
            // gpu_fill_data (left), gpu_fill_transpose (right)

        } 
        else if (lValIndex[0] == -1)// pass right 
        {
            // call gpu_fill (left), gpu_fill_transpose_data (right)

        }
    }*/

    /*
     Q4
     if (gb->math_op == MULTIPLY && (lValIndex[0] != -1 && rValIndex[0] != -1))
     both lValIndex/rValIndex are not -1, all pass value into func 
     */

    /*
     Else case -- leave it with the general matrix multiplication => return join_count
     */

    /*
    gpu_fill<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
            MATRIX_K,
            d_fp16_A,
            leftTupleNum,
            jNode->leftTable->attrType[jNode->leftKeyIndex]);
    cudaErrCheck(cudaFree(gpu_fact));
    
    gpu_fill_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_dim,
            MATRIX_K,
            d_fp16_BT,
            rightTupleNum,
            jNode->rightTable->attrType[jNode->rightKeyIndex]);
    cudaErrCheck(cudaFree(gpu_dim));
    */
    
#endif
    clock_gettime(CLOCK_REALTIME, &fill_end); 

#endif // end of fill matrix for CUBLAS_HALF

#elif CUBLAS

#ifdef MICRO
    //int A_tupleNum = jNode->leftTable->tupleNum;
    //int B_tupleNum = jNode->rightTable->tupleNum;

    clock_gettime(CLOCK_REALTIME, &cuMemcpy_start);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact_j,jNode->leftTable->content[0], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim_j,jNode->rightTable->content[0], primaryKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact_val,jNode->leftTable->content[1], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim_val,jNode->rightTable->content[1], primaryKeySize,cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);

    clock_gettime(CLOCK_REALTIME, &fill_start); 
    microbenchmark<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
            gpu_fact_j,
            gpu_fact_val,
            MATRIX_K,
            d_fp32_A,
            leftTupleNum,
            jNode->leftTable->attrType[jNode->leftKeyIndex]);
    cudaErrCheck(cudaFree(gpu_fact));
    microbenchmark_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_dim,
            gpu_dim_j,
            gpu_dim_val,
            MATRIX_K,
            d_fp32_BT,
            rightTupleNum,
            jNode->rightTable->attrType[jNode->rightKeyIndex]);
    cudaErrCheck(cudaFree(gpu_dim));

    clock_gettime(CLOCK_REALTIME, &fill_end); 
#else
    // No other modes for now

#endif // end of MICRO

#endif // all modes (CUBLAS_HALF and CUBLAS) filling matrix end

// set up mask for reduction if required
clock_gettime(CLOCK_REALTIME, &maskRED_start); 
#ifdef CUBLAS_HALF

#ifdef MICRO
// do nothing

#else

    // TODO: move this into RED after decouple
    setVector(h_red, MATRIX_N);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_red, h_red, sizeof(float) * MATRIX_N, cudaMemcpyHostToDevice));
#ifdef RED
//    setVector(h_red, MATRIX_N);
//    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_red, h_red, sizeof(float) * MATRIX_N, cudaMemcpyHostToDevice));
    setVector(h_red2, MATRIX_M);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_red2, h_red2, sizeof(float) * MATRIX_M, cudaMemcpyHostToDevice));
#endif

#endif // end of else MICRO
#endif // end of setting mask for CUBLAS_HALF
clock_gettime(CLOCK_REALTIME, &maskRED_end); 

// transpose B matrix
#ifdef CUBLAS_HALF

#ifdef OUTDEGREE
    //outdegree doesn't need to transpose
#elif MICRO
    //microbenchmark doesn't need to transpose
#else    
    //clock_gettime(CLOCK_REALTIME, &transpose_start);
    // no need to transpose if use gpu_fill_transpose
//    gpu_transpose<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp16_BT, d_fp16_B, MATRIX_N, MATRIX_K);
    //clock_gettime(CLOCK_REALTIME, &transpose_end);
//    cudaErrCheck(cudaFree(d_fp16_B));
#endif // end of OUTDEGREE -- transpose B matrix

#elif CUBLAS

    //clock_gettime(CLOCK_REALTIME, &transpose_start);
    //TODO: call gpu_fill_transpose directly
    //gpu_transpose<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp32_BT, d_fp32_B, MATRIX_N, MATRIX_K);
    //clock_gettime(CLOCK_REALTIME, &transpose_end);
    //cudaErrCheck(cudaFree(d_fp32_B));
#endif // end of transposing B matrix

// start cublasGemm lib to perform MM
#ifdef CUBLAS_HALF

#ifdef MICRO
    printf("Running Matrix Multiplication (dense matrix) using GemmEx with TCUs...\n");
    cudaErrCheck(cudaEventRecord(startcublasEX));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha,
                d_fp16_A, CUDA_R_16F, MATRIX_N,
                d_fp16_BT, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11
    cudaErrCheck(cudaFree(d_fp16_A));
    cudaErrCheck(cudaFree(d_fp16_B));

#elif OUTDEGREE
    printf("Running compute_outdegree with GemmEx with TCUs...\n");
    cudaErrCheck(cudaEventRecord(startcublasEX));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_M, MATRIX_M, 1,
                &alpha,
                d_fp16_B, CUDA_R_16F, MATRIX_M,
                d_fp16_A, CUDA_R_16F, 1,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11
    cudaErrCheck(cudaFree(d_fp16_A));
    cudaErrCheck(cudaFree(d_fp16_B));

// end of OUTDEGREE tcu cublasEX
#else

    float *red_sum;
    int *gbCount;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&red_sum, MATRIX_M * sizeof(float)));
    if (gb && gb->gbExp[1].func == COUNT) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gbCount, sizeof(int)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gbCount, 0, sizeof(int)));
    }

#ifdef RED
    float *red_sum2;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&red_sum2, 1 * sizeof(float)));
#endif
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_fp16_cublas, (uint64_t)MATRIX_M * (uint64_t)MATRIX_N * sizeof(half)));
    //printf("Running with cuBLAS on TCUs...\n");
    cudaErrCheck(cudaEventRecord(startcublasEX));
/*    cublasErrCheck(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha_fp16,
                d_fp16_BT,MATRIX_N,
                d_fp16_A,MATRIX_K,
                &beta_fp16,
                c_fp16_cublas, MATRIX_N));*/
#ifdef RED    
    printf("Running GemmEx RED on TCUs...\n");
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha,
                d_fp16_BT, CUDA_R_16F, MATRIX_N,
                d_fp16_A, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_N,
                //CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11
    cudaErrCheck(cudaFree(d_fp16_A));
    cudaErrCheck(cudaFree(d_fp16_BT));
#else
    // NOTE: YDB's groupby is not group by clause but aggregate function
    // outdegree.sql, gb->gbExp[0].func == DESC
    if (gb && gb->gbExp[1].func == COUNT) {
        printf("Running GemmEx COUNT on TCUs...\n");
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha,
                d_fp16_BT, CUDA_R_16F, MATRIX_N,
                d_fp16_A, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_N,
                //CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11
        cudaErrCheck(cudaFree(d_fp16_A));
        cudaErrCheck(cudaFree(d_fp16_BT));

        clock_gettime(CLOCK_REALTIME, &gbCount_start);
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, MATRIX_M, MATRIX_N,
                &alpha,
                d_red, CUDA_R_32F, 1,
                c_cublas, CUDA_R_32F, MATRIX_N,
                &beta,
                red_sum, CUDA_R_32F, 1,
                //CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11

        
        // implements COUNT operation -- print node.id with outdegree cnt
        /*
        if (gb->gbExp[1].func == COUNT)
            count_op<<<(MATRIX_M + 255) / 256, 256>>> (red_sum, MATRIX_M);
        */
        
        //clock_gettime(CLOCK_REALTIME, &gbCount_start);
        gb_count<<<(MATRIX_M + 255) / 256, 256>>> (red_sum, MATRIX_M, gbCount);
        clock_gettime(CLOCK_REALTIME, &gbCount_end);
    } else if (gb && gb->gbExp[0].func == SUM) { // no group by clause, only SUM
        
        /*
        cublasErrCheck(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha_fp16,
                d_fp16_BT,MATRIX_N,
                d_fp16_A,MATRIX_K,
                &beta_fp16,
                c_fp16_cublas, MATRIX_N));
        */
        //printf("MxK:%dx%d\tKxN:%dx%d\n", MATRIX_M, MATRIX_K, MATRIX_K, gbMatWidth);
        printf("MxK:%dx%d\tKxN:%dx%d\n", MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N);
        // TODO: have a logic to judge left/right?

        printf("Running GemmEx (Group-by aggregates) on TCUs...\n");
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha,
                d_fp16_BT, CUDA_R_16F, MATRIX_N,
                d_fp16_A, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cudaErrCheck(cudaFree(d_fp16_A));
        cudaErrCheck(cudaFree(d_fp16_BT));
        
        // If has groupBy, return gbCount after MM /
        if (gbConstant != 1) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMat, 1 * MATRIX_M * sizeof(char)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMatFp16, 1 * MATRIX_M * sizeof(half)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_redMat, 1, MATRIX_M * sizeof(char)));
            convertCharToFp16 <<< (MATRIX_M + 255) / 256, 256 >>> (d_redMatFp16, 
                d_redMat, MATRIX_M);
        
            //TODO: compute groupBy count by performing reduction
            half *temp_c;
            //CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&temp_c, MATRIX_M * gbMatWidth * sizeof(half)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&temp_c, MATRIX_M * MATRIX_N * sizeof(half)));
            
            //convertFp32ToFp16 <<< (MATRIX_M * gbMatWidth + 255) / 256, 256 >>> (temp_c, c_cublas, MATRIX_M * gbMatWidth);
            convertFp32ToFp16 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (temp_c, c_cublas, MATRIX_M * MATRIX_N);

            float *d_reduction_res;
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_reduction_res, 1 * MATRIX_N * sizeof(float)));

            printf("Perform groupBy reduction...\n");
            cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    //gbMatWidth, 1, MATRIX_M,
                    MATRIX_N, 1, MATRIX_M,
                    &alpha,
                    //temp_c, CUDA_R_16F, gbMatWidth,
                    temp_c, CUDA_R_16F, MATRIX_N,
                    d_redMatFp16, CUDA_R_16F, MATRIX_M,
                    &beta,
                    //d_reduction_res, CUDA_R_32F, gbMatWidth,
                    d_reduction_res, CUDA_R_32F, MATRIX_N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            cudaErrCheck(cudaFree(temp_c));
            cudaErrCheck(cudaFree(d_redMatFp16));

            // count number of column with values
            int *d_gbCount, *h_gbCount;
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_gbCount, 1 * sizeof(int)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_gbCount, 0, 1 * sizeof(int)));
            h_gbCount = (int*)malloc(1 * sizeof(int));
            
            //groupByCount<<<(gbMatWidth+255), 256>>> (d_reduction_res, gbMatWidth, d_gbCount);
            groupByCount<<<(MATRIX_N+255), 256>>> (d_reduction_res, MATRIX_N, d_gbCount);
            cudaErrCheck(cudaFree(d_reduction_res));
            cudaErrCheck(cudaMemcpy(h_gbCount, d_gbCount, 1 * sizeof(int), cudaMemcpyDeviceToHost));
            printf("GroupBy Count: %d\n", *h_gbCount);

        }

//        pageRankAdd<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_cublas, MATRIX_M*MATRIX_N, pageRankAlpha, MATRIX_K);
        // verify result
        /*
        cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
        verify_result(c_host_cublas, MATRIX_M, MATRIX_N);
        */  
    }
    else if (gbConstant !=1) { // contains groupBy keyword

        float *test_red;
        int *gbcount;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_red, MATRIX_M * sizeof(float)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gbcount, sizeof(int)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gbcount, 0, sizeof(int)));

        printf("Running GemmEX (w/ groupBy...)\n");
        // call TCU join operator
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha,
                d_fp16_BT, CUDA_R_16F, MATRIX_N,
                d_fp16_A, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_N,
                //CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11
        cudaErrCheck(cudaFree(d_fp16_A));
        cudaErrCheck(cudaFree(d_fp16_BT));

        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, MATRIX_M, MATRIX_N,
                &alpha,
                d_red, CUDA_R_32F, 1,
                c_cublas, CUDA_R_32F, MATRIX_N,
                &beta,
                test_red, CUDA_R_32F, 1,
                //CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11

        // TODO: need to reduce one time (groupBy) then count 
//        cublasStatus_t ret;
//        ret = cublasCreate(&cublasHandle);
//        float *cb_res = (float*)malloc(sizeof(float));
//        ret = cublasSasum(cublasHandle, MATRIX_M*MATRIX_N, c_cublas, 1, cb_res);
//        printf("groupBy count: %.0f\n", *cb_res);
        // call TCU groupBy operator => return gbCount
        printf("MATRIX_M: %d\n", MATRIX_M);
        gb_count<<<(MAX_THREADS+MATRIX_M-1)/MAX_THREADS,MAX_THREADS>>> (test_red, MATRIX_M, gbcount);

        int h_gbCount = 0;
        cudaErrCheck(cudaMemcpy(&h_gbCount, gbcount, sizeof(int), cudaMemcpyDeviceToHost));
        printf("groupBy count: %d\n", h_gbCount);

    } 
    else {
        printf("Running Hgemm...\n");
        // no group by keyword, directly perform cublasHgemm
        cublasErrCheck(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha_fp16,
                d_fp16_BT,MATRIX_N,
                d_fp16_A,MATRIX_K,
                &beta_fp16,
                c_fp16_cublas, MATRIX_N));
        cudaErrCheck(cudaFree(d_fp16_A));
        cudaErrCheck(cudaFree(d_fp16_BT));
    }
//    res->attrTotalSize[2] = 4*MATRIX_M*MATRIX_N;
#endif // end of RED TCU operation

#endif // end of MICRO -- L727
    cudaErrCheck(cudaEventRecord(stopcublasEX));

#ifdef RED // reduction to get correct join counts
    clock_gettime(CLOCK_REALTIME, &gbCount_start);
    // 1st reduction -> single column
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, MATRIX_M, MATRIX_N,
                &alpha,
                d_red, CUDA_R_32F, 1,
                c_cublas, CUDA_R_32F, MATRIX_N,
                &beta,
                red_sum, CUDA_R_32F, 1,
                //CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11

    // return groupBy count
    if (gb && gb->gbExp[1].func == COUNT) {
        // FIXME: why red_sum is 0, gb_count check 0.0 or 0?
        gb_count<<<(MATRIX_M + 255) / 256, 256>>> (red_sum, MATRIX_M, gbCount);
    }
    
    // 2nd reduction -> sinlge value
    // return join count
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, 1, MATRIX_M,
                &alpha,
                red_sum, CUDA_R_32F, 1,
                d_red2, CUDA_R_32F, MATRIX_M,
                &beta,
                red_sum2, CUDA_R_32F, 1,
                //CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // CUDA 11
    clock_gettime(CLOCK_REALTIME, &gbCount_end);
    cudaErrCheck(cudaFree(red_sum));
    cudaErrCheck(cudaFree(d_red2));
    
#endif
//    cudaErrCheck(cudaEventRecord(stopcublasEX));
#elif CUBLAS
    printf("Running with SGemm...\n");
    cudaErrCheck(cudaEventRecord(startcublas));
    cublasSgemm(cublasHandle_default, CUBLAS_OP_N, CUBLAS_OP_N,
            MATRIX_N, MATRIX_M, MATRIX_K,
            &alpha,
            d_fp32_BT, MATRIX_N,
            d_fp32_A, MATRIX_K,
            &beta,
            c_sgemm, MATRIX_N);
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaFree(d_fp32_A));
    cudaErrCheck(cudaFree(d_fp32_BT));
#endif    

#ifdef CUBLAS_HALF
    float cublasEXTime;

    cudaErrCheck(cudaEventSynchronize(stopcublasEX));
    cudaErrCheck(cudaEventElapsedTime(&cublasEXTime, startcublasEX, stopcublasEX));

    // test output
    /*
    float *tmp_res;
    tmp_res = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    cudaErrCheck(cudaMemcpy(tmp_res, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < MATRIX_M*MATRIX_N; i++) {
        printf("%.2f\t", tmp_res[i]);
        if ((i+1)%MATRIX_N == 0)
            printf("\n\n");
    }
    */

#ifdef RED
    clock_gettime(CLOCK_REALTIME, &count_start);
    float *ans;
    ans = (float*)calloc(1, sizeof(float));
    cudaErrCheck(cudaMemcpy(ans, red_sum2, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    if (gb && (gb->gbExp[1].func == COUNT || gb->gbExp[0].func == COUNT)) {
        int h_gbCount = 0;
        cudaErrCheck(cudaMemcpy(&h_gbCount, gbCount, sizeof(int), cudaMemcpyDeviceToHost));
        printf("groupBy count: %d\n", h_gbCount);
        double gbCount_elapse = (gbCount_end.tv_sec -  gbCount_start.tv_sec)* BILLION + gbCount_end.tv_nsec - gbCount_start.tv_nsec;
        printf("GroupBy Time: %lf(ms)\n", gbCount_elapse/(1000*1000));
    }
    clock_gettime(CLOCK_REALTIME, &count_end);
    printf("c_host_cublas reduction sum: %.0f\n", ans[0]);
    free(ans);
//    cudaErrCheck(cudaFree(red_sum));
    cudaErrCheck(cudaFree(red_sum2));
#elif PAGERANK
    // print is time consuming, cudaMemcpy time is also for the purpose of verification
    clock_gettime(CLOCK_REALTIME, &pagerankVerify_start);
    cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, &pagerankVerify_end);
#elif MICRO
    // only for verification
    //cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, sizeof(float)*MATRIX_M*MATRIX_N, cudaMemcpyDeviceToHost));

#elif OUTDEGREE
// do nothing for now

#else // not using Reduction, sum using cublasSasum
    clock_gettime(CLOCK_REALTIME, &count_start);
    if (gb && (gb->gbExp[1].func == COUNT || gb->gbExp[0].func == COUNT)) {
        int h_gbCount = 0;
        cudaErrCheck(cudaMemcpy(&h_gbCount, gbCount, sizeof(int), cudaMemcpyDeviceToHost));
        printf("groupBy count: %d\n", h_gbCount);
        double gbCount_elapse = (gbCount_end.tv_sec -  gbCount_start.tv_sec)* BILLION + gbCount_end.tv_nsec - gbCount_start.tv_nsec;
        printf("GroupBy Time: %lf(ms)\n", gbCount_elapse/(1000*1000));

    } else {
        // previous calculate by cublasHgemm: need conversion
        convertFp16ToFp32<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_cublas, c_fp16_cublas, MATRIX_M * MATRIX_N);
    }

    uint64_t input_len = MATRIX_M*MATRIX_N;
    int asum_len = 200000000; // Sasum addition per section

    cublasStatus_t ret;
    ret = cublasCreate(&cublasHandle);
//    printf("input_len: %lu\n", input_len);

    if (input_len < asum_len) {
        float *cb_res = (float*)malloc(sizeof(float));
        ret = cublasSasum(cublasHandle, MATRIX_M*MATRIX_N, c_cublas, 1, cb_res);
        clock_gettime(CLOCK_REALTIME, &count_end);
        printf("c_host_cublas sum: %.0f\n", *cb_res);
    } else { // support on machine has sufficient device memory ~15GB
        int num_sec = (int)(ceil(input_len/(float)asum_len));
        int remain = input_len % asum_len;
        float cb_res = 0;
        uint64_t pos = 0;
        uint64_t sum_res = 0;
        int i;
        for (i = 0; i < num_sec-1; i++) {
            ret = cublasSasum(cublasHandle, asum_len, c_cublas+pos, 1, &cb_res);
            pos += asum_len;
            sum_res += (uint64_t)cb_res;
            //printf("i: %d\tcb_res: %f\tsum_res: %lu\n",i,cb_res,sum_res);
        }
        ret = cublasSasum(cublasHandle, remain, c_cublas+pos, 1, &cb_res);
        sum_res += (uint64_t)cb_res;
        clock_gettime(CLOCK_REALTIME, &count_end);
        printf("c_host_cublas sum: %lu\n", sum_res);
    }
#endif
    printf("cublasEX tensor cores (FP16) took %fms\n", cublasEXTime);
    
    free(c_host_cublas);
//    cudaErrCheck(cudaFree(c_cublas));
#elif CUBLAS
    float cublasTime;

//    cudaErrCheck(cudaMemcpy(c_host_sgemm, c_sgemm, sizeof(float)*MATRIX_M*MATRIX_N, cudaMemcpyDeviceToHost));

    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    clock_gettime(CLOCK_REALTIME, &count_start);
    /*
    cublasStatus_t sgemm_ret;
    sgemm_ret = cublasCreate(&cublasHandle_default);
    float *cbsgemm_res = (float*)malloc(sizeof(float));
    sgemm_ret = cublasSasum(cublasHandle_default, MATRIX_M*MATRIX_N, c_sgemm, 1, cbsgemm_res);
    clock_gettime(CLOCK_REALTIME, &count_end);
    printf("c_host_sgemm sum: %.0f\n", *cbsgemm_res);
    */
    printf("cublas sgemm (FP32) took %fms\n", cublasTime);

    cudaErrCheck(cudaEventDestroy(startcublas));
    cudaErrCheck(cudaEventDestroy(stopcublas));
    free(c_host_sgemm);
    cudaErrCheck(cudaFree(c_sgemm));

#endif

// free those data structures
#ifdef CUBLAS_HALF


#elif CUBLAS
    free(h_fp32_A);
    free(h_fp32_B);
#endif

    clock_gettime(CLOCK_REALTIME, &tcu_end);
    double tcu_fill = (fill_end.tv_sec -  fill_start.tv_sec)* BILLION + fill_end.tv_nsec - fill_start.tv_nsec;
    //double tcu_convert = (convert_end.tv_sec -  convert_start.tv_sec)* BILLION + convert_end.tv_nsec - convert_start.tv_nsec;
    double tcu_elapse = (tcu_end.tv_sec -  tcu_start.tv_sec)* BILLION + tcu_end.tv_nsec - tcu_start.tv_nsec;
    double init_elapse = (init_end.tv_sec -  init_start.tv_sec)* BILLION + init_end.tv_nsec - init_start.tv_nsec;
    double cuMemcpy_elapse = (cuMemcpy_end.tv_sec -  cuMemcpy_start.tv_sec)* BILLION + cuMemcpy_end.tv_nsec - cuMemcpy_start.tv_nsec;
#if defined(CUBLAS_HALF) || defined(CUBLAS)
    double count_elapse = (count_end.tv_sec -  count_start.tv_sec)* BILLION + count_end.tv_nsec - count_start.tv_nsec;
    //double debug_elapse = (debug_end.tv_sec -  debug_start.tv_sec)* BILLION + debug_end.tv_nsec - debug_start.tv_nsec;
    //double transpose_elapse = (transpose_end.tv_sec -  transpose_start.tv_sec)* BILLION + transpose_end.tv_nsec - transpose_start.tv_nsec;
#endif


#ifdef PAGERANK
    double pagerankVerify_elapse = (pagerankVerify_end.tv_sec -  pagerankVerify_start.tv_sec)* BILLION + pagerankVerify_end.tv_nsec - pagerankVerify_start.tv_nsec;
#endif
    
    printf("Initialization: %lf(ms)\n", init_elapse/(1000*1000));
    printf("Matrices filling: %lf(ms)\n", tcu_fill/(1000*1000));
    printf("cudaMemcpy: %lf(ms)\n", cuMemcpy_elapse/(1000*1000));
    printf("MMA total time: %lf(ms)\n", tcu_elapse/(1000*1000));
#ifdef CUBLAS_HALF

if (gb && (gb->gbExp[1].func == COUNT || gb->gbExp[0].func == COUNT)) {
    //printf("cublasEX join time: %lf(ms)\n", test_elapse/(1000*1000));
    printf("cublasEX sum counting: %lf(ms)\n", count_elapse/(1000*1000));
}
    //printf("cublasCreate cold start: %lf(ms)\n", debug_elapse/(1000*1000));
    //printf("gpu transpose: %lf(ms)\n", transpose_elapse/(1000*1000));
#ifdef PAGERANK
    printf("PageRank Verify cudaMemcpy time: %lf(ms)\n", pagerankVerify_elapse/(1000*1000));
#endif

#elif CUBLAS
    //printf("cublasSGEMM sum counting: %lf(ms)\n", count_elapse/(1000*1000));
    printf("cublasCreate cold start: %lf(ms)\n", debug_elapse/(1000*1000));
    //printf("gpu transpose: %lf(ms)\n", transpose_elapse/(1000*1000));
#endif
//#ifdef DEBUG
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();
//#endif
    return res; // FIXME: return res table if second join need this as input  

}
