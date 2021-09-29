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
#include <cuda_runtime_api.h>
#include <cusparse.h>         // cusparseSpMM
#include "tcuSpMM.h"
//#ifdef DEBUG
//#include "../include/cuPrintf.cu"
//#include "../include/cuPrintf.cuh"
//#endif

using namespace nvcuda;

#define MAX_THREADS 1024 // For NVIDIA Turing Architecture
#define BLOCK_SIZE 1024

// Define some error checking macros.

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n",
                cudaGetErrorString(stat),
                file, line);
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

#define cusparseErrCheck(stat) { cusparseErrCheck_((stat), __FILE__, __LINE__); }
void cusparseErrCheck_(cusparseStatus_t stat, const char *file, int line) {
    if (stat != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE API failed: %d : %s %s %d\n", stat, 
                cusparseGetErrorString(stat), file, line);
    }
}

void* cublasCreateThread(void *x)
{
    cublasHandle_t* cublasHandle = (cublasHandle_t *)x;
    cublasErrCheck(cublasCreate(cublasHandle));
    cublasErrCheck(cublasSetMathMode(*cublasHandle,CUBLAS_TENSOR_OP_MATH));
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
__global__ void static gpu_fill(char *column, int matWidth, half *matA, 
        size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    //int value = (int)column[index]; // char -> int will lose 3 bytes
    int *value   = (int*)&column[index];
    matA[i*matWidth + (*value)] = __float2half(1.0f);
}

__global__ void static gpu_fill_2data(char *join_column, char *data_column, 
        char *data_column2, int matWidth_k, half *matA, size_t tupleNum, 
        int attrType, int scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];// col
    int *data_value = (int*)&data_column[index];// val
    int *data2 = (int*)&data_column2[index];    // row
    matA[(*data2) * matWidth_k + (*join_value)] = __float2half((float)(*data_value)/scale);
    //printf("matA[%d]: %f\n", (*data2) * matWidth_k + (*join_value), (float)(*data_value)/scale);
    //matA[(*data2) * matWidth_k + (*join_value)] = __float2half(65504.0f);
    //printf("row: %d\tcol: %d\tval: %d\n", *data2, *join_value, *data_value);
}

__global__ void static gpu_fill_moredata(char *join_column, char *data_column, 
        char *data_column2, int matWidth_k, half *matA, 
        size_t tupleNum, int attrType, int scale) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];
    int *data1 = (int*)&data_column[index];
    int *data2 = (int*)&data_column2[index];
    matA[i*matWidth_k+(*join_value)] = __float2half((float)(*data1)*(*data2)/scale);
}

/* Fill matrix with data value. */
__global__ void static gpu_fill_data(char *join_column, char *data_column, 
        int matWidth_k, half *matA, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];
    int *data_value = (int*)&data_column[index];
    matA[i * matWidth_k + (*join_value)] = __float2half((float)(*data_value));
}

__global__ void static gpu_fill_gb(char *join_column, char *data_column, 
        int matWidth_k, half *matA, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];
    int *data_value = (int*)&data_column[index];
    matA[(*data_value) * matWidth_k + (*join_value)] = __float2half(1.0f);
}

__global__ void static gpu_fill_data_transpose(char *join_column, 
        char *data_column, int matWidth_n, half *matB, size_t tupleNum, 
        int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *join_value = (int*)&join_column[index];
    int *data_value = (int*)&data_column[index];
    matB[(*join_value) * matWidth_n + i] = __float2half((float)(*data_value));
}

/* Fill matrix with ones according to groupBy column in transpose format. */
__global__ void static gpu_fill_gb_transpose(char *join_column, 
        char *data_column, int matWidth_n, half *matB, 
        size_t tupleNum, int attrType) {
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
__global__ void static gpu_fill_transpose(char *column, int matWidth, 
        half *matB, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int index = i * attrType;
    int *value   = (int*)&column[index];
    int pos = (*value)*tupleNum+i;
    matB[pos] = __float2half(1.0f);
}

/* Fill matrix in dense format for matrix multiplication */
__global__ void static microbenchmark(char *mat_i, char *mat_j, char *mat_val, 
        int matWidth, half *mat, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= tupleNum) return;

    int index = i * attrType;
    int *row  = (int*)&mat_i[index]; 
    int *col  = (int*)&mat_j[index]; 
    int *val  = (int*)&mat_val[index];
    mat[(*row)*matWidth+(*col)] = __int2half_rn(*val);
}

__global__ void static microbenchmark_transpose(char *mat_i, char *mat_j, 
        char *mat_val, int matWidth, half *mat, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= tupleNum) return;

    int index = i * attrType;
    int *row  = (int*)&mat_i[index]; 
    int *col  = (int*)&mat_j[index]; 
    int *val  = (int*)&mat_val[index];
    mat[(*col)*matWidth+(*row)] = __int2half_rn(*val);
}

__global__ void static outdegree_fill(char *column_val, half *mat, 
        size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= tupleNum) return;

    int index = i * attrType;
    int *val        = (int*)&column_val[index];
    mat[(*val)] = __hadd(mat[(*val)], __int2half_rn(1));
}

/* COUNT operator (histogram) */
__global__ void count(char * d_in, int * d_bins, int sz) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > sz) return;

	int *idx = (int*)&d_in[i * 4];
	atomicAdd(&d_bins[(*idx)], 1);
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

__global__ void static pageRankAdd(float *mat, int n, float pageRankAlpha, 
        int numNodes) {
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

__global__ void static naiveCount(float *res, int n, int *count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        if (res[idx] > 0.000001) {
            //__syncthreads();
            atomicAdd(count, 1);
            //printf("res[%d]: %f\n", idx, res[idx]);
            //printf("count: %d\n", *count);
        }
    }
}

__global__ void placeCount(float *out, float *in, unsigned size)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= size) return;
    if (in[tid] > 0.00001)
        out[tid] = 1.0f;
    else
        out[tid] = 0.0f;
}

__host__ uint64_t getCount(float *in, uint64_t size)
{
    const uint64_t asumLen = 2e9;

    cublasStatus_t ret;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    uint64_t iter = size / asumLen;
    uint64_t overflow = size % asumLen;
    uint64_t offset = 0;
    float partialRes;
    uint64_t sumRes = 0;

    for (uint64_t i = 0; i < iter; i++)
    {
        ret = cublasSasum(cublasHandle, asumLen, in+offset, 1, &partialRes);
        offset += asumLen;
        sumRes += (uint64_t) partialRes;
    }

    // handle overflow
    if (overflow)
    {
        ret = cublasSasum(cublasHandle, overflow, in+offset, 1, &partialRes);
        sumRes += (uint64_t) partialRes;
    }

    cublasDestroy(cublasHandle);
    return sumRes;
} 

__global__ void reductionCount(int *out, float *in, unsigned size) 
{
    __shared__ int partialSum[256];
    unsigned int tid   = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    partialSum[tid] = 0;
    if (tid >= size) return;

    if (tid + start + blockDim.x < size) 
    {
        if (in[tid + start] > 0.000001) 
        {
            partialSum[tid]++;
        }
        if (in[tid + start + blockDim.x] > 0.000001) 
        {
            partialSum[tid]++;
        }
    }
    else 
    {
        if (in[tid + start] > 0.000001) 
        {
            partialSum[tid]++;
        }
    }
    
    __syncthreads();
    //if (partialSum[tid])
    //    printf("tid: %d\t partialSum: %d\n", tid, partialSum[tid]);
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) { // reduce to left triangle
            partialSum[tid] += partialSum[tid + stride];
        }
    }

    if (tid == 0)
        out[blockIdx.x] = partialSum[0];
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

/* only for small matrix result verification. */
__host__ static void print_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        printf("%.0f\t", mat[i]);
        if ((i+1) % cols == 0) {
            printf("\n");
        }
    }
}

/* Get column index from aggregate function for later data copy. */
__host__ static void getValIndex(struct joinNode *jNode, struct groupByNode *gb, 
        int *lValIndex, int *rValIndex, 
        int &lgbIndex, int &rgbIndex, 
        //int *ldataColIndex, int &rdataColIndex) {
        int &ldataColIndex, int &rdataColIndex,
        int &ldata2) {

    for (int i = 0; i < jNode->leftOutputAttrNum; i++) {
        for (int j = 0; j < gb->numFuncExpCol; j++) {
            // find index of aggFunc, e.g. SUM(X.column_name)
            if (jNode->leftPos[i] == gb->funcExpColIndex[j]) {
                lValIndex[i] = jNode->leftOutputIndex[i];

                if (ldataColIndex == -1) {
                    ldataColIndex = jNode->leftOutputIndex[i];
                    //printf("agg left ldataColIndex[%d]: %d\n", i,ldataColIndex);
                }
            }
            for (int k = 0; k < gb->groupByColNum; k++) {
                if (jNode->leftPos[i] == gb->groupByIndex[k])
                    lgbIndex = 1;
            }
        }
        // TODO: hard to know which colIdx belong to sum() as value or use as one dimension
        if (ldata2 == -1) {
            ldata2 = jNode->leftOutputIndex[i];
        }
        /*
        if (ldataColIndex[i] == -1) {
            ldataColIndex[i] = jNode->leftOutputIndex[i];
            printf("left ldataColIndex[%d]: %d\n", i,ldataColIndex[i]);
        }*/
    } 
    
    for (int i = 0; i < jNode->rightOutputAttrNum; i++) {
        for (int j = 0; j < gb->numFuncExpCol; j++) {
            //if (jNode->rightPos[i] == gb->funcExpColIndex[j]) {
            //    rValIndex[i] = jNode->rightOutputIndex[i];

                if (rdataColIndex == -1) {
                    rdataColIndex = jNode->rightOutputIndex[i];
                    //printf("right rdataColIndex: %d\n", rdataColIndex);
                }
            //}

            for (int k = 0; k < gb->groupByColNum; k++) {
                if (jNode->rightPos[i] == gb->groupByIndex[k])
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
        //printf("i: %d\tval: %d\n", i, *val);
        //if (*val > 100000)
        //    printf("i: %d\tval: %d\n", i, *val);
        if (localMax < *val) {
            localMax = *val;
        }
    }
    return localMax;
}

__host__ int getMinVal(char *column, size_t tupleNum, int attrType) {
    int localMin = 0;

    for (int i = 0; i < tupleNum; i++) {
        int *val = (int*)&column[i*attrType];
        if (localMin > *val) {
            localMin = *val;
        }
    }
    return localMin;
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

struct gpu_timer {
    gpu_timer() {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start, 0);
    }

    float milliseconds_elapsed() {
        float elapsed_time;
        cudaEventRecord(m_end, 0);
        cudaEventSynchronize(m_end);
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        return elapsed_time;      
    }

    float seconds_elapsed() {
        return milliseconds_elapsed() / 1000.0;
    }

  protected:
    cudaEvent_t m_start, m_end;
};

/* Read tableNode and convert into Coo matrix.
 * transpose -- 0: NON-TRANSPOSE, 1: TRANSPOSE 
 * fillOne   -- 0: fill data value, 1: fill 1 */
void tbl2coo(int XtupleNum, char *XjoinKey, char *Xdata,
             int *cooRowInd, int *cooColInd, float *cooValues,
             int transpose, int fillOne)
{
    if (transpose)
    {
        for (int i = 0; i < XtupleNum; i++)
        {
            int *key = (int*)&XjoinKey[i*sizeof(int)];
            cooRowInd[i] = *key;
            //cooRowInd[i] = (int)XjoinKey[i*sizeof(int)];
            cooColInd[i] = i;
            //printf("coo_T row: %d col: %d\n", i, *key);
            if (fillOne) {
                cooValues[i] = 1.0f;

                //printf("coo_T row: %d col: %d val: %f\n", cooRowInd[i], cooColInd[i], cooValues[i]);
            } else {
                cooValues[i] = (float)Xdata[i*sizeof(float)];
            }
        }
    } else
    {
        for (int i = 0; i < XtupleNum; i++)
        {
            int *key = (int*)&XjoinKey[i*sizeof(int)];
            cooRowInd[i] = i;
            //cooColInd[i] = (int)XjoinKey[i*sizeof(int)];
            cooColInd[i] = *key;
            //printf("coo row: %d col: %d\n", i, *key);
            if (fillOne) {
                cooValues[i] = 1.0f;
                //printf("coo row: %d col: %d val: %f\n", cooRowInd[i], cooColInd[i], cooValues[i]);
            } else {
                cooValues[i] = (float)Xdata[i*sizeof(float)];
            }
        }
    }
}

/* If has groupBy keyword, one matrix width will need to update.
 * Instead of using tupleNum, using Xdata as one dimension.  */
void tbl2coo_gb(int XtupleNum, char *XjoinKey, char *Xdata,
                int *cooRowInd, int *cooColInd, float *cooValues,
                int transpose, int fillOne)
{
    if (transpose)
    {
        for (int i = 0; i < XtupleNum; i++)
        {
            cooRowInd[i] = (int)XjoinKey[i*sizeof(int)];
            cooColInd[i] = (int)Xdata[i*sizeof(int)];
            if (fillOne) {
                cooValues[i] = 1.0f;
            } else {
                cooValues[i] = (float)Xdata[i*sizeof(float)];
            }
        }
    } else
    {
        for (int i = 0; i < XtupleNum; i++)
        {
            cooRowInd[i] = (int)Xdata[i*sizeof(int)];
            cooColInd[i] = (int)XjoinKey[i*sizeof(int)];
            if (fillOne) {
                cooValues[i] = 1.0f;
            } else {
                cooValues[i] = (float)Xdata[i*sizeof(float)];
            }
        }
    }
}

/* Convert matrix format from Coo to Csr. */
void coo2csr(int X_num_rows, int Xnnz,
             int *X_cooRowInd, int *X_cooColInd, float *X_cooValues,
             int *csrOffsets, int *csrColumns, float *csrValues)
{
    // check how many elements in each row
    int *num_elems_each_row = (int*)calloc(X_num_rows, sizeof(int));

    // count num_elems
    for (int i = 0; i < Xnnz; i++)
    {
        num_elems_each_row[X_cooRowInd[i]]++;
    }

    // prefix sum
    for (int i = 0; i < X_num_rows; i++)
    {
        csrOffsets[i+1] = num_elems_each_row[i] + csrOffsets[i];
    }

    for (int i = 0; i < Xnnz; i++)
    {
        num_elems_each_row[X_cooRowInd[i]]--;
        int r = X_cooRowInd[i];
        int offset = csrOffsets[r] + num_elems_each_row[X_cooRowInd[i]];
        csrColumns[offset] = X_cooColInd[i];
        csrValues[offset] = X_cooValues[i];
    }
}

/* Prepare CSR format from table entries. */
void tbl2csr(int tupleNum, char *joinKey, char *Xdata,
             int *csrOffsets, int *csrColumns, float *csrValues,
             int fillOne)
{
    for (int i = 0; i < tupleNum; i++) {
        int *key = (int*)&joinKey[i * sizeof(int)];
        //int key = *(joinKey + i * sizeof(int));
        csrOffsets[i+1] = i+1;
        csrColumns[i] = *key;

        if (fillOne) {
            csrValues[i] = 1.0f;
        } else {
            float *data = (float*)&Xdata[i * sizeof(float)];
            csrValues[i] = *data;
        }
    }
}

/* Calculate number of elements per row. */
__global__ void row_count(int nnz, int *num_elems_per_row,
        char *joinKey) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nnz) return;
    int *key = (int*)&joinKey[i * sizeof(int)];
    atomicAdd_system(&num_elems_per_row[*key], 1);
}

/* Prefixsum for csrRowOffsets. */
__global__ void prefixsum(int num_rows,
        int *num_elems_per_row, int *d_csrOffsets)
{
    unsigned int d_hist_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (d_hist_idx >= num_rows) return;
    unsigned int cdf_val = 0;
    for (int i = 0; i <= d_hist_idx; ++i) {
        cdf_val = cdf_val + num_elems_per_row[i];
    }
    d_csrOffsets[d_hist_idx] = cdf_val;
}

__global__ void gpu_tbl2csr_transpose(int tupleNum,
        char *joinKey, char *Xdata,
        int *num_elems_per_row,
        int *csrOffsets, int *csrColumns, float *csrValues,
        int fillOne)
{
    //int i = blockDim.x * blockIdx.x + threadIdx.x;
    //if (i >= tupleNum) return;

    for (int i = 0; i < tupleNum; i++) {
        int *key = (int*)&joinKey[i * sizeof(int)];
        //TODO:these two lines require atomic operations
        num_elems_per_row[*key]--;
        int offset = csrOffsets[*key] + num_elems_per_row[*key];

        csrColumns[offset] = i;
        if (fillOne) {
            csrValues[i] = 1.0f;
        } else {
            float *data = (float*)&Xdata[i * sizeof(float)];
            csrValues[offset] = *data;
        }
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
 * 4. Metadata such as sparsity and number of non-zero elements are known from user.
 * Here, we have our assumption based on our filling methods.
 */
struct tableNode * tcuJoin(struct joinNode *jNode, struct statistic *pp, 
        struct groupByNode *gb)
        //int *matrix_dim, struct groupByNode *gb)
{
    cudaDeviceReset();

    float initTime, cudaMemcpyTime, fillTime, end2endTime;
    float tcu_compute_time, tcu_groupBy_time;
    float count_time;

    struct tableNode * res = NULL;
    int leftTupleNum  = jNode->leftTable->tupleNum;
    int rightTupleNum = jNode->rightTable->tupleNum;
    uint64_t MATRIX_M, MATRIX_N, MATRIX_K; // avoid overflow
    uint64_t Annz, Bnnz;
    uint64_t pagerank_num_nodes = 0;
    float pagerank_alpha = 0.85;
    uint64_t left_gbWidth = 1, right_gbWidth = 1;

    res = (struct tableNode*) malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);
    res->totalAttr = jNode->totalAttr;
    res->tupleSize = jNode->tupleSize;
    res->attrType = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->attrType);
    res->attrSize = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->attrSize);
    res->attrIndex = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->attrIndex);
    res->attrTotalSize = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->attrTotalSize);
    res->dataPos = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->dataPos);
    res->dataFormat = (int *) malloc(res->totalAttr * sizeof(int));
    CHECK_POINTER(res->dataFormat);
    res->content = (char **) malloc(res->totalAttr * sizeof(char *));
    CHECK_POINTER(res->content);

    for(int i=0;i<jNode->leftOutputAttrNum;i++)
    {
        int pos = jNode->leftPos[i];
        res->attrType[pos] = jNode->leftOutputAttrType[i];
        int index = jNode->leftOutputIndex[i];
        res->attrSize[pos] = jNode->leftTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    for(int i=0;i<jNode->rightOutputAttrNum;i++)
    {
        int pos = jNode->rightPos[i];
        res->attrType[pos] = jNode->rightOutputAttrType[i];
        int index = jNode->rightOutputIndex[i];
        res->attrSize[pos] = jNode->rightTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    int maxLeftJoin = 0, maxRightJoin = 0;
    
    maxLeftJoin = getMaxVal(jNode->leftTable->content[jNode->leftKeyIndex],
                            leftTupleNum,
                            jNode->leftTable->attrType[jNode->leftKeyIndex]);
    
    maxRightJoin = getMaxVal(jNode->rightTable->content[jNode->rightKeyIndex],
                            rightTupleNum,
                            jNode->rightTable->attrType[jNode->rightKeyIndex]);

    // scan to find #uniq_k -- assume known in DB
    int uniq_K = max(maxLeftJoin, maxRightJoin)+1;
#ifdef DEBUG
    printf("M: %d N: %d\n", leftTupleNum, rightTupleNum);
    printf("MATRIX_K: %d\n", uniq_K);
#endif
    
    MATRIX_K = uniq_K; // determine the sparsity
    MATRIX_M = leftTupleNum;
    MATRIX_N = rightTupleNum;
    pagerank_num_nodes = MATRIX_M;

    // assume each row contains only 1 element, i.e., Xnnz == XtupleNum
    Annz = MATRIX_M;
    Bnnz = MATRIX_N;

    int *hA_csrOffsets, *hA_csrColumns;
    int *hB_csrOffsets, *hB_csrColumns;
    float *hA_csrValues, *hB_csrValues;

    int A_num_rows = leftTupleNum;
    int A_num_cols = MATRIX_K;
    int B_num_rows = MATRIX_K;
    int B_num_cols = rightTupleNum;

    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
    long primaryKeySize = jNode->rightTable->attrTotalSize[jNode->rightKeyIndex];
    
    int gbConstant = 0;   // 0: has groupBy, 1: no groupBy keyword
    int gbLeftRight = -1; // 0: gb by left, 1: gb by right
    int gbMatWidth = 0;   // size of dom(gb_column.val)
    
    int *lValIndex, *rValIndex;
    int ldataColIndex = -1;
    int ldata2 = -1;
    //int *ldataColIndex;
    int rdataColIndex = -1;
    int lgbIndex = -1, rgbIndex = -1;

    int quantizedScale = 1;  // quantization scale
    //int quantizedScale = 7237036;
    //ldataColIndex = (int *)malloc(sizeof(int) * jNode->leftOutputAttrNum);
    //memset(ldataColIndex, -1, sizeof(int) * jNode->leftOutputAttrNum);

    // printf("pagerank alpha: %.2f\n", pagerank_alpha);
    // printf("pagerank #nodes: %d\n", pagerank_num_nodes);

    if (gb && (gb->groupByColNum == 1 && gb->groupByIndex[0] == -1)) 
    {
        gbConstant = 1;
    }
        
    // handle func first or dense/sparse first -> then update gbMatWidth
    int gbLeftIndex = -1, gbRightIndex = -1;
    if (gb && gb->gbLeftColIndex != NULL && gb->gbRightColIndex != NULL) {
#ifdef DEBUG
        printf("group by both left and right\n");
#endif
        gbLeftIndex = gb->gbLeftColIndex[0];
        gbRightIndex = gb->gbRightColIndex[0];
    }

    if (gb && gb->gbLeftColIndex != NULL) {
#ifdef DEBUG
        printf("group by left\n");
#endif
        gbLeftIndex = gb->gbLeftColIndex[0];
    }

    if (gb && gb->gbRightColIndex != NULL) {
#ifdef DEBUG
        printf("group by right\n");
#endif
        gbRightIndex = gb->gbRightColIndex[0];
    }

#ifdef DEBUG
    printf("LgbIdx: %d\tRgbIdx: %d\n", gbLeftIndex, gbRightIndex);
    printf("leftTupleNum: %d\trightTupleNum: %d\n", leftTupleNum, rightTupleNum);
#endif

    if (gb && gbConstant != 1) // contains groupBy keyword
    {
        lValIndex = (int *)malloc(sizeof(int) * jNode->leftOutputAttrNum);
        rValIndex = (int *)malloc(sizeof(int) * jNode->rightOutputAttrNum);
        memset(lValIndex, -1, sizeof(int) * jNode->leftOutputAttrNum);
        memset(rValIndex, -1, sizeof(int) * jNode->rightOutputAttrNum);

        getValIndex(jNode, gb, lValIndex, rValIndex, lgbIndex, rgbIndex, 
                ldataColIndex, rdataColIndex, ldata2);
#ifdef DEBUG
        
        printf("numFuncExpCol: %d\n", gb->numFuncExpCol);
        printf("lValIndex[0]: %d\n", lValIndex[0]);
        printf("rValIndex[0]: %d\n", rValIndex[0]);
        printf("lgbIndex: %d\n", lgbIndex);
        printf("rgbIndex: %d\n", rgbIndex);
        printf("ldataColIndex: %d\n", ldataColIndex);
        printf("rdataColIndex: %d\n", rdataColIndex);
        
#endif

    } // end of contains groupBy keyword

    //FIXME: hard code for SSB q2_1 -- breakdown3 requires p_brand1 as one dim, comment out if not q2_1
/*    
    printf("ldata2:%d\n", ldata2);
    int update_M = getMaxVal(jNode->leftTable->content[0],jNode->leftTable->tupleNum, 4);
    printf("p_brand1 max: %d\n", update_M+1);
    MATRIX_M = update_M+1;

    quantizedScale = getMaxVal(jNode->leftTable->content[ldataColIndex],
            jNode->leftTable->tupleNum, jNode->leftOutputAttrType[0]);
*/
    
//    printf("quantizedScale: %d\n", quantizedScale);

    struct gpu_timer initStart, end2endStart;
    // read row data from column store
    char *gpu_fact, *gpu_dim;         
    char *gpu_ldata, *gpu_rdata;      // data values of left/right tables
    char *d_redMat;                   // dense matrix groupBy reduction
    half *d_redMatFp16;
    char *gpu_ldata2;
    int *d_bins;

    float alpha = 1.0f, beta = 0.0f;
    half *d_fp16_A, *d_fp16_BT;
    float *c_cublas;
    half *c_fp16_cublas;

    half alpha_fp16 = __float2half(1.0f);
    half beta_fp16 = __float2half(1.0f);
    float *c_host_cublas;

    cublasHandle_t cublasHandle;
    cudaEvent_t startcublasEX;
    cudaEvent_t stopcublasEX;

    cudaErrCheck(cudaEventCreate(&startcublasEX));
    cudaErrCheck(cudaEventCreate(&stopcublasEX));
    cublasErrCheck(cublasCreate(&cublasHandle));
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

    // allocate device memory for join keys
#ifdef CUSPARSE
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact,foreignKeySize));
    
    if (gb && gb->gbExp[gb->aggFuncIndex].func == COUNT) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));
    }
#else
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));
#endif    
    
    // data value - leftAggColIndex, groupBy attribute columnIndex - gbLeftColIndex
    if (gb && gbConstant != 1) 
    {
        if (gb->gbLeftColIndex != NULL) {
            // update left_gbWidth
            left_gbWidth = getMaxVal(jNode->leftTable->content[gbLeftIndex],
                                     leftTupleNum, 4) + 1;
#ifdef DEBUG
            printf("left_gbWidth: %d\n", left_gbWidth);
#endif
        }

        if (gb->leftAggNum == 1) 
        {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_ldata,foreignKeySize));
#ifdef DEBUG
            printf("cudaMalloc gpu_ldata column\n");
#endif
        } else if (gb->leftAggNum == 2) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_ldata,foreignKeySize));
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_ldata2,foreignKeySize));
#ifdef DEBUG
            printf("cudaMalloc gpu_ldata column\n");
            printf("cudaMalloc gpu_ldata2 column\n");
#endif
        }

        if (gb->gbRightColIndex != NULL) {
            // update right_gbWidth
            right_gbWidth = getMaxVal(jNode->rightTable->content[gbRightIndex],
                                      rightTupleNum, 4)+1;
#ifdef DEBUG
            printf("right_gbWidth: %d\n", right_gbWidth);
#endif
        }
        if (gb->rightAggNum == 1) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_rdata,primaryKeySize));
#ifdef DEBUG
            printf("cudaMalloc gpu_rdata column\n");
#endif
        }
    }

#ifdef CUSPARSE
    
#else
    // dense filling
    c_host_cublas = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_cublas,(uint64_t)MATRIX_M*(uint64_t)MATRIX_N*sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_A,(uint64_t)MATRIX_M*(uint64_t)MATRIX_K*sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_BT,(uint64_t)MATRIX_K*(uint64_t)MATRIX_N*sizeof(half)));
    
#endif

    initTime = initStart.milliseconds_elapsed();

    //FIXME: SSB q1-series sql
    //quantizedScale = getMaxVal(jNode->leftTable->content[0], jNode->leftTable->tupleNum, jNode->leftOutputAttrType[0]);
    
    struct gpu_timer cudaMemcpyStart;
#ifdef CUSPARSE
    if (gb && gb->gbExp[gb->aggFuncIndex].func == COUNT) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));
    } else {
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));
    }
#else
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));
#endif
    
    if (gb && gbConstant != 1) 
    {
        if (gb->leftAggNum == 1)
        {
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_ldata,jNode->leftTable->content[gb->leftAggColIndex[0]], foreignKeySize,cudaMemcpyHostToDevice));
#ifdef DEBUG
            printf("cudaMemcpy gpu_ldata\n");
#endif
        }

        // only for q2_1 test
        else if (gb->leftAggNum == 2)
        {
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_ldata,jNode->leftTable->content[gb->leftAggColIndex[0]], foreignKeySize,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_ldata2,jNode->leftTable->content[gb->leftAggColIndex[1]], foreignKeySize,cudaMemcpyHostToDevice));
#ifdef DEBUG
            printf("cudaMemcpy gpu_ldata\n");
            printf("cudaMemcpy gpu_ldata2\n");
#endif
        }

        if (gb->rightAggNum == 1)
        {
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_rdata,jNode->rightTable->content[gb->rightAggColIndex[0]], primaryKeySize,cudaMemcpyHostToDevice));
#ifdef DEBUG
            printf("cudaMemcpy gpu_rdata\n");
#endif
        }
    }

    cudaMemcpyTime = cudaMemcpyStart.milliseconds_elapsed();
    // TODO: determine whether to use 1) dense 2) cuSPARSE 3) blocked-GeMM
    // assume sparsity, A_nnz, B_nnz is given
        if (gb && gb->gbExp[gb->aggFuncIndex].func == SUM) 
        {
            if (gb->numFuncExpCol == 1) // only has one func keyword (groupBy not include) 
            {
                //if (rValIndex[0] == -1)
                if (gb->gbLeftColIndex && gb->gbRightColIndex) //groupBy left and right
                {
#ifdef CUSPARSE
                    tcuspmm_gbAB(Annz, A_num_rows, A_num_cols,
                                 Bnnz, B_num_rows, B_num_cols,
                                 MATRIX_K,
                                 leftTupleNum, jNode->leftTable->content[jNode->leftKeyIndex],
                                 jNode->leftTable->content[gb->leftAggColIndex[0]],
                                 left_gbWidth, jNode->leftTable->content[gbLeftIndex],
                                 right_gbWidth, jNode->rightTable->content[gbRightIndex],
                                 rightTupleNum, jNode->rightTable->content[jNode->rightKeyIndex], NULL);
#endif
                }
                else if (gb->gbLeftColIndex && !gb->gbRightColIndex)
                {
#ifdef CUSPARSE
            printf("SSB Q2_1b3\n");
            tcuspmm_gbA(Annz, A_num_rows, A_num_cols,
                        Bnnz, B_num_rows, B_num_cols,
                        MATRIX_K,
                        leftTupleNum, jNode->leftTable->content[jNode->leftKeyIndex], 
                        jNode->leftTable->content[gb->leftAggColIndex[0]],
                        left_gbWidth, jNode->leftTable->content[gbLeftIndex],
                        rightTupleNum, jNode->rightTable->content[jNode->rightKeyIndex],
                        NULL);

#else
                    struct gpu_timer fillStart;

                    if (ldata2 != -1) // specifically for Q2_1b3.sql
                    {
                        printf("calling gpu_fill_2data\n");
                        gpu_fill_2data<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
                                gpu_ldata,
                                gpu_ldata2,
                                MATRIX_K,
                                d_fp16_A,
                                leftTupleNum,
                                jNode->leftTable->attrType[jNode->leftKeyIndex],
                                quantizedScale);

                    } 
                    else 
                    {
                        gpu_fill_data<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
                        gpu_ldata,    
                        MATRIX_K,
                        d_fp16_A,
                        leftTupleNum,
                        jNode->leftTable->attrType[jNode->leftKeyIndex]);

                    }
                    cudaErrCheck(cudaFree(gpu_fact));
                    cudaErrCheck(cudaFree(gpu_ldata));
                
                    gpu_fill_gb_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                        gpu_dim,
                        gpu_rdata,
                        MATRIX_N,
                        d_fp16_BT,
                        rightTupleNum,
                        jNode->rightTable->attrType[jNode->rightKeyIndex]);
                    fillTime = fillStart.milliseconds_elapsed();
    
                    cudaErrCheck(cudaFree(gpu_dim));
                    cudaErrCheck(cudaFree(gpu_rdata));
    
                    printf("MxK:%dx%d\tKxN:%dx%d\n", MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N);
                    printf("Running GemmEx (Group-by aggregates) on TCUs...\n");
                    cudaErrCheck(cudaEventRecord(startcublasEX));
                    struct gpu_timer compute_start;
                    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            MATRIX_N, MATRIX_M, MATRIX_K,
                            &alpha,
                            d_fp16_BT, CUDA_R_16F, MATRIX_N,
                            d_fp16_A, CUDA_R_16F, MATRIX_K,
                            &beta,
                            c_cublas, CUDA_R_32F, MATRIX_N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    cudaErrCheck(cudaEventRecord(stopcublasEX)); 
                    tcu_compute_time = compute_start.milliseconds_elapsed();
                    cudaErrCheck(cudaFree(d_fp16_A));
                    cudaErrCheck(cudaFree(d_fp16_BT));

                    placeCount<<<(MATRIX_M*MATRIX_N+255)/256, 256>>> (c_cublas, c_cublas, MATRIX_M*MATRIX_N);
                    printf("Join Count: %lld\n", getCount(c_cublas, MATRIX_M*MATRIX_N));
                
                    if (gbConstant != 1) // perform groupBy reduction
                    {
                        struct gpu_timer groupBy_start;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMat, 1 * MATRIX_M * sizeof(char)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMatFp16, 1 * MATRIX_M * sizeof(half)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_redMat, 1, MATRIX_M * sizeof(char)));
                        convertCharToFp16 <<< (MATRIX_M + 255) / 256, 256 >>> (d_redMatFp16, 
                            d_redMat, MATRIX_M);
                        cudaErrCheck(cudaFree(d_redMat));
                    
                        // compute groupBy count by performing reduction
                        half *temp_c;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&temp_c, MATRIX_M * MATRIX_N * sizeof(half)));
                        
                        convertFp32ToFp16 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (temp_c, c_cublas, MATRIX_M * MATRIX_N);
                        cudaErrCheck(cudaFree(c_cublas));
            
                        float *d_reduction_res;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_reduction_res, 1 * MATRIX_N * sizeof(float)));
            
                        printf("Perform groupBy reduction...\n");
                        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            MATRIX_N, 1, MATRIX_M,
                            &alpha,
                            temp_c, CUDA_R_16F, MATRIX_N,
                            d_redMatFp16, CUDA_R_16F, MATRIX_M,
                            &beta,
                            d_reduction_res, CUDA_R_32F, MATRIX_N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            
                        cudaErrCheck(cudaFree(temp_c));
                        cudaErrCheck(cudaFree(d_redMatFp16));
            
                        int *d_gbCount;
                        int h_gbCount = 0;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_gbCount, 1 * sizeof(int)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_gbCount, 0, 1 * sizeof(int)));
                        
                        //groupByCount<<<(MATRIX_N+255), 256>>> (d_reduction_res, MATRIX_N, d_gbCount);
                        naiveCount<<<(MATRIX_N+255)/256, 256>>> (d_reduction_res, MATRIX_N, d_gbCount);
                        cudaErrCheck(cudaFree(d_reduction_res));
                        cudaErrCheck(cudaMemcpy(&h_gbCount, d_gbCount, 1 * sizeof(int), cudaMemcpyDeviceToHost));
                        tcu_groupBy_time = groupBy_start.milliseconds_elapsed();
                        cudaErrCheck(cudaFree(d_gbCount));
                        printf("GroupBy Count: %d\n", h_gbCount);
        
                    }
#endif
                } // groupBy left 
                else if (!gb->gbLeftColIndex && gb->gbRightColIndex)    
                {
#ifdef CUSPARSE
            tcuspmm_gbB(Annz, A_num_rows, A_num_cols,
                        Bnnz, B_num_rows, B_num_cols,
                        MATRIX_K, foreignKeySize,
                        leftTupleNum, gpu_fact, 
                        jNode->leftTable->content[gb->leftAggColIndex[0]],
                        right_gbWidth, jNode->rightTable->content[gbRightIndex],
                        rightTupleNum, jNode->rightTable->content[jNode->rightKeyIndex],
                        NULL);

#else
                    struct gpu_timer fillStart;
                    gpu_fill_gb<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                        gpu_fact,
                        gpu_ldata,    
                        MATRIX_K,
                        d_fp16_A,
                        leftTupleNum,
                        jNode->leftTable->attrType[jNode->leftKeyIndex]);
                    cudaErrCheck(cudaFree(gpu_fact));
                    cudaErrCheck(cudaFree(gpu_ldata));
    
                    gpu_fill_data_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                        gpu_dim,
                        gpu_rdata,
                        MATRIX_N,
                        d_fp16_BT,
                        rightTupleNum,
                        jNode->rightTable->attrType[jNode->rightKeyIndex]);
                    fillTime = fillStart.milliseconds_elapsed();

                    cudaErrCheck(cudaFree(gpu_dim));
                    cudaErrCheck(cudaFree(gpu_rdata));
    
                    printf("MxK:%dx%d\tKxN:%dx%d\n", MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N);
                    printf("Running GemmEx (Group-by aggregates) on TCUs...\n");
                    struct gpu_timer compute_start;
                    cudaErrCheck(cudaEventRecord(startcublasEX));
                    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        MATRIX_N, MATRIX_M, MATRIX_K,
                        &alpha,
                        d_fp16_BT, CUDA_R_16F, MATRIX_N,
                        d_fp16_A, CUDA_R_16F, MATRIX_K,
                        &beta,
                        c_cublas, CUDA_R_32F, MATRIX_N,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    cudaErrCheck(cudaEventRecord(stopcublasEX));
                    tcu_compute_time = compute_start.milliseconds_elapsed();
                    cudaErrCheck(cudaFree(d_fp16_A));
                    cudaErrCheck(cudaFree(d_fp16_BT));

                    if (gbConstant != 1) 
                    {
                        struct gpu_timer groupBy_start;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMat, 1 * MATRIX_N * sizeof(char)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMatFp16, 1 * MATRIX_N * sizeof(half)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_redMat, 1, MATRIX_N * sizeof(char)));
                        convertCharToFp16 <<< (MATRIX_N + 255) / 256, 256 >>> (d_redMatFp16, 
                            d_redMat, MATRIX_N);
                        cudaErrCheck(cudaFree(d_redMat));
                    
                        half *temp_c;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&temp_c, MATRIX_M * MATRIX_N * sizeof(half)));
                        
                        convertFp32ToFp16 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (temp_c, c_cublas, MATRIX_M * MATRIX_N);
                        cudaErrCheck(cudaFree(c_cublas));
            
                        float *d_reduction_res;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_reduction_res, MATRIX_M * 1 * sizeof(float)));
            
                        printf("Perform groupBy reduction...\n");
                        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                1, MATRIX_M, MATRIX_N,
                                &alpha,
                                d_redMatFp16, CUDA_R_16F, 1,
                                temp_c, CUDA_R_16F, MATRIX_N,
                                &beta,
                                d_reduction_res, CUDA_R_32F, 1,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            
                        cudaErrCheck(cudaFree(temp_c));
                        cudaErrCheck(cudaFree(d_redMatFp16));
            
                        int *d_gbCount; 
                        int h_gbCount = 0;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_gbCount, 1 * sizeof(int)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_gbCount, 0, 1 * sizeof(int)));
                        
                        //groupByCount<<<(MATRIX_M+255), 256>>> (d_reduction_res, MATRIX_M, d_gbCount);
                        naiveCount<<<(MATRIX_M+255)/256, 256>>> (d_reduction_res, MATRIX_M, d_gbCount);
                        cudaErrCheck(cudaFree(d_reduction_res));
                        cudaErrCheck(cudaMemcpy(&h_gbCount, d_gbCount, 1 * sizeof(int), cudaMemcpyDeviceToHost));
                        tcu_groupBy_time = groupBy_start.milliseconds_elapsed();
                        cudaErrCheck(cudaFree(d_gbCount));
                        printf("GroupBy Count: %d\n", h_gbCount);
                    }
#endif
                } // groupBy right
            } // end of gb->numFuncExpCol == 1
            // dense matrix-multiplication case
            else if (!jNode->rightOutputAttrNum && gb->numFuncExpCol == 2 && gb->math_op == MULTIPLY)
            {
#ifdef CUSPARSE
            printf("SSB Q1-series sql using cuSPARSE\n");
            tcuspmm(Annz, A_num_rows, A_num_cols,
                    Bnnz, B_num_rows, B_num_cols,
                    MATRIX_K, foreignKeySize,
                    leftTupleNum, gpu_fact, jNode->leftTable->content[0],
                    rightTupleNum, jNode->rightTable->content[jNode->rightKeyIndex],
                    jNode->rightTable->content[0]); 
#else

            printf("SSB Q1-series using dense-filling\n"); 
            struct gpu_timer fillStart;
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_ldata,foreignKeySize));
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_ldata2,foreignKeySize));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_ldata,jNode->leftTable->content[0], foreignKeySize,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_ldata2,jNode->leftTable->content[1], foreignKeySize,cudaMemcpyHostToDevice));
            
             // c_cublas will be out of memory
            gpu_fill_moredata<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
                gpu_ldata,    
                gpu_ldata2,
                MATRIX_K,
                d_fp16_A,
                leftTupleNum,
                jNode->leftTable->attrType[jNode->leftKeyIndex],
                quantizedScale);

            gpu_fill_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                gpu_dim,
                MATRIX_N,
                d_fp16_BT,
                rightTupleNum,
                jNode->rightTable->attrType[jNode->rightKeyIndex]);
            fillTime = fillStart.milliseconds_elapsed();
            
            printf("MxK:%dx%d\tKxN:%dx%d\n", MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N);
            printf("Running GemmEx on TCUs...\n");
            struct gpu_timer compute_start;
            cudaErrCheck(cudaEventRecord(startcublasEX));
            cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha,
                d_fp16_BT, CUDA_R_16F, MATRIX_N,
                d_fp16_A, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            cudaErrCheck(cudaEventRecord(stopcublasEX));
            tcu_compute_time = compute_start.milliseconds_elapsed();
            cudaErrCheck(cudaFree(d_fp16_A));
            cudaErrCheck(cudaFree(d_fp16_BT));

            placeCount<<<(MATRIX_M*MATRIX_N+255)/256, 256>>> (c_cublas, c_cublas, MATRIX_M*MATRIX_N);
            printf("Join Count: %lld\n", getCount(c_cublas, MATRIX_M*MATRIX_N));
#endif

            } 
            else if (gb->numFuncExpCol == 2 && gb->math_op == MINUS) // SSB Q4_1
            {
                    printf("SSB Q4_1\n");
                    struct gpu_timer fillStart;

                    if (ldata2 != -1) // specifically for Q2_1b3.sql
                    {
                        printf("calling gpu_fill_2data\n");
                        gpu_fill_2data<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
                                gpu_ldata,
                                gpu_ldata2,
                                MATRIX_K,
                                d_fp16_A,
                                leftTupleNum,
                                jNode->leftTable->attrType[jNode->leftKeyIndex],
                                quantizedScale);
                        // update MATIRX_M 

                    } 
                    else 
                    {
                        gpu_fill_data<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
                        gpu_ldata,    
                        MATRIX_K,
                        d_fp16_A,
                        leftTupleNum,
                        jNode->leftTable->attrType[jNode->leftKeyIndex]);

                    }
                    cudaErrCheck(cudaFree(gpu_fact));
                    cudaErrCheck(cudaFree(gpu_ldata));
                
                    gpu_fill_gb_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                        gpu_dim,
                        gpu_rdata,
                        MATRIX_N,
                        d_fp16_BT,
                        rightTupleNum,
                        jNode->rightTable->attrType[jNode->rightKeyIndex]);
                    fillTime = fillStart.milliseconds_elapsed();
    
                    cudaErrCheck(cudaFree(gpu_dim));
                    cudaErrCheck(cudaFree(gpu_rdata));
    
                    printf("MxK:%dx%d\tKxN:%dx%d\n", MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N);
                    printf("Running GemmEx (Group-by aggregates) on TCUs...\n");
                    cudaErrCheck(cudaEventRecord(startcublasEX));
                    struct gpu_timer compute_start;
                    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            MATRIX_N, MATRIX_M, MATRIX_K,
                            &alpha,
                            d_fp16_BT, CUDA_R_16F, MATRIX_N,
                            d_fp16_A, CUDA_R_16F, MATRIX_K,
                            &beta,
                            c_cublas, CUDA_R_32F, MATRIX_N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    cudaErrCheck(cudaEventRecord(stopcublasEX)); 
                    tcu_compute_time = compute_start.milliseconds_elapsed();
                    cudaErrCheck(cudaFree(d_fp16_A));
                    cudaErrCheck(cudaFree(d_fp16_BT));

                    placeCount<<<(MATRIX_M*MATRIX_N+255)/256, 256>>> (c_cublas, c_cublas, MATRIX_M*MATRIX_N);
                    printf("Join Count: %lld\n", getCount(c_cublas, MATRIX_M*MATRIX_N));
                
                    if (gbConstant != 1) // perform groupBy reduction
                    {
                        struct gpu_timer groupBy_start;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMat, 1 * MATRIX_M * sizeof(char)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_redMatFp16, 1 * MATRIX_M * sizeof(half)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_redMat, 1, MATRIX_M * sizeof(char)));
                        convertCharToFp16 <<< (MATRIX_M + 255) / 256, 256 >>> (d_redMatFp16, 
                            d_redMat, MATRIX_M);
                        cudaErrCheck(cudaFree(d_redMat));
                    
                        // compute groupBy count by performing reduction
                        half *temp_c;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&temp_c, MATRIX_M * MATRIX_N * sizeof(half)));
                        
                        convertFp32ToFp16 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (temp_c, c_cublas, MATRIX_M * MATRIX_N);
                        cudaErrCheck(cudaFree(c_cublas));
            
                        float *d_reduction_res;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_reduction_res, 1 * MATRIX_N * sizeof(float)));
            
                        printf("Perform groupBy reduction...\n");
                        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            MATRIX_N, 1, MATRIX_M,
                            &alpha,
                            temp_c, CUDA_R_16F, MATRIX_N,
                            d_redMatFp16, CUDA_R_16F, MATRIX_M,
                            &beta,
                            d_reduction_res, CUDA_R_32F, MATRIX_N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            
                        cudaErrCheck(cudaFree(temp_c));
                        cudaErrCheck(cudaFree(d_redMatFp16));
            
                        int *d_gbCount;
                        int h_gbCount = 0;
                        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_gbCount, 1 * sizeof(int)));
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_gbCount, 0, 1 * sizeof(int)));
                        
                        //groupByCount<<<(MATRIX_N+255), 256>>> (d_reduction_res, MATRIX_N, d_gbCount);
                        naiveCount<<<(MATRIX_N+255)/256, 256>>> (d_reduction_res, MATRIX_N, d_gbCount);
                        cudaErrCheck(cudaFree(d_reduction_res));
                        cudaErrCheck(cudaMemcpy(&h_gbCount, d_gbCount, 1 * sizeof(int), cudaMemcpyDeviceToHost));
                        tcu_groupBy_time = groupBy_start.milliseconds_elapsed();
                        cudaErrCheck(cudaFree(d_gbCount));
                        printf("GroupBy Count: %d\n", h_gbCount);
                    }
            }
            else if (gb->numFuncExpCol == 2 && gb->math_op == MULTIPLY) // can't differentiate sec3-q4 and ssb q1_1b1.sql 
            {
                //printf("sec3-q4\n");
                struct gpu_timer fillStart;
                gpu_fill_data<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (gpu_fact,
                    gpu_ldata,    
                    MATRIX_K,
                    d_fp16_A,
                    leftTupleNum,
                    jNode->leftTable->attrType[jNode->leftKeyIndex]);
    
                cudaErrCheck(cudaFree(gpu_fact));
                cudaErrCheck(cudaFree(gpu_ldata));
    
                gpu_fill_data_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                    gpu_dim,
                    gpu_rdata,
                    MATRIX_N,
                    d_fp16_BT,
                    rightTupleNum,
                    jNode->rightTable->attrType[jNode->rightKeyIndex]);
                fillTime = fillStart.milliseconds_elapsed();
                cudaErrCheck(cudaFree(gpu_dim));
                cudaErrCheck(cudaFree(gpu_rdata));
    
                printf("MxK:%dx%d\tKxN:%dx%d\n", MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N);
                printf("Running GemmEx (Aggregation) on TCUs...\n");
                struct gpu_timer compute_start;
                cudaErrCheck(cudaEventRecord(startcublasEX));
                cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    MATRIX_N, MATRIX_M, MATRIX_K,
                    &alpha,
                    d_fp16_BT, CUDA_R_16F, MATRIX_N,
                    d_fp16_A, CUDA_R_16F, MATRIX_K,
                    &beta,
                    c_cublas, CUDA_R_32F, MATRIX_N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                cudaErrCheck(cudaEventRecord(stopcublasEX));
                tcu_compute_time = compute_start.milliseconds_elapsed();
                cudaErrCheck(cudaFree(d_fp16_A));
                cudaErrCheck(cudaFree(d_fp16_BT));
    
#ifdef DEBUG
                float *testRes;
                testRes = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
                cudaErrCheck(cudaMemcpy(testRes, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
                print_matrix(testRes, MATRIX_M, MATRIX_N);
#endif
            } // end of SUM(A.Val * B.Val)
            else if (gb->numFuncExpCol == 2 && gb->math_op == DIVIDE) { // pagerank
#ifdef CUSPARSE
                tcuspmm_pr(Annz, A_num_rows, A_num_cols,
                    Bnnz, B_num_rows, B_num_cols,
                    MATRIX_K, foreignKeySize,
                    leftTupleNum, gpu_fact, jNode->leftTable->content[jNode->leftOutputIndex[0]],
                    rightTupleNum, jNode->rightTable->content[jNode->rightKeyIndex],
                    jNode->rightTable->content[jNode->rightOutputIndex[0]]);
#endif
            }
        } // end of func == SUM 
        else if (gb && gb->gbExp[gb->aggFuncIndex].func == COUNT) 
        {
            printf("PR Q1\n");
            struct gpu_timer cudaCountStart;
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_bins, MATRIX_M * sizeof(int)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_bins, 0, MATRIX_M * sizeof(int)));
            count<<<(BLOCK_SIZE+MATRIX_N-1)/BLOCK_SIZE, BLOCK_SIZE>>> (gpu_dim, d_bins, MATRIX_N);
            count_time = cudaCountStart.milliseconds_elapsed();
        } 
        else // simply return Join counts (general matrix-multiplication)
        {
            // using cuSPARSE -- tbl2coo->coo2csr
#ifdef CUSPARSE
            printf("SSB Q2_1b1 or two-way join cases\n");
            tcuspmm(Annz, A_num_rows, A_num_cols,
                    Bnnz, B_num_rows, B_num_cols,
                    MATRIX_K, foreignKeySize,
                    leftTupleNum, gpu_fact, jNode->rightTable->content[0],
                    rightTupleNum, jNode->rightTable->content[jNode->rightKeyIndex],
                    jNode->rightTable->content[0]); 

#else
            
            struct gpu_timer fillStart;
            gpu_fill<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                gpu_fact,
                MATRIX_K,
                d_fp16_A,
                leftTupleNum,
                jNode->leftTable->attrType[jNode->leftKeyIndex]);
            cudaErrCheck(cudaFree(gpu_fact));

            gpu_fill_transpose<<<(MAX_THREADS+rightTupleNum-1)/MAX_THREADS,MAX_THREADS>>> (
                gpu_dim,
                MATRIX_N,
                d_fp16_BT,
                rightTupleNum,
                jNode->rightTable->attrType[jNode->rightKeyIndex]);
            fillTime = fillStart.milliseconds_elapsed();

            cudaErrCheck(cudaFree(gpu_dim));
            cudaErrCheck(cudaFree(gpu_rdata));

            printf("MxK:%dx%d\tKxN:%dx%d\n", MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N);
            printf("Running GemmEx (2-way natural join) on TCUs...\n");
            cudaErrCheck(cudaEventRecord(startcublasEX));
            struct gpu_timer compute_start;
            cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha,
                d_fp16_BT, CUDA_R_16F, MATRIX_N,
                d_fp16_A, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            cudaErrCheck(cudaEventRecord(stopcublasEX));
            tcu_compute_time = compute_start.milliseconds_elapsed();
            cudaErrCheck(cudaFree(d_fp16_A));
            cudaErrCheck(cudaFree(d_fp16_BT));

            placeCount<<<(MATRIX_M*MATRIX_N+255)/256, 256>>> (c_cublas, c_cublas, MATRIX_M*MATRIX_N);
            printf("Join Count: %lld\n", getCount(c_cublas, MATRIX_M*MATRIX_N));
            
#endif
        }
    
    end2endTime = end2endStart.milliseconds_elapsed(); // contain API's cold-start cost

    float cublasEXTime;
#ifdef CUSPARSE

#else
    // deprecated, since cublasGemmEx may be called twice to perform reduction
    cudaErrCheck(cudaEventSynchronize(stopcublasEX));
    cudaErrCheck(cudaEventElapsedTime(&cublasEXTime, startcublasEX, stopcublasEX));
#endif

    printf("Initialization:   %f ms\n", initTime);
    printf("cudaMemcpy:       %f ms\n", cudaMemcpyTime);
    
    if (gb && gb->gbExp[gb->aggFuncIndex].func == COUNT) {
        printf("GPU count time: %f ms\n", count_time);
    }

#ifdef CUSPARSE
    // sparse-filling time metrics in tcuSpMM
#else
    // dense-filling time metrics
    printf("Matrices filling: %f ms\n", fillTime);
    printf("TCU compute time: %f ms\n", tcu_compute_time);
    if (gb && gbConstant != 1) {
        printf("TCU groupBy time: %f ms\n", tcu_groupBy_time);
    }
#endif
    // printf("End-to-end time:  %f ms\n", end2endTime);

    cudaDeviceReset();
    return res; // return intermediate results  
}
