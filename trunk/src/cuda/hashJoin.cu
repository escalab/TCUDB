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
#include "../include/hashJoin.h"
#include "../include/gpuCudaLib.h"
#include "scanImpl.cu"
#include <cuda_fp16.h>
#include <curand.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda;

// For wmma API, these must be multiples fo 16
#define MATRIX_M 16
#define MATRIX_N 16
#define MATRIX_K 16

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

/*
 * Count the number of dimension keys for each bucket.
 */

__global__ static void count_hash_num(char *dim, long  inNum,int *num,int hsize){
    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *)dim)[i]; // 0 1 1 2 2 -- mat2.i
        int hKey = joinKey & (hsize-1);
        atomicAdd(&(num[hKey]),1);
    }
}

/*
 * All the buckets are stored in a continues memory region.
 * The starting position of each bucket is stored in the psum array.
 * For star schema quereis, the size of fact table is usually much
 * larger than the size of the dimension table. In this case, hash probing is much more
 * time consuming than building hash table. By avoiding pointer, the efficiency of hash probing
 * can be improved.
 */

__global__ static void build_hash_table(char *dim, long inNum, int *psum, char * bucket,int hsize){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *) dim)[i]; 
        int hKey = joinKey & (hsize-1);
        int pos = atomicAdd(&psum[hKey],1) * 2;
        ((int*)bucket)[pos] = joinKey;
        pos += 1;
        int dimId = i+1;
        ((int*)bucket)[pos] = dimId;
    }

}

/*
 * Count join result for each thread for dictionary encoded column. 
 */
__global__ static void count_join_result_dict(int *num, int* psum, char* bucket, struct dictHeader *dheader, int dNum, int* dictFilter,int hsize){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<dNum;i+=stride){
        int fkey = dheader->hash[i];
        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int fvalue = 0;

        for(int j=0;j<keyNum;j++){
            int pSum = psum[hkey];
            int dimKey = ((int *)(bucket))[2*j + 2*pSum];
            if( dimKey == fkey){
                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                fvalue = dimId;
                break;
            }
        }

        dictFilter[i] = fvalue;
    }

}

/*
 * Transform the dictionary filter to the final filter than can be used to generate the result
 */

__global__ static void transform_dict_filter(int * dictFilter, char *fact, long tupleNum, int dNum,  int * filter){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    struct dictHeader *dheader;
    dheader = (struct dictHeader *) fact;

    int byteNum = dheader->bitNum/8;
    int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int)  ; 

    for(long i=offset; i<numInt; i += stride){
        int tmp = ((int *)(fact + sizeof(struct dictHeader)))[i];

        for(int j=0; j< sizeof(int)/byteNum; j++){
            int fkey = 0;
            memcpy(&fkey, ((char *)&tmp) + j*byteNum, byteNum);

            filter[i* sizeof(int)/byteNum + j] = dictFilter[fkey];
        }
    }
}

/*
 * count the number that is not zero in the filter
 */
__global__ static void filter_count(long tupleNum, int * count, int * factFilter){

    int lcount = 0;
    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(long i=offset; i<tupleNum; i+=stride){
        if(factFilter[i] !=0)
            lcount ++;
    }
    count[offset] = lcount;
}

/*
 * count join result for rle-compressed key.
 */

__global__ static void count_join_result_rle(int* num, int* psum, char* bucket, char* fact, long tupleNum, int * factFilter,int hsize){

    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    struct rleHeader *rheader = (struct rleHeader *)fact;
    int dNum = rheader->dictNum;

    for(int i=offset; i<dNum; i += stride){
        int fkey = ((int *)(fact+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int pSum = psum[hkey];

        for(int j=0;j<keyNum;j++){

            int dimKey = ((int *)(bucket))[2*j + 2*pSum];

            if( dimKey == fkey){

                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                for(int k=0;k<fcount;k++)
                    factFilter[fpos+k] = dimId;

                break;
            }
        }
    }

}

/* Use 1D array to represent 2D factFilter so that it can store more than one match tuples */
__global__ static void count_join_result2(int* num, int* psum, char* bucket, char* fact, long inNum, int* count, int * factFilter, int * newFactFilter,int hsize, int right_tupleNum) {
    int lcount = 0;
    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for (int i = offset; i < inNum; i += stride) {
        int fkey = ((int *)(fact))[i];
        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int fvalue = 0;

        for (int j = 0; j < keyNum; j++) { 
            int pSum = psum[hkey];
            int dimKey = ((int *)(bucket))[2*j + 2*pSum];

            // NOTE: i -- left table row id; dimId -- right table row id
            if (dimKey == fkey) { // matched tuple
                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                dimId = dimId-1; // dirty fix for now, build_hash_table increase dimId by 1

                lcount ++;
                fvalue = dimId;
                newFactFilter[i*right_tupleNum+fvalue] = 1;
                //printf("FactFilter index: %d\tleft: %d\tright: %d\n", (i*right_tupleNum+fvalue),i,dimId);
            }
        }

        factFilter[i] = fvalue; // orig code
        // values are used by joinFact_int function
    }

    count[offset] = lcount;
}


/*
 * Count join result for uncompressed column
 */
// gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum, gpu_count,gpuFactFilter,hsize
__global__ static void count_join_result(int* num, int* psum, char* bucket, char* fact, long inNum, int* count, int * factFilter,int hsize){
    int lcount = 0;
    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int fkey = ((int *)(fact))[i]; // fact table match condition 
        int hkey = fkey &(hsize-1); 
        int keyNum = num[hkey];
        int fvalue = 0;

        for(int j=0;j<keyNum;j++){
            int pSum = psum[hkey];
            int dimKey = ((int *)(bucket))[2*j + 2*pSum];
            
            if( dimKey == fkey){
                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];

                lcount ++;
                fvalue = dimId;
                break; // cannot support multi-matched tuples
            }
        }
        factFilter[i] = fvalue;
    }
    count[offset] = lcount;
}

/*
 * Unpact the rle-compressed data
 */

__global__ void static unpack_rle(char * fact, char * rle, long tupleNum,int dNum){

    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<dNum; i+=stride){

        int fvalue = ((int *)(fact+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

        for(int k=0;k<fcount;k++){
            ((int*)rle)[fpos + k] = fvalue;
        }
    }
}

/*
 * generate psum for RLE compressed column based on filter
 * current implementaton: scan through rle element and find the correponsding element in the filter
 */

__global__ void static rle_psum(int *count, char * fact,  long  tupleNum, int * filter){

    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    struct rleHeader *rheader = (struct rleHeader *) fact;
    int dNum = rheader->dictNum;

    for(int i= offset; i<dNum; i+= stride){

        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];
        int lcount= 0;

        for(int k=0;k<fcount;k++){
            if(filter[fpos + k]!=0)
                lcount++;
        }
        count[i] = lcount;
    }

}

/*
 * filter the column that is compressed using Run Length Encoding
 */

__global__ void static joinFact_rle(int *resPsum, char * fact,  int attrSize, long  tupleNum, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    struct rleHeader *rheader = (struct rleHeader *) fact;
    int dNum = rheader->dictNum;

    for(int i = startIndex; i<dNum; i += stride){
        int fkey = ((int *)(fact+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

        int toffset = resPsum[i];
        for(int j=0;j<fcount;j++){
            if(filter[fpos-j] !=0){
                ((int*)result)[toffset] = fkey ;
                toffset ++;
            }
        }
    }

}

/*
 * filter the column in the fact table that is compressed using dictionary encoding
 */
__global__ void static joinFact_dict_other(int *resPsum, char * fact,  struct dictHeader *dheader, int byteNum,int attrSize, long  num, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            int key = 0;
            memcpy(&key, fact + sizeof(struct dictHeader) + i* byteNum, byteNum);
            memcpy(result + localOffset, &dheader->hash[key], attrSize);
            localOffset += attrSize;
        }
    }
}

__global__ void static joinFact_dict_int(int *resPsum, char * fact, struct dictHeader *dheader, int byteNum, int attrSize, long  num, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            int key = 0;
            memcpy(&key, fact + sizeof(struct dictHeader) + i* byteNum, byteNum);
            ((int*)result)[localCount] = dheader->hash[key];
            localCount ++;
        }
    }
}

__global__ void static joinFact_other(int *resPsum, char * fact,  int attrSize, long  num, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            memcpy(result + localOffset, fact + i*attrSize, attrSize);
            localOffset += attrSize;
        }
    }
}

// fact table -> left table
__global__ void static joinFact_int(int *resPsum, char * fact,  int attrSize, long  num, int * filter, char * result, int right_tupleNum){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for (int i = startIndex; i < (int) num; i += stride) {
        for (int j = 0; j < right_tupleNum; j++) {
            if (filter[i*right_tupleNum+j] != 0) {
                ((int*)result)[localCount] = ((int *)fact)[i];
                //printf("fact_row#: %d\tdim_row#: %d\tfact_val: %d\n", i, j, ((int *)fact)[i]);
                localCount++;

            }
        }
    }

    /*
    // orig code: assume each row only match once
    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            ((int*)result)[localCount] = ((int *)fact)[i];
            localCount ++;
        }
    }*/
}

__global__ void static joinDim_rle(int *resPsum, char * dim, int attrSize, long tupleNum, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    struct rleHeader *rheader = (struct rleHeader *) dim;
    int dNum = rheader->dictNum;

    for(int i = startIndex; i<tupleNum; i += stride){
        int dimId = filter[i];
        if(dimId != 0){
            for(int j=0;j<dNum;j++){
                int dkey = ((int *)(dim+sizeof(struct rleHeader)))[j];
                int dcount = ((int *)(dim+sizeof(struct rleHeader)))[j + dNum];
                int dpos = ((int *)(dim+sizeof(struct rleHeader)))[j + 2*dNum];

                if(dpos == dimId || ((dpos < dimId) && (dpos + dcount) > dimId)){
                    ((int*)result)[localCount] = dkey ;
                    localCount ++;
                    break;
                }

            }
        }
    }
}

__global__ void static joinDim_dict_other(int *resPsum, char * dim, struct dictHeader *dheader, int byteNum, int attrSize, long num,int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        if( dimId != 0){
            int key = 0;
            memcpy(&key, dim + sizeof(struct dictHeader)+(dimId-1) * byteNum, byteNum);
            memcpy(result + localOffset, &dheader->hash[key], attrSize);
            localOffset += attrSize;
        }
    }
}

__global__ void static joinDim_dict_int(int *resPsum, char * dim, struct dictHeader *dheader, int byteNum, int attrSize, long num,int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        if( dimId != 0){
            int key = 0;
            memcpy(&key, dim + sizeof(struct dictHeader)+(dimId-1) * byteNum, byteNum);
            ((int*)result)[localCount] = dheader->hash[key];
            localCount ++;
        }
    }
}

// dimension table -> right table
__global__ void static joinDim_int(int *resPsum, char * dim, int attrSize, long num,int * filter, char * result, int right_tupleNum){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for(int i = startIndex; i < (int) num; i += stride) {
        for (int j = 0; j < right_tupleNum; j++) {
            if (filter[i*right_tupleNum+j] != 0) {
                ((int*)result)[localCount] = ((int*)dim)[j];
                //printf("fact_row#: %d\tdim_row#: %d\tdim_val: %d\n", i, j, ((int *)dim)[j]);
                localCount++;
            }
        }
    }
    /*
    // orig code   
    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        //printf("dimId: %d\n", dimId);
        if( dimId != 0){
            ((int*)result)[localCount] = ((int*)dim)[dimId-1];
            printf(" <= joinDim_int val: \t\t%d\n", ((int *)dim)[dimId-1]);
            localCount ++;
        }
    }*/
}

__global__ void static joinDim_other(int *resPsum, char * dim, int attrSize, long num,int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        if( dimId != 0){
            memcpy(result + localOffset, dim + (dimId-1)* attrSize, attrSize);
            localOffset += attrSize;
        }
    }
}

/* Map the table entires into matrix for tensor core to use 
 * Assum both matrix have the same dimension and the value in INT type for now, e.g., both 16x16 dim
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
        int k = 0; // k is row-index of the table
        
        for (j = 0; j < leftTupleNum * attr_type1; j+=attr_type1) {
            
            if (left_col_idx == 0) { // match to schema's i
                mat1_i[k] = jNode->leftTable->content[i][j];
            }
            else if (left_col_idx == 1) { // match to schema's j
                mat1_j[k] = jNode->leftTable->content[i][j];
            }
            else { // match to schema's val
                // read 4 bytes at once because the type is int
                int * tmp;
                tmp = (int*)(&jNode->leftTable->content[i][j]);
                mat1_val[k] = *tmp;
            }
            k++;
        }
    }

    
    for (i = 0; i < attr_num2; i++) {
        int right_col_idx = jNode->rightTable->attrIndex[i];
        int k = 0;
        
        for (j = 0; j < rightTupleNum * attr_type2; j+=attr_type2) {
            
            if (right_col_idx == 0) {
                mat2_i[k] = jNode->rightTable->content[i][j];
            }
            else if (right_col_idx == 1) {
                mat2_j[k] = jNode->rightTable->content[i][j];
            }
            else {
                int * tmp;
                tmp = (int*)(&jNode->rightTable->content[i][j]);
                mat2_val[k] = *tmp;
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
   1) Matrices are packed in memory.
   2) M, N and K are multiples of 16.
   3) Neither A nor B are transposed.

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

// TODO: implement TCU join and time the elapse
__global__ void static tcu_join() {


}

/*
 * hashJoin implements the foreign key join between a fact table and dimension table.
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
struct tableNode * hashJoin(struct joinNode *jNode, struct statistic *pp){

    struct timespec start,end;
    clock_gettime(CLOCK_REALTIME,&start);

    struct tableNode * res = NULL;

    char *gpu_result = NULL, *gpu_bucket = NULL, *gpu_fact = NULL, *gpu_dim = NULL;
    int *gpu_count = NULL,  *gpu_psum = NULL, *gpu_resPsum = NULL, *gpu_hashNum = NULL;

    int defaultBlock = 4096;

    dim3 grid(defaultBlock);
    dim3 block(256);

    int blockNum;
    int threadNum;

    blockNum = jNode->leftTable->tupleNum / block.x + 1;
    if(blockNum < defaultBlock)
        grid = blockNum;
    else
        grid = defaultBlock;

/*
    // For WMMA
    dim3 gridDim;
    dim3 blockDim;
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
*/

    res = (struct tableNode*) malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);
    // get data from jNode tableNode
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

    for(int i=0;i<jNode->leftOutputAttrNum;i++){
        int pos = jNode->leftPos[i];
        res->attrType[pos] = jNode->leftOutputAttrType[i];
        int index = jNode->leftOutputIndex[i];
        res->attrSize[pos] = jNode->leftTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    for(int i=0;i<jNode->rightOutputAttrNum;i++){
        int pos = jNode->rightPos[i];
        res->attrType[pos] = jNode->rightOutputAttrType[i];
        int index = jNode->rightOutputIndex[i];
        res->attrSize[pos] = jNode->rightTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    /*
    printf("left attrType: %d\n", jNode->leftTable->attrType[0]);              // 4
    printf("right attrType: %d\n", jNode->rightTable->attrType[0]);            // 4
    printf("left attrSize: %d\n", jNode->leftTable->attrSize[0]);              // 4
    printf("right attrSize: %d\n", jNode->rightTable->attrSize[0]);            // 4
    printf("left attrTotalSize: %d\n", jNode->leftTable->attrTotalSize[0]);    // 20
    printf("right attrTotalSize: %d\n", jNode->rightTable->attrTotalSize[0]);  // 24
    printf("left totalAttr: %d\n", jNode->leftTable->totalAttr);               // 3
    printf("right totalAttr: %d\n", jNode->rightTable->totalAttr);             // 3
    */

    long primaryKeySize = sizeof(int) * jNode->rightTable->tupleNum;

/*
 *  build hash table on GPU
 */
    
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
    float *c_cublas;
    float *c_sgemm; // single precision

    // for error checking
    float *c_host_wmma;
    float *c_host_cublas;
    float *c_host_sgemm;

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

    mat1_fp16 = (half*)malloc(sizeof(half) * MATRIX_M * MATRIX_K);
    mat2_fp16 = (half*)malloc(sizeof(half) * MATRIX_K * MATRIX_N);
    mat1_fp32 = (float*)malloc(sizeof(float) * MATRIX_M * MATRIX_K);
    mat2_fp32 = (float*)malloc(sizeof(float) * MATRIX_K * MATRIX_N);

    //curandGenerator_t gen;
    cublasHandle_t cublasHandle;         // tcu
    cublasHandle_t cublasHandle_default; // cublas

    //curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 2020ULL));

    cudaEvent_t startWMMA;
    cudaEvent_t stopWMMA;
    cudaEvent_t startcublasEX;
    cudaEvent_t stopcublasEX;
    cudaEvent_t startcublas; // for sgemm
    cudaEvent_t stopcublas;

    cudaErrCheck(cudaEventCreate(&startWMMA));
    cudaErrCheck(cudaEventCreate(&stopWMMA)); 
    cudaErrCheck(cudaEventCreate(&startcublasEX));
    cudaErrCheck(cudaEventCreate(&stopcublasEX));
    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));

    // use tensor core or cublas
    cublasErrCheck(cublasCreate(&cublasHandle));
    cublasErrCheck(cublasCreate(&cublasHandle_default));

    // enable tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
    cublasErrCheck(cublasSetMathMode(cublasHandle_default, CUBLAS_DEFAULT_MATH));

    c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    c_host_sgemm = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    clock_gettime(CLOCK_REALTIME, &init_end);

    // fill matrices from jNode by mapping inputs into 1D array
    struct timespec fill_start, fill_end, convert_start, convert_end;
    struct timespec cuMemcpy_start, cuMemcpy_end;
    clock_gettime(CLOCK_REALTIME, &fill_start);
    fill_matrix(jNode, matrix1, matrix2, MATRIX_M, 
            jNode->leftTable->totalAttr, jNode->rightTable->totalAttr, jNode->leftTable->attrType[0], jNode->rightTable->attrType[0]);
    clock_gettime(CLOCK_REALTIME, &fill_end);
    //DEBUG: print matrix here

    clock_gettime(CLOCK_REALTIME, &convert_start);
    // convert to half type for wmma API
    convertIntToFp32(mat1_fp32, matrix1, MATRIX_M);
    convertIntToFp32(mat2_fp32, matrix2, MATRIX_N);
    //convertFp32ToFp16<<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (mat1_fp16, mat1_fp32, MATRIX_M * MATRIX_K);
    //convertFp32ToFp16<<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (mat2_fp16, mat2_fp32, MATRIX_K * MATRIX_N);
    //convertFp32ToFp16(mat1_fp16, mat1_fp32, MATRIX_M);
    //convertFp32ToFp16(mat2_fp16, mat2_fp32, MATRIX_N);
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
    //cudaErrCheck(cudaMemset(c, 0, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));
    //cudaErrCheck(cudaMemset(c_wmma, 0, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
    //cudaErrCheck(cudaMemset(c_cublas, 0, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&c_sgemm, MATRIX_M * MATRIX_N * sizeof(float)));
    //cudaErrCheck(cudaMemset(c_sgemm, 0, MATRIX_M * MATRIX_N * sizeof(float)));

    //c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    //c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    //c_host_sgemm = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    //cudaErrCheck(cudaMemcpy(mat1_dev, mat1_fp16, sizeof(half) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
    //cudaErrCheck(cudaMemcpy(mat2_dev, mat2_fp16, sizeof(half) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(mat1_dev_fp32, mat1_fp32, sizeof(float) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(mat2_dev_fp32, mat2_fp32, sizeof(float) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));
    convertFp32ToFp16<<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (mat1_dev, mat1_dev_fp32, MATRIX_M * MATRIX_K);
    convertFp32ToFp16<<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (mat2_dev, mat2_dev_fp32, MATRIX_K * MATRIX_N);
    cudaErrCheck(cudaMemcpy(mat1_fp16, mat1_dev, sizeof(half) * MATRIX_M * MATRIX_K, cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(mat2_fp16, mat2_dev, sizeof(half) * MATRIX_K * MATRIX_N, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);

    //diff_mat(mat1_fp32, mat1_fp16, MATRIX_M);
    //printf("\n");
    //diff_mat(mat2_fp32, mat2_fp16, MATRIX_N);
    
    //printf("Matrix 1 Fp16:\n");
    //print_matrix(mat1_fp16, MATRIX_M);
    //printf("Matrix 2 Fp16:\n");
    //print_matrix(mat2_fp16, MATRIX_M);
    

    // create data for result matrix directly on device
    //curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
    //curandErrCheck(curandDestroyGenerator(gen));

    //cudaErrCheck(cudaMemcpy(c_wmma, c, sizeof(float) * MATRIX_M * MATRIX_N, cudaMemcpyDeviceToDevice));
    //cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
    //cudaErrCheck(cudaMemcpy(c_sgemm, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

    printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

    printf("Running with wmma...\n");
    cudaErrCheck(cudaEventRecord(startWMMA));
    wmma_example <<< gridDim, blockDim >>> (mat1_dev, mat2_dev, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta); 
    cudaErrCheck(cudaEventRecord(stopWMMA));

    printf("Running with sgemm...\n");
    cudaErrCheck(cudaEventRecord(startcublas));
    cublasSgemm(cublasHandle_default, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, mat1_dev_fp32, MATRIX_M, mat2_dev_fp32, MATRIX_N, &beta, c_sgemm, MATRIX_K);
    cudaErrCheck(cudaEventRecord(stopcublas));

    printf("Running with cuBLAS on TCUs...\n");
    cudaErrCheck(cudaEventRecord(startcublasEX));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha,
                mat1_dev, CUDA_R_16F, MATRIX_M,
                mat2_dev, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
    cudaErrCheck(cudaEventRecord(stopcublasEX));

    struct timespec chkRes_start, chkRes_end;
    clock_gettime(CLOCK_REALTIME,&chkRes_start);
    // Copy result back to the host for error checking
    cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(c_host_sgemm, c_sgemm, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

    // print error checking, cublasGemmEx and cublas
    printf("\nChecking results with tensor cores...\n");
    // 0.01% relative tolerance. 1e-5 absolute tolerance.
    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = c_host_sgemm[i];
        float v2 = c_host_cublas[i];

        // TODO: abs diff failed
        if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-3) {
            errors++;
            if (errors < 10) printf("%.1f %.1f diff:%.1f\n", v1, v2, abs(v1-v2));
        }
    }

    if (errors > 0) {
        printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
    }
    clock_gettime(CLOCK_REALTIME,&chkRes_end);

    // print time
    float wmmaTime;
    float cublasTime;

    cudaErrCheck(cudaEventSynchronize(stopWMMA));
    cudaErrCheck(cudaEventSynchronize(stopcublasEX));
    cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));

    printf("wmma took %fms\n", wmmaTime);
    printf("cublas (FP32) took %fms\n", cublasTime);
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublasEX, stopcublasEX));
    printf("cublas tensor cores (FP16) took %fms\n", cublasTime);

    // free those data structures
    cudaErrCheck(cudaEventDestroy(startWMMA));
    cudaErrCheck(cudaEventDestroy(stopWMMA));
    cudaErrCheck(cudaEventDestroy(startcublasEX));
    cudaErrCheck(cudaEventDestroy(stopcublasEX));

    cudaErrCheck(cudaFree(mat1_dev));
    cudaErrCheck(cudaFree(mat2_dev));
    cudaErrCheck(cudaFree(c));
    cudaErrCheck(cudaFree(c_wmma));
    cudaErrCheck(cudaFree(c_cublas));
    cudaErrCheck(cudaFree(c_sgemm));

    free(matrix1);
    free(matrix2);
    free(mat1_fp16);
    free(mat2_fp16);
    free(mat1_fp32);
    free(mat2_fp32);
    free(c_host_wmma);
    free(c_host_cublas);
    free(c_host_sgemm);

    clock_gettime(CLOCK_REALTIME, &tcu_end);
    double tcu_fill = (fill_end.tv_sec -  fill_start.tv_sec)* BILLION + fill_end.tv_nsec - fill_start.tv_nsec;
    double tcu_convert = (convert_end.tv_sec -  convert_start.tv_sec)* BILLION + convert_end.tv_nsec - convert_start.tv_nsec;
    double tcu_elapse = (tcu_end.tv_sec -  tcu_start.tv_sec)* BILLION + tcu_end.tv_nsec - tcu_start.tv_nsec;
    double init_elapse = (init_end.tv_sec -  init_start.tv_sec)* BILLION + init_end.tv_nsec - init_start.tv_nsec;
    double cuMemcpy_elapse = (cuMemcpy_end.tv_sec -  cuMemcpy_start.tv_sec)* BILLION + cuMemcpy_end.tv_nsec - cuMemcpy_start.tv_nsec;
    double chkRes_elapse = (chkRes_end.tv_sec -  chkRes_start.tv_sec)* BILLION + chkRes_end.tv_nsec - chkRes_start.tv_nsec;
    
    printf("Time to initialize: %lf\n", init_elapse/(1000*1000));
    printf("Time to fill matrices: %lf\n", tcu_fill/(1000*1000));
    printf("Time to convert data type: %lf\n", tcu_convert/(1000*1000));
    printf("Time for cudaMemcpy: %lf\n", cuMemcpy_elapse/(1000*1000));
    printf("Time to check result: %lf\n", chkRes_elapse/(1000*1000));
    printf("TCU overall MatMul Time: %lf\n", tcu_elapse/(1000*1000));

    // YDB hashJoin below this point

    int *gpu_psum1 = NULL;

    int hsize = jNode->rightTable->tupleNum;
    //printf("hsize: %d\n", hsize); // hsize seems to be used for count_group_num
    NP2(hsize);

    //printf("after NP2 function hsize: %d\n", hsize); // 8
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_hashNum,sizeof(int)*hsize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_hashNum,0,sizeof(int)*hsize));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_psum,hsize*sizeof(int)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_bucket, 2*primaryKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_psum1,hsize*sizeof(int)));

    int dataPos = jNode->rightTable->dataPos[jNode->rightKeyIndex];

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));

    }else if (dataPos == GPU || dataPos == UVA){
        gpu_dim = jNode->rightTable->content[jNode->rightKeyIndex];
    }

    count_hash_num<<<grid,block>>>(gpu_dim,jNode->rightTable->tupleNum,gpu_hashNum,hsize);
    scanImpl(gpu_hashNum,hsize,gpu_psum, pp);

    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_psum1,gpu_psum,sizeof(int)*hsize,cudaMemcpyDeviceToDevice));

    build_hash_table<<<grid,block>>>(gpu_dim,jNode->rightTable->tupleNum,gpu_psum1,gpu_bucket,hsize);

    if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED)
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_dim));

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum1));

/*
 *  join on GPU
 */
    //TODO: TCU join calling point

    threadNum = grid.x * block.x;

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_count,sizeof(int)*threadNum));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_resPsum,sizeof(int)*threadNum));

    int *gpuFactFilter = NULL;
    int *newFactFilter = NULL; // a[x][y] -> a[x*dim_size+y], x: left table's row id, dim_size: right table's tupleNum, y: right table's row id

    dataPos = jNode->leftTable->dataPos[jNode->leftKeyIndex];
    int format = jNode->leftTable->dataFormat[jNode->leftKeyIndex];

    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
    long filterSize = jNode->leftTable->attrSize[jNode->leftKeyIndex] * jNode->leftTable->tupleNum;
    long newSize = jNode->rightTable->tupleNum * jNode->leftTable->tupleNum; // worst case is all matched
    int right_tupleNum = jNode->rightTable->tupleNum;

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact, foreignKeySize));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));

    }else if (dataPos == GPU || dataPos == UVA){
        gpu_fact = jNode->leftTable->content[jNode->leftKeyIndex];
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFactFilter,filterSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuFactFilter,0,filterSize));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&newFactFilter,newSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(newFactFilter,0,newSize));

    if(format == UNCOMPRESSED) {
        count_join_result2<<<grid,block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum, gpu_count,gpuFactFilter,newFactFilter,hsize, right_tupleNum);
        //count_join_result<<<grid,block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum, gpu_count,gpuFactFilter,hsize);
    }
    else if(format == DICT){
        int dNum;
        struct dictHeader * dheader = NULL;
        struct dictHeader * gpuDictHeader = NULL;

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuDictHeader, sizeof(struct dictHeader)));

        if(dataPos == MEM || dataPos == MMAP || dataPos == UVA || dataPos == PINNED){
            dheader = (struct dictHeader *) jNode->leftTable->content[jNode->leftKeyIndex];
            dNum = dheader->dictNum;
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));

        }else if (dataPos == GPU){
            dheader = (struct dictHeader *) malloc(sizeof(struct dictHeader));
            memset(dheader,0,sizeof(struct dictHeader));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(dheader,gpu_fact,sizeof(struct dictHeader), cudaMemcpyDeviceToHost));
            dNum = dheader->dictNum;
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));
            free(dheader);
        }

        int * gpuDictFilter;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

        count_join_result_dict<<<grid,block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpuDictHeader, dNum, gpuDictFilter,hsize);

        transform_dict_filter<<<grid,block>>>(gpuDictFilter, gpu_fact, jNode->leftTable->tupleNum, dNum, gpuFactFilter);

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

        filter_count<<<grid,block>>>(jNode->leftTable->tupleNum, gpu_count, gpuFactFilter);

    }else if (format == RLE){

        count_join_result_rle<<<512,64>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum, gpuFactFilter,hsize);

        filter_count<<<grid, block>>>(jNode->leftTable->tupleNum, gpu_count, gpuFactFilter);
    }

    int tmp1, tmp2;

    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp1,&gpu_count[threadNum-1],sizeof(int),cudaMemcpyDeviceToHost));
    scanImpl(gpu_count,threadNum,gpu_resPsum, pp);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp2,&gpu_resPsum[threadNum-1],sizeof(int),cudaMemcpyDeviceToHost));

    res->tupleNum = tmp1 + tmp2;
    printf("[INFO]Number of join results: %d\n",res->tupleNum);

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_bucket));

    for(int i=0; i<res->totalAttr; i++){

        int index, pos;
        long colSize = 0, resSize = 0;
        int leftRight = 0;

        int attrSize, attrType;
        char * table = NULL;
        int found = 0 , dataPos, format;

        if (jNode->keepInGpu[i] == 1)
            res->dataPos[i] = GPU;
        else
            res->dataPos[i] = MEM;

        for(int k=0;k<jNode->leftOutputAttrNum;k++){
            if (jNode->leftPos[k] == i){
                found = 1;
                leftRight = 0;
                pos = k;
                break;
            }
        }
        if(!found){
            for(int k=0;k<jNode->rightOutputAttrNum;k++){
                if(jNode->rightPos[k] == i){
                    found = 1;
                    leftRight = 1;
                    pos = k;
                    break;
                }
            }
        }

        if(leftRight == 0){
            index = jNode->leftOutputIndex[pos]; // 0
            dataPos = jNode->leftTable->dataPos[index];
            format = jNode->leftTable->dataFormat[index];

            table = jNode->leftTable->content[index];
            attrSize  = jNode->leftTable->attrSize[index];
            attrType  = jNode->leftTable->attrType[index];
            colSize = jNode->leftTable->attrTotalSize[index];

            resSize = res->tupleNum * attrSize;
        }else{
            index = jNode->rightOutputIndex[pos]; // 1
            dataPos = jNode->rightTable->dataPos[index];
            format = jNode->rightTable->dataFormat[index];

            table = jNode->rightTable->content[index];
            attrSize = jNode->rightTable->attrSize[index];
            attrType = jNode->rightTable->attrType[index];
            colSize = jNode->rightTable->attrTotalSize[index];

            resSize = attrSize * res->tupleNum;
            leftRight = 1;
        }


        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_result,resSize));

        if(leftRight == 0){
            if(format == UNCOMPRESSED){
                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                if(attrSize == sizeof(int))
                    // TODO: or call TCU join here
                    joinFact_int<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,newFactFilter,gpu_result, right_tupleNum);
                    //joinFact_int<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);
                else
                    joinFact_other<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

            }else if (format == DICT){
                struct dictHeader * dheader = NULL;
                int byteNum;
                struct dictHeader * gpuDictHeader = NULL;

                dheader = (struct dictHeader *)table;
                byteNum = dheader->bitNum/8;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuDictHeader,sizeof(struct dictHeader)));
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                if (attrSize == sizeof(int))
                    joinFact_dict_int<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);
                else
                    joinFact_dict_other<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

            }else if (format == RLE){

                struct rleHeader* rheader = NULL;

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                rheader = (struct rleHeader*)table;

                int dNum = rheader->dictNum;

                char * gpuRle = NULL;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRle, jNode->leftTable->tupleNum * sizeof(int)));

                unpack_rle<<<grid,block>>>(gpu_fact, gpuRle,jNode->leftTable->tupleNum, dNum);

                joinFact_int<<<grid,block>>>(gpu_resPsum,gpuRle, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result, newSize);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));

            }

        }else{
            if(format == UNCOMPRESSED){

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                if(attrType == sizeof(int))
                    joinDim_int<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, newFactFilter,gpu_result,jNode->rightTable->tupleNum);
                    //joinDim_int<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
                else
                    joinDim_other<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);

            }else if (format == DICT){
                struct dictHeader * dheader = NULL;
                int byteNum;
                struct dictHeader * gpuDictHeader = NULL;

                dheader = (struct dictHeader *)table;
                byteNum = dheader->bitNum/8;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuDictHeader,sizeof(struct dictHeader)));
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));
                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                if(attrSize == sizeof(int))
                    joinDim_dict_int<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
                else
                    joinDim_dict_other<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader, byteNum, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

            }else if (format == RLE){

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                joinDim_rle<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
            }
        }
        
        res->attrTotalSize[i] = resSize;
        res->dataFormat[i] = UNCOMPRESSED;
        if(res->dataPos[i] == MEM){
            res->content[i] = (char *) malloc(resSize);
            memset(res->content[i],0,resSize);
            // Copy result back to host
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i],gpu_result,resSize,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_result));

        }else if(res->dataPos[i] == GPU){
            res->content[i] = gpu_result;
            char * tmp = (char *)malloc(resSize);
            memset(tmp, 0, resSize);
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(tmp,gpu_result,resSize,cudaMemcpyDeviceToHost));
        }
        if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));

    }

    CUDA_SAFE_CALL(cudaFree(gpuFactFilter));

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_count));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_hashNum));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_resPsum));

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    printf("HashJoin Time: %lf\n", timeE/(1000*1000));

    return res;

}
