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
//#include "../include/cuPrintf.cu"
//#include "../include/cuPrintf.cuh"

using namespace nvcuda;

// For wmma API, these must be multiples fo 16
#define MATRIX_M 16
#define MATRIX_N 16
#define MATRIX_K 16

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Define some error checking macros.
/*
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }}

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

/*
 * Count the number of dimension keys for each bucket.
 */

__global__ static void count_hash_num(char *dim, long  inNum,int *num,int hsize){
    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *)dim)[i]; // dim table join key content
        int hKey = joinKey & (hsize-1); // seems like hash into different buckets
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
        int pos = atomicAdd(&psum[hKey],1) * 2; // 2 -- 2*primaryKeySize for gpu_bucket
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

    //printf("left tuple#: %d\n", (int)inNum);
    //printf("right tuple#: %d\n", right_tupleNum);

    for (int i = offset; i < inNum; i += stride) {
        int fkey = ((int *)(fact))[i];
        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int fvalue = 0;

        for (int j = 0; j < keyNum; j++) { 
            int pSum = psum[hkey];
            int dimKey = ((int *)(bucket))[2*j + 2*pSum];

            // NOTE: i -- left (fact) table row number; dimId -- right (dim) table row number
            if (dimKey == fkey) { // check whether the tuple match WHERE condition
                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                dimId = dimId-1; // dirty fix for now, build_hash_table increase dimId by 1

                lcount ++;
                fvalue = dimId;

                newFactFilter[i*right_tupleNum+fvalue] = 1; // fact_tupleIndex*dim_tuple_num+dim_tupleIndex
                // make sure we can store duplicate tuples without replacing the previous matched tuple
            }
        }

        //factFilter[i] = fvalue; // orig code
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
        int fkey = ((int *)(fact))[i]; // foreign key of fact table 
        int hkey = fkey &(hsize-1); 
        int keyNum = num[hkey]; // gpu_hashNum -- sizeof(int)*hsize
        int fvalue = 0;

        for(int j=0;j<keyNum;j++){
            int pSum = psum[hkey]; // starting position of each bucket
            int dimKey = ((int *)(bucket))[2*j + 2*pSum]; // retrieve dim primary key from bucket
            
            if( dimKey == fkey){
                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];

                lcount ++;
                fvalue = dimId;
                break; // cannot support multi-matched tuples
            }
        }
        factFilter[i] = fvalue; // factFilter -- sizeof(int)*leftTupleNum
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

__global__ void static joinFact_other2(int *resPsum, char * fact,  int attrSize, long  num, int * filter, char * result, int right_tupleNum){
    //printf("joinFact_other2\n");

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //long localOffset = resPsum[startIndex] * attrSize;
    long localCount = resPsum[startIndex];

    //cuPrintf("fact localCount: %d\n", localCount);
    for(long i=startIndex;i<num;i+=stride){
        for (int j = 0; j < right_tupleNum; j++) {
            if(filter[i*right_tupleNum+j] != 0){
                
                //cuPrintf("fact_row#: %d\tdim_row#: %d\tfact_val: %.10f\n", i, j, ((float*)fact)[i]);
                ((float*)result)[localCount] = ((float *)fact)[i];
                //memcpy(result + localOffset, fact + i*attrSize, attrSize);
                //localOffset += attrSize;
                localCount++;
            }
        }
    }
}

// fact table -> left table
__global__ void static joinFact_int2(int *resPsum, char * fact,  int attrSize, long  num, int * filter, char * result, int right_tupleNum){
    //printf("joinFact_int2\n");

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for (int i = startIndex; i < (int) num; i += stride) {
        for (int j = 0; j < right_tupleNum; j++) {
            // judge if there is a matched tuple from newFactFilter, val==1 means match, localCount+=1
            if (filter[i*right_tupleNum+j] != 0) {
                ((int*)result)[localCount] = ((int *)fact)[i];
                //cuPrintf("fact_row#: %d\tdim_row#: %d\tfact_val: %d\n", i, j, ((int *)fact)[i]);
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

__global__ void static joinFact_int(int *resPsum, char * fact,  int attrSize, long  num, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];
    
    // orig code: assume each row only match once
    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            ((int*)result)[localCount] = ((int *)fact)[i];
            localCount ++;
        }
    }
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
__global__ void static joinDim_int2(int *resPsum, char * dim, int attrSize, long num,int * filter, char * result, int right_tupleNum){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];
    //cuPrintf("dim localCount: %lu\n", localCount); // from 0-16, and stop at 16

    for(int i = startIndex; i < (int) num; i += stride) {
        for (int j = 0; j < right_tupleNum; j++) {
            if (filter[i*right_tupleNum+j] != 0) {
                ((int*)result)[localCount] = ((int*)dim)[j];
                //cuPrintf("fact_row#: %d\tdim_row#: %d\tdim_val: %d\n", i, j, ((int *)dim)[j]);
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

__global__ void static joinDim_int(int *resPsum, char * dim, int attrSize, long num,int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    
    // orig code   
    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        //printf("dimId: %d\n", dimId);
        if( dimId != 0){
            ((int*)result)[localCount] = ((int*)dim)[dimId-1];
            //printf(" <= joinDim_int val: \t\t%d\n", ((int *)dim)[dimId-1]);
            localCount ++;
        }
    }
}

__global__ void static joinDim_other2(int *resPsum, char * dim, int attrSize, long num,int * filter, char * result, int right_tupleNum){
    //printf("joinDim_other2\n");

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        for (int j = 0; j < right_tupleNum; j++) {
            if (filter[i*right_tupleNum+j] != 0) {
                memcpy(result + localOffset, dim + j* attrSize, attrSize);
                localOffset += attrSize;
            }
        }
    }
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
    //cudaPrintfInit();

    /*
    int x, y;
    for (x = 0; x < 3; x++) {
        int idx = jNode->rightTable->attrIndex[x];
        //int idx = jNode->leftTable->attrIndex[x];
        //printf("idx: %d\tx: %d\n", idx, x);
        for (y = 0; y < jNode->rightTable->tupleNum * 4; y +=4) {
        //for (y = 0; y < jNode->leftTable->tupleNum * 4; y +=4) {
            int *temp;
            temp = (int*)(&jNode->rightTable->content[x][y]);
            //temp = (int*)(&jNode->leftTable->content[x][y]);

            if (idx == 1) // mat A 0: i; 1: j | mat B 0:i; 1:j
                printf("idx: %d\tx: %d\n", idx, x);
                printf("%d\n", *temp);

        }
    }
    //printf("leftKeyIndex: %d\n", jNode->rightTable->attrIndex[2]);
    */

    struct timespec start,end; // hashJoin overall timing
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

    //printf("leftOutputAttrNum: %d\n", jNode->leftOutputAttrNum);
    for(int i=0;i<jNode->leftOutputAttrNum;i++){
        int pos = jNode->leftPos[i];
        res->attrType[pos] = jNode->leftOutputAttrType[i];
        int index = jNode->leftOutputIndex[i];
        //printf("left index: %d\n", index);
        res->attrSize[pos] = jNode->leftTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    //printf("rightOutputAttrNum: %d\n", jNode->rightOutputAttrNum);
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

    // YDB hashJoin below this point

    int *gpu_psum1 = NULL;

    int hsize = jNode->rightTable->tupleNum;
    //printf("hsize: %d\n", hsize); // hsize seems to be used for count_group_num
    NP2(hsize); // return hsize the nearest power of 2

    //printf("after NP2 function hsize: %d\n", hsize); // 8
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_hashNum,sizeof(int)*hsize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_hashNum,0,sizeof(int)*hsize));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_psum,hsize*sizeof(int)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_bucket, 2*primaryKeySize));
    //printf("primaryKeySize: %d\n", primaryKeySize);
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

    threadNum = grid.x * block.x;

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_count,sizeof(int)*threadNum));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_resPsum,sizeof(int)*threadNum));

    int *gpuFactFilter = NULL;
    int *newFactFilter = NULL; // a[x][y] -> a[x*dim_size+y], x: left table's row id, dim_size: right table's tupleNum, y: right table's row id

    dataPos = jNode->leftTable->dataPos[jNode->leftKeyIndex];
    int format = jNode->leftTable->dataFormat[jNode->leftKeyIndex];

    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
    long filterSize = jNode->leftTable->attrSize[jNode->leftKeyIndex] * jNode->leftTable->tupleNum;
    //printf("left attrSize: %d\n", jNode->leftTable->attrSize[jNode->leftKeyIndex]);
    //printf("left keyIndex: %d\n", jNode->leftKeyIndex);
    long newSize = jNode->rightTable->tupleNum * jNode->leftTable->tupleNum * sizeof(int); // worst case is all matched
    //printf("newSize: %d\n", (int)newSize);
    int right_tupleNum = jNode->rightTable->tupleNum;
    //printf("join on index: %d\n", jNode->leftKeyIndex);

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact, foreignKeySize));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));

    }else if (dataPos == GPU || dataPos == UVA){
        gpu_fact = jNode->leftTable->content[jNode->leftKeyIndex];
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFactFilter,filterSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuFactFilter,0,filterSize));
    //printf("gpuFactFilter: %d\n", filterSize);

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&newFactFilter,newSize));
    //printf("newFactFilter: %d\n", newSize);
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

    // gpu_count result come from count_join_result
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp1,&gpu_count[threadNum-1],sizeof(int),cudaMemcpyDeviceToHost));
    scanImpl(gpu_count,threadNum,gpu_resPsum, pp);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp2,&gpu_resPsum[threadNum-1],sizeof(int),cudaMemcpyDeviceToHost));

    res->tupleNum = tmp1 + tmp2;
    //printf("tmp1: %d\n", tmp1); // 0
    //printf("tmp2: %d\n", tmp2);
    printf("[INFO]Number of join results: %d\n",res->tupleNum);

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_bucket));

    //printf("res->totalAttr: %d\n", res->totalAttr);
    for(int i=0; i<res->totalAttr; i++){

        int index, pos;
        long colSize = 0, resSize = 0;
        int leftRight = 0; // seems like left (fact) table = 0; right (dim) table = 1

        int attrSize, attrType;
        char * table = NULL;
        int found = 0 , dataPos, format;

        //printf("keepInGpu: %d\n", jNode->keepInGpu[1]); //GPU
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

        // judge whether fact or dim table and set the corresponding attrSize/attrType
        if(leftRight == 0){ // left (fact) table
            index = jNode->leftOutputIndex[pos]; 
            dataPos = jNode->leftTable->dataPos[index];
            format = jNode->leftTable->dataFormat[index];

            table = jNode->leftTable->content[index]; // used for later joinFact
            //printf("val index: %d\n", index);
            attrSize  = jNode->leftTable->attrSize[index];
            attrType  = jNode->leftTable->attrType[index];
            //printf("fact attrType: %d\n", attrType);
            colSize = jNode->leftTable->attrTotalSize[index];
            //printf("val colSize: %d\n", colSize);

            resSize = res->tupleNum * attrSize;
        }else{ // right (dim) table
            index = jNode->rightOutputIndex[pos]; 
            dataPos = jNode->rightTable->dataPos[index];
            format = jNode->rightTable->dataFormat[index];

            table = jNode->rightTable->content[index]; // used for later joinDim
            attrSize = jNode->rightTable->attrSize[index];
            attrType = jNode->rightTable->attrType[index];
            //printf("dim attrType: %d\n", attrType);
            colSize = jNode->rightTable->attrTotalSize[index];

            resSize = attrSize * res->tupleNum;
            leftRight = 1;
        }

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_result,resSize));

        if(leftRight == 0){ // means left talbe, call joinFact
            if(format == UNCOMPRESSED){
                //printf("dataPos == MEM: %d\n", dataPos == MEM);
                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                //if(attrSize == sizeof(int))
                if(attrType == INT)
                    joinFact_int2<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,newFactFilter,gpu_result, right_tupleNum);
                    //joinFact_int<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);
                else
                    joinFact_other2<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,newFactFilter,gpu_result, right_tupleNum);
                    //joinFact_other<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

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

                joinFact_int<<<grid,block>>>(gpu_resPsum,gpuRle, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));

            }

        }else{ // leftRight = 1, means right table, call joinDim
            if(format == UNCOMPRESSED){

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                if(attrType == sizeof(int))
                    joinDim_int2<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, newFactFilter,gpu_result,jNode->rightTable->tupleNum);
                    //joinDim_int<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
                else
                    joinDim_other2<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, newFactFilter,gpu_result, right_tupleNum);
                    //joinDim_other<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);

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
            //printf("res->dataPos in MEM\n");
            res->content[i] = (char *) malloc(resSize);
            memset(res->content[i],0,resSize);
            // Copy result back to host
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i],gpu_result,resSize,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_result));

        }else if(res->dataPos[i] == GPU){
            res->content[i] = gpu_result;
            //printf("res->dataPos[%d] in GPU, with resSize: %d\n", i, resSize);
            char * tmp = (char *)malloc(resSize);
            //float * tmp = (float *)malloc(resSize);
            //int * tmp = (int *)malloc(resSize);
            memset(tmp, 0, resSize);
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(tmp,gpu_result,resSize,cudaMemcpyDeviceToHost));
            /*
            for (int k = 0; k < resSize/sizeof(float); k++) {
                    printf("tmp content: %d\n", tmp[k]);
            }
            */
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
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();

    return res;

}
