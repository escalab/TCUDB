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
#include <cuda.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include "../include/common.h"
#include "../include/gpuCudaLib.h"
#include "scanImpl.cu"

#define SHARED_SIZE_LIMIT 1024 

__device__ static int gpu_strcmp(const char *s1, const char *s2, int len){
        int res = 0;

        for(int i=0;i < len;i++){
                if(s1[i]<s2[i]){
                        res = -1;
                        break;
                }else if(s1[i]>s2[i]){
                        res = 1;
                        break;
                }
        }
        return res;
}



/* use one GPU thread to count the number of unique key */

__global__ static void count_unique_keys_int(int *key, int tupleNum, int * result){
    int i = 0;
    int res = 1;
    for(i=0;i<tupleNum -1;i++){
        if(key[i+1] != key[i])
            res ++;
    }
    *result = res;
}

__global__ static void count_unique_keys_float(float *key, int tupleNum, int * result){
    int i = 0;
    int res = 1;
    for(i=0;i<tupleNum -1;i++){
        if(key[i+1] != key[i])
            res ++;
    }
    *result = res;
}

__global__ static void count_unique_keys_string(char *key, int tupleNum, int keySize,int * result){
    int i = 0;
    int res = 1;
    for(i=0;i<tupleNum -1;i++){
        if(gpu_strcmp(key+i*keySize, key+(i+1)*keySize,keySize) != 0)
            res ++;
    }
    *result = res;
}

/*
 * Count the number of each key using one single GPU thread. 
 */

__global__ static void count_key_num_int(int *key, int tupleNum, int * count){
    int pos = 0, i = 0;
    int lcount = 1;
    for(i = 0;i <tupleNum -1; i ++){
        if(i == tupleNum -2){
            if(key[i] != key[i+1]){
                count[pos] = lcount;
                count[pos+1] = 1;
            }else{
                count[pos] = lcount +1;
            }
        }else{
            if(key[i] != key[i+1]){
                count[pos] = lcount;
                lcount = 1;
                pos ++;
            }else{
                lcount ++;
            }
        }
    }
}

__global__ static void count_key_num_float(float *key, int tupleNum, int * count){
    int pos = 0, i = 0;
    int lcount = 1;
    for(i = 0;i <tupleNum -1; i ++){
        if(i == tupleNum -2){
            if(key[i] != key[i+1]){
                count[pos] = lcount;
                count[pos+1] = 1;
            }else{
                count[pos] = lcount +1;
            }
        }else{
            if(key[i] != key[i+1]){
                count[pos] = lcount;
                lcount = 1;
                pos ++;
            }else{
                lcount ++;
            }
        }
    }
}


__global__ static void count_key_num_string(char *key, int tupleNum, int keySize, int * count){
    int pos = 0, i = 0;
    int lcount = 1;
    for(i = 0;i <tupleNum -1; i ++){
        if(i == tupleNum -2){
            if(gpu_strcmp(key+i*keySize, key+(i+1)*keySize,keySize)!=0){
                count[pos] = lcount;
                count[pos+1] = 1;
            }else{
                count[pos] = lcount +1;
            }
        }else{
            if(gpu_strcmp(key+i*keySize, key+(i+1)*keySize,keySize)!=0){
                count[pos] = lcount;
                lcount = 1;
                pos ++;
            }else{
                lcount ++;
            }
        }
    }
}

__device__ static inline void ComparatorInt(
    int &keyA,int &valA,int &keyB,int &valB,int dir)
{
    int t;

    if ((keyA > keyB) == dir)
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
        t = valA;
        valA = valB;
        valB = t;
    }
}

__device__ static inline void ComparatorFloat(
    float &keyA,int &valA,float &keyB,int &valB,int dir)
{
    float t1;
    int t2;

    if ((keyA > keyB) == dir)
    {
        t1 = keyA;
        keyA = keyB;
        keyB = t1;
        t2 = valA;
        valA = valB;
        valB = t2;
    }
}


__device__ static inline void Comparator(
    char * keyA,
    int &valA,
    char * keyB,
    int &valB,
    int keySize,
    int dir
)
{
        int t;
        char buf[32];

    if ((gpu_strcmp(keyA,keyB,keySize) == 1) == dir)
    {
        memcpy(buf, keyA, keySize);
        memcpy(keyA, keyB, keySize);
        memcpy(keyB, buf, keySize);
        t = valA;
        valA = valB;
        valB = t;
    }
}



#define NTHREAD  (SHARED_SIZE_LIMIT/2)

__global__ static void sort_key_string(char * key, int tupleNum, int keySize, char *result, int *pos,int dir){
    int lid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ char bufKey[SHARED_SIZE_LIMIT * 32];
    __shared__ int bufVal[SHARED_SIZE_LIMIT];

    int gid = bid * SHARED_SIZE_LIMIT + lid;

    memcpy(bufKey + lid*keySize, key + gid*keySize, keySize);
    bufVal[lid] = gid;
    memcpy(bufKey + (lid+blockDim.x)*keySize, key +(gid+blockDim.x)*keySize, keySize);
    bufVal[lid+blockDim.x] = gid+ blockDim.x;

    __syncthreads();

    for (int size = 2; size < tupleNum && size < SHARED_SIZE_LIMIT; size <<= 1){
            int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

            for (int stride = size / 2; stride > 0; stride >>= 1){
                    __syncthreads();
                    int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                    Comparator(
                            bufKey+pos*keySize, bufVal[pos +      0],
                            bufKey+(pos+stride)*keySize, bufVal[pos + stride],
                            keySize,
                            ddd
                    );
            }
    }

    {
        for (int stride = blockDim.x ; stride > 0; stride >>= 1)
        {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                bufKey+pos*keySize, bufVal[pos +      0],
                bufKey+(pos+stride)*keySize, bufVal[pos + stride],
                keySize,
                dir
            );
        }
    }

    __syncthreads();

    memcpy(result + gid*keySize, bufKey + lid*keySize, keySize);

    ((int *)pos)[gid] = bufVal[lid];
    memcpy(result + (gid+blockDim.x)*keySize, bufKey + (lid+blockDim.x)*keySize,keySize);
    ((int *)pos)[gid+blockDim.x] = bufVal[lid+blockDim.x];

}

/*
 * Sorting small number of intergers.
 */

__global__ static void sort_key_int(int * key, int tupleNum, int *result, int *pos,int dir){
    int lid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ int bufKey[SHARED_SIZE_LIMIT];
    __shared__ int bufVal[SHARED_SIZE_LIMIT];

    int gid = bid * SHARED_SIZE_LIMIT + lid;

    bufKey[lid] = key[gid];
    bufVal[lid] = gid;
    bufKey[lid + blockDim.x] = key[gid + blockDim.x];
    bufVal[lid+blockDim.x] = gid+ blockDim.x;

    __syncthreads();

    for (int size = 2; size < tupleNum && size < SHARED_SIZE_LIMIT; size <<= 1){
        int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (int stride = size / 2; stride > 0; stride >>= 1){
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            ComparatorInt(
                bufKey[pos + 0], bufVal[pos +      0],
                bufKey[pos + stride], bufVal[pos + stride],
                ddd
            );
        }
    }

    {
        for (int stride = blockDim.x ; stride > 0; stride >>= 1)
        {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            ComparatorInt(
                bufKey[pos + 0], bufVal[pos +      0],
                bufKey[pos + stride], bufVal[pos + stride],
                dir
            );
        }
    }

    __syncthreads();

    result[gid] = bufKey[lid];
    pos[gid] = bufVal[lid];
    result[gid + blockDim.x] = bufKey[lid + blockDim.x];
    pos[gid+blockDim.x] = bufVal[lid+blockDim.x];

}


/*
 * Sorting small number of floats.
 */

__global__ static void sort_key_float(float * key, int tupleNum,  float *result, int *pos,int dir){
    int lid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ float bufKey[SHARED_SIZE_LIMIT];
    __shared__ int bufVal[SHARED_SIZE_LIMIT];

    int gid = bid * SHARED_SIZE_LIMIT + lid;

    bufKey[lid] = key[gid];
    bufVal[lid] = gid;
    bufKey[lid + blockDim.x] = key[gid + blockDim.x];
    bufVal[lid+blockDim.x] = gid+ blockDim.x;

    __syncthreads();

    for (int size = 2; size < tupleNum && size < SHARED_SIZE_LIMIT; size <<= 1){
        int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (int stride = size / 2; stride > 0; stride >>= 1){
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            ComparatorFloat(
                bufKey[pos + 0], bufVal[pos +      0],
                bufKey[pos + stride], bufVal[pos + stride],
                ddd
            );
        }
    }

    {
        for (int stride = blockDim.x ; stride > 0; stride >>= 1)
        {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            ComparatorFloat(
                bufKey[pos + 0], bufVal[pos +      0],
                bufKey[pos + stride], bufVal[pos + stride],
                dir
            );
        }
    }

    __syncthreads();

    result[gid] = bufKey[lid];
    pos[gid] = bufVal[lid];
    result[gid + blockDim.x] = bufKey[lid + blockDim.x];
    pos[gid+blockDim.x] = bufVal[lid+blockDim.x];

}

/*
 * Naive sort. One thread per block.
 */

__global__ static void sec_sort_key_int(int *key, int *psum, int *count ,int tupleNum, int *inputPos, int* outputPos){
    int tid = blockIdx.x; 
    int start = psum[tid];
    int end = start + count[tid]; 

    for(int i=start; i< end-1; i++){
        int min = key[i];
        int tmp = min;
        int pos = i;
        for(int j=i+1;j<end;j++){
            if(min > key[j]){
                min = key[j];
                pos = j;
            }
        }
        key[pos] = tmp;
        outputPos[i] = inputPos[pos];
        inputPos[pos] = inputPos[i];
    }
    outputPos[end-1] = inputPos[end-1];
}

__global__ static void sec_sort_key_float(float *key, int *psum, int *count ,int tupleNum, int *inputPos, int* outputPos){
    int tid = blockIdx.x;
    int start = psum[tid];
    int end = start + count[tid];

    for(int i=start; i< end-1; i++){
        float min = key[i];
        float tmp = min;
        int pos = i;
        for(int j=i+1;j<end;j++){
            if(min > key[j]){
                min = key[j];
                pos = j;
            }
        }
        key[pos] = tmp;
        outputPos[i] = inputPos[pos];
        inputPos[pos] = inputPos[i];
    }
    outputPos[end-1] = inputPos[end-1];
}

__global__ static void sec_sort_key_string(char *key, int keySize, int *psum, int *count ,int tupleNum, int *inputPos, int* outputPos){
    int tid = blockIdx.x;
    int start = psum[tid];
    int end = start + count[tid];

    for(int i=start; i< end-1; i++){
        char min[128]; 
        char tmp[128];
        memcpy(min,key + i*keySize, keySize);
        memcpy(tmp,key + i*keySize, keySize);
        int pos = i;
        for(int j=i+1;j<end;j++){
            if(gpu_strcmp(min, key+j*keySize,keySize)>0){
                memcpy(min,key + j*keySize, keySize);
                pos = j;
            }
        }
        memcpy(key+pos*keySize,tmp, keySize);
        outputPos[i] = inputPos[pos];
        inputPos[pos] = inputPos[i];
    }
    outputPos[end-1] = inputPos[end-1];
}


__global__ static void set_key_string(char *key, int tupleNum){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=tid;i<tupleNum;i+=stride)
        key[i] = CHAR_MAX;

}

__global__ static void set_key_int(int *key, int tupleNum){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=tid;i<tupleNum;i+=stride)
        key[i] = INT_MAX;

}

__global__ static void set_key_float(float *key, int tupleNum){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=tid;i<tupleNum;i+=stride)
        key[i] = FLT_MAX;
}

/*
 * gather the elements from the @col into @result.
 */

__global__ static void gather_col_int(int * keyPos, int* col, int newNum, int tupleNum, int*result){
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=index;i<newNum;i+=stride){
        int pos = keyPos[i];
        if(pos<tupleNum)
            result[i] = col[pos];
    }
}

__global__ static void gather_col_float(int * keyPos, float* col, int newNum, int tupleNum, float*result){
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=index;i<newNum;i+=stride){
        int pos = keyPos[i];
        if(pos<tupleNum)
            result[i] = col[pos];
    }
}

__global__ static void gather_col_string(int * keyPos, char* col, int newNum, int tupleNum, int keySize,char*result){
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=index;i<newNum;i+=stride){
        int pos = keyPos[i];
        if(pos<tupleNum)
            memcpy(result + i*keySize, col + pos*keySize, keySize);
    }
}



/* generate the final result*/

__global__ static void gather_result(int * keyPos, char ** col, int newNum, int tupleNum, int *size, int colNum, char **result){
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int j=0;j<colNum;j++){
        for(int i=index;i<newNum;i+=stride){
            int pos = keyPos[i];
            if(pos<tupleNum)
                memcpy(result[j] + i*size[j], col[j] +pos*size[j], size[j]);
        }
    }
}


/*
 * orderBy: sort the input data by the order by columns
 *
 * Prerequisite:
 *  input data are not compressed
 *
 * Input:
 *  odNode: the groupby node which contains the input data and groupby information
 *  pp: records the statistics such as kernel execution time
 *
 * Return:
 *  a new table node
 */


struct tableNode * orderBy(struct orderByNode * odNode, struct statistic *pp){
    struct tableNode * res = NULL;
    struct timespec start, end;

    clock_gettime(CLOCK_REALTIME,&start);

    assert(odNode->table->tupleNum < SHARED_SIZE_LIMIT);

    res = (struct tableNode *)malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);
    res->tupleNum = odNode->table->tupleNum;
    res->totalAttr = odNode->table->totalAttr;
    res->tupleSize = odNode->table->tupleSize;

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
    res->content = (char **) malloc(sizeof(char *) * res->totalAttr);
    CHECK_POINTER(res->content);

    int gpuTupleNum = odNode->table->tupleNum;
    char *gpuKey = NULL, **column = NULL, **gpuContent = NULL;
    char *gpuSortedKey = NULL;
    int *gpuSize = NULL, *gpuPos = NULL;

    column = (char**) malloc(sizeof(char*) *res->totalAttr);
    CHECK_POINTER(column);
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuContent, sizeof(char *) * res->totalAttr));

    for(int i=0;i<res->totalAttr;i++){
        res->attrType[i] = odNode->table->attrType[i];
        res->attrSize[i] = odNode->table->attrSize[i];
        res->attrTotalSize[i] = odNode->table->attrTotalSize[i];
        res->dataPos[i] = MEM;
        res->dataFormat[i] = UNCOMPRESSED;
        res->content[i] = (char *) malloc( res->attrSize[i] * res->tupleNum);
        CHECK_POINTER(res->content[i]);

        int attrSize = res->attrSize[i];
        if(odNode->table->dataPos[i] == MEM){
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&column[i], attrSize *res->tupleNum));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[i], odNode->table->content[i], attrSize*res->tupleNum, cudaMemcpyHostToDevice));
        }else if (odNode->table->dataPos[i] == GPU){
            column[i] = odNode->table->content[i];
        }

        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &column[i], sizeof(char *), cudaMemcpyHostToDevice));
    }

    int newNum = 1;
    while(newNum<gpuTupleNum){
        newNum *=2;
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuPos, sizeof(int)*newNum));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuSize, sizeof(int) * res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuSize, res->attrSize, sizeof(int) * res->totalAttr, cudaMemcpyHostToDevice););

    char ** gpuResult = NULL;
    char ** result = NULL;
    result = (char**)malloc(sizeof(char *) * res->totalAttr);
    CHECK_POINTER(result);

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuResult, sizeof(char*)*res->totalAttr));
    for(int i=0;i<res->totalAttr;i++){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&result[i], res->attrSize[i]* gpuTupleNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuResult[i], &result[i], sizeof(char*), cudaMemcpyHostToDevice));
    }
    

    /* Sort by the first orderby column first */

    int dir;
    if(odNode->orderBySeq[0] == ASC)
        dir = 1;
    else
        dir = 0;

    int index = odNode->orderByIndex[0];
    int type = odNode->table->attrType[index];

    if(type == INT){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuKey, sizeof(int) * newNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuSortedKey, sizeof(int) * newNum));
        set_key_int<<<8,128>>>((int*)gpuKey,newNum);
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuKey, column[index], sizeof(int)*gpuTupleNum,cudaMemcpyDeviceToDevice));
        sort_key_int<<<1, newNum/2>>>((int*)gpuKey, newNum, (int*)gpuSortedKey, gpuPos, dir);

    }else if (type == FLOAT){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuKey, sizeof(float) * newNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuSortedKey, sizeof(float) * newNum));
        set_key_float<<<8,128>>>((float*)gpuKey,newNum);
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuKey, column[index], sizeof(int)*gpuTupleNum,cudaMemcpyDeviceToDevice));
        sort_key_float<<<1, newNum/2>>>((float*)gpuKey, newNum, (float*)gpuSortedKey, gpuPos, dir);

    }else if (type == STRING){
        int keySize = odNode->table->attrSize[index];
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuKey, keySize * newNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuSortedKey, keySize * newNum));
        set_key_string<<<8,128>>>(gpuKey,newNum*keySize);
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuKey, column[index], keySize*gpuTupleNum,cudaMemcpyDeviceToDevice));
        sort_key_string<<<1, newNum/2>>>(gpuKey, newNum, keySize,gpuSortedKey, gpuPos, dir);
    }

    /* Currently we only support no more than 2 orderBy columns */

    if (odNode->orderByNum == 2){
        int keySize = odNode->table->attrSize[index];
        int secIndex = odNode->orderByIndex[1];
        int keySize2 = odNode->table->attrSize[secIndex];
        int secType = odNode->table->attrType[secIndex];
        int *keyNum = NULL, *keyCount = NULL, *keyPsum = NULL;

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&keyNum, sizeof(int)));
        if(type == INT){
            count_unique_keys_int<<<1,1>>>((int *)gpuSortedKey, gpuTupleNum,keyNum);
        }else if (type == FLOAT){
            count_unique_keys_float<<<1,1>>>((float *)gpuSortedKey, gpuTupleNum, keyNum);

        }else if (type == STRING){
            count_unique_keys_string<<<1,1>>>(gpuSortedKey, gpuTupleNum,keySize,keyNum);
        }

        int cpuKeyNum;
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&cpuKeyNum, keyNum, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&keyCount, sizeof(int)* cpuKeyNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&keyPsum, sizeof(int)* cpuKeyNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(keyPsum, 0, sizeof(int) * cpuKeyNum));

        if(type == INT){
            count_key_num_int<<<1,1>>>((int*)gpuSortedKey,gpuTupleNum,keyCount);
        }else if (type == FLOAT){
            count_key_num_float<<<1,1>>>((float*)gpuSortedKey,gpuTupleNum,keyCount);

        }else if (type == STRING){
            count_key_num_string<<<1,1>>>(gpuSortedKey,gpuTupleNum,keySize,keyCount);
        }
        scanImpl(keyCount, cpuKeyNum, keyPsum, pp);

        int * gpuPos2 = NULL;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuPos2, sizeof(int)* newNum));
        char * gpuKey2 = NULL;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuKey2, keySize2 * newNum));

        if(secType == INT){
            gather_col_int<<<8,128>>>(gpuPos,(int*)column[secIndex],newNum, gpuTupleNum, (int*)gpuKey2);
            sec_sort_key_int<<<cpuKeyNum,1>>>((int*)gpuKey2, keyPsum, keyCount , gpuTupleNum, gpuPos, gpuPos2);
        }else if (secType == FLOAT){
            gather_col_float<<<8,128>>>(gpuPos,(float*)column[secIndex],newNum, gpuTupleNum, (float*)gpuKey2);
            sec_sort_key_float<<<cpuKeyNum,1>>>((float*)gpuKey2, keyPsum, keyCount , gpuTupleNum, gpuPos, gpuPos2);
        }else if (secType == STRING){
            gather_col_string<<<8,128>>>(gpuPos,column[secIndex],newNum, gpuTupleNum, keySize2,gpuKey2);
            sec_sort_key_string<<<cpuKeyNum,1>>>(gpuKey2, keySize2, keyPsum, keyCount , gpuTupleNum, gpuPos, gpuPos2);
        }

        gather_result<<<8,128>>>(gpuPos2, gpuContent, newNum, gpuTupleNum, gpuSize,res->totalAttr,gpuResult);
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(keyCount));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(keyNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuPos2));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuKey2));

    }else{
        gather_result<<<8,128>>>(gpuPos, gpuContent, newNum, gpuTupleNum, gpuSize,res->totalAttr,gpuResult);
    }

    for(int i=0; i<res->totalAttr;i++){
        int size = res->attrSize[i] * gpuTupleNum;
        memset(res->content[i],0, size);
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i], result[i],size, cudaMemcpyDeviceToHost));
    }

    for(int i=0;i<res->totalAttr;i++){
        if(odNode->table->dataPos[i] == MEM)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[i]));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(result[i]));
    }

    free(column);
    free(result);

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuKey));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuSortedKey));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuContent));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResult));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuPos));

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    printf("OrderBy Time: %lf\n", timeE/(1000*1000));

    return res;
}
