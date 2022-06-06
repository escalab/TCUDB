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
#include <string.h>
#include <time.h>
#include <cuda.h>
#include "../include/common.h"
#include "../include/schema.h"
#include "../include/gpuCudaLib.h"

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)


__global__ static void materialize(char ** content,  int colNum, int *attrSize, long tupleNum, int tupleSize, char *result){
    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    
        int stride = blockDim.x * gridDim.x;

    for(long i=startIndex;i<tupleNum;i+=stride){
        int offset = 0;
        for(int j=0;j<colNum;j++){
            int aSize = attrSize[j];
            memcpy(result+i*tupleSize + offset, content[j]+ i*aSize, aSize);
            offset += aSize;
        }
    }
}

char * materializeCol(struct materializeNode * mn, struct statistic * pp){

    struct timespec start,end;
    clock_gettime(CLOCK_REALTIME,&start);

    struct tableNode *tn = mn->table;
    char * res, * gpuResult;
    char **gpuContent, **column;
    long size = tn->tupleNum * tn->tupleSize;
    int * gpuAttrSize;

    column = (char **) malloc(sizeof(char *) * tn->totalAttr);
    CHECK_POINTER(column);
    
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuContent, sizeof(char *) * tn->totalAttr));

    res = (char *) malloc(size);
    CHECK_POINTER(res);
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuResult, size));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuAttrSize,sizeof(int) * tn->totalAttr));

    for(int i=0;i<tn->totalAttr;i++){
        if(tn->dataPos[i] == MEM){
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&column[i], tn->tupleNum*tn->attrSize[i]));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[i], tn->content[i], tn->tupleNum *tn->attrSize[i], cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &column[i], sizeof(char *), cudaMemcpyHostToDevice));
        }else if(tn->dataPos[i] == GPU){
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &tn->content[i], sizeof(char *), cudaMemcpyHostToDevice));
        }
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuAttrSize, tn->attrSize, sizeof(int) * tn->totalAttr, cudaMemcpyHostToDevice));

    dim3 grid(512);
    dim3 block(128);

    materialize<<<grid,block>>> (gpuContent, tn->totalAttr, gpuAttrSize, tn->tupleNum, tn->tupleSize, gpuResult);

    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res, gpuResult, size, cudaMemcpyDeviceToHost));

    for(int i=0;i<tn->totalAttr;i++){
        if(tn->dataPos[i] == MEM){
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[i]));
        }
    }

    free(column);

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuContent));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuAttrSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResult));

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    printf("Materialization Time: %lf\n", timeE/(1000*1000));

    return res;
}
