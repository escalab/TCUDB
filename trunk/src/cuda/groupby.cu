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
#include <string.h>
#include <time.h>
#include "../include/common.h"
#include "../include/gpuCudaLib.h"
#include "../include/cudaHash.h"
#include "scanImpl.cu"
#include "../include/cuPrintf.cu"
#include "../include/cuPrintf.cuh"

/*
#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
*/

/*
 * Combine the group by columns to build the group by keys. 
 */

__global__ static void build_groupby_key(char ** content, int gbColNum, int * gbIndex, int * gbType, int * gbSize, long tupleNum, int * key, int *num, int* groupNum){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(long i = offset; i< tupleNum; i+= stride){
        char buf[128] = {0};
        for (int j=0; j< gbColNum; j++){
            char tbuf[32]={0};
            int index = gbIndex[j];

            if (index == -1){
                gpuItoa(1,tbuf,10);
                gpuStrncat(buf,tbuf,1);

            }else if (gbType[j] == STRING){
                gpuStrncat(buf, content[index] + i*gbSize[j], gbSize[j]);

            }else if (gbType[j] == INT){
                int key = ((int *)(content[index]))[i];
                gpuItoa(key,tbuf,10);
                gpuStrcat(buf,tbuf);
            }
        }
        int hkey = StringHash(buf) % HSIZE;
        key[i]= hkey;
        num[hkey] = 1; // gb count +1
        atomicAdd(&(groupNum[hkey]), 1);
    }
}


/*
 * Count the number of groups 
 */

__global__ static void count_group_num(int *num, int tupleNum, int *totalCount){
        int stride = blockDim.x * gridDim.x;
        int offset = blockIdx.x * blockDim.x + threadIdx.x;
        int localCount = 0;

        for(int i=offset; i<tupleNum; i+= stride){
                if(num[i] == 1){
                        localCount ++;
                }
        }

        atomicAdd(totalCount,localCount);
}

/*
 * Calculate the groupBy expression.
 */

__device__ static float calMathExp(char **content, struct mathExp exp, int pos){
    float res;

    // terminate condition
    if(exp.op == NOOP){
        // opType -- regular column or a constant
        if (exp.opType == CONS)
            res = exp.opValue;
        else{
            int index = exp.opValue;
            int type  = exp.dataType;
            //cuPrintf("pos for content: %d\n", pos); // 0-15
            if (type == INT) {
                res = ((int *)(content[index]))[pos];
            } else { // type == FLOAT
                res = ((float *)(content[index]))[pos];
            }
            //res = ((int *)(content[index]))[pos];
            //res = ((float *)(content[index]))[pos];
        }
    
    }else if(exp.op == PLUS ){
        res = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos) + calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);

    }else if (exp.op == MINUS){
        res = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos) - calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);

    }else if (exp.op == MULTIPLY){
        // NOTE: here only perform multiply, so duplicates may happen
        res = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos) * calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);

    }else if (exp.op == DIVIDE){
        float left = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos);
        float right = calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);
        //cuPrintf("left: %f\tright: %.0f\n", left, right);
        res = left / right;
        //res = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos) / calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);
    }

    return res;
}

/*
 * group by constant. Currently only support SUM function.
 */

__global__ static void agg_cal_cons(char ** content, int colNum, struct groupByExp* exp, long tupleNum, char ** result, float *gpuPageRank){

    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float buf[32];

    for(int i=0;i<32;i++)
        buf[i] = 0;

    // peform computation on all matched tuples
    //cuPrintf("tupleNum: %d\n", tupleNum); // 16
    for(int i=index;i<tupleNum;i+=stride){
        for(int j=0;j<colNum;j++){ //j is 0
            int func = exp[j].func;
            if (func == SUM){
                //cuPrintf("opNum: %d\n", exp[j].exp.opNum);//2
                float tmpRes = calMathExp(content, exp[j].exp, i);
                //FIXME: for fair comparison, store into output array
                gpuPageRank[i] = tmpRes;
                //cuPrintf("%.8f\n", tmpRes);
                buf[j] += tmpRes;
            }else if (func == AVG){

                float tmpRes = calMathExp(content, exp[j].exp, i)/tupleNum;
                buf[j] += tmpRes;
            }
        }
    }

    // final result
    for(int i=0;i<colNum;i++)
        atomicAdd(&((float *)result[i])[0], buf[i]);
}

/*
 * gropu by
 */

__global__ static void agg_cal(char ** content, int colNum, struct groupByExp* exp, int * gbType, int * gbSize, long tupleNum, int * key, int *psum, int * groupNum, char ** result){

    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=index;i<tupleNum;i+=stride){

        int hKey = key[i];
        int offset = psum[hKey];

        for(int j=0;j<colNum;j++){
            int func = exp[j].func;
            
            if(func ==NOOP){
                int type = exp[j].exp.opType;

                if(type == CONS){
                    int value = exp[j].exp.opValue;
                    ((int *)result[j])[offset] = value;
                }else{
                    int index = exp[j].exp.opValue;
                    int attrSize = gbSize[j];
                    if(attrSize == sizeof(int)) {
                        ((int *)result[j])[offset] = ((int*)content[index])[i];
                        //No-op
                    }     
                    else
                        memcpy(result[j] + offset*attrSize, content[index] + i * attrSize, attrSize);
                }

            }else if (func == SUM ){
                float tmpRes = calMathExp(content, exp[j].exp, i);

                atomicAdd(& ((float *)result[j])[offset], tmpRes);
                //printf("result: %.0f\tj: %d\toffset: %d\n", ((float *)result[j])[offset], j, offset);
            } else if (func == AVG){
                float tmpRes = calMathExp(content, exp[j].exp, i)/groupNum[hKey];
                atomicAdd(& ((float *)result[j])[offset], tmpRes);
            }
        }
    }
}


/* 
 * groupBy: group by the data and calculate. 
 * 
 * Prerequisite:
 *  input data are not compressed
 *
 * Input:
 *  gb: the groupby node which contains the input data and groupby information
 *  pp: records the statistics such as kernel execution time 
 *
 * Return:
 *  a new table node
 */

struct tableNode * groupBy(struct groupByNode * gb, struct statistic * pp){

    cudaPrintfInit();

    struct timespec start,end;
    struct timespec cudaMemcpy_start,cudaMemcpy_end;
    clock_gettime(CLOCK_REALTIME,&start);
    int *gpuGbIndex = NULL, gpuTupleNum, gpuGbColNum;
    int *gpuGbType = NULL, *gpuGbSize = NULL;

    int *gpuGbKey = NULL;
    char ** gpuContent = NULL, **column = NULL;
    float *gpuPageRank;

    /*
     * @gbCount: the number of groups
     * gbConstant: whether group by constant
     */

    int gbCount;
    int gbConstant = 0;

    //printf("factType: %d\n", gb->table->factType);
    //printf("dimType: %d\n", gb->table->dimType);
    struct tableNode *res = (struct tableNode *) malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);
    res->tupleSize = gb->tupleSize;
    res->totalAttr = gb->outputAttrNum;
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
        res->attrType[i] = gb->attrType[i];
        res->attrSize[i] = gb->attrSize[i];
        res->dataFormat[i] = UNCOMPRESSED;
    }
    
    gpuTupleNum = gb->table->tupleNum;
    gpuGbColNum = gb->groupByColNum;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuPageRank, gpuTupleNum * sizeof(float *)));
    /*
    float * tmp = (float *)malloc(64);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(tmp,gb->table->content[0],64,cudaMemcpyDeviceToHost));
    for (int k = 0; k < 16; k++) {
        printf("tmp content: %f\n", tmp[k]);
    }
    free(tmp);
    */

    //printf("gpuGbColNum: %d\n", gb->groupByColNum);

    // groupByIndex == -1 means query doesn't contain group by keyword
    if(gpuGbColNum == 1 && gb->groupByIndex[0] == -1){
        gbConstant = 1;
    }


    dim3 grid(1024);
    //dim3 grid(512);
    dim3 block(128);
    int blockNum = gb->table->tupleNum / block.x + 1;
    if(blockNum < 1024)
        grid = blockNum;

    int *gpu_hashNum = NULL, *gpu_psum = NULL, *gpuGbCount = NULL, *gpu_groupNum = NULL;

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuContent, gb->table->totalAttr * sizeof(char *)));
    column = (char **) malloc(sizeof(char *) * gb->table->totalAttr);
    CHECK_POINTER(column);

    // copy table content for group by operation
    for(int i=0;i<gb->table->totalAttr;i++){
        int attrSize = gb->table->attrSize[i];
        if(gb->table->dataPos[i]==MEM){
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)& column[i], attrSize * gb->table->tupleNum));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[i], gb->table->content[i], attrSize *gb->table->tupleNum, cudaMemcpyHostToDevice));

            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &column[i], sizeof(char *), cudaMemcpyHostToDevice));
        }else{
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &gb->table->content[i], sizeof(char *), cudaMemcpyHostToDevice));
        }
    }

    if(gbConstant != 1){ // query has group by keywords, need build_groupby_key, count_group_num and scanImpl to update gbCount

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbType, sizeof(int) * gb->groupByColNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbType,gb->groupByType, sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbSize, sizeof(int) * gb->groupByColNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbSize,gb->groupBySize, sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));


        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbKey, gb->table->tupleNum * sizeof(int)));

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbIndex, sizeof(int) * gb->groupByColNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbIndex, gb->groupByIndex,sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_hashNum,sizeof(int)*HSIZE));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_hashNum,0,sizeof(int)*HSIZE));

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_groupNum,sizeof(int)*HSIZE));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_groupNum,0,sizeof(int)*HSIZE));

        build_groupby_key<<<grid,block>>>(gpuContent,gpuGbColNum, gpuGbIndex, gpuGbType,gpuGbSize,gpuTupleNum, gpuGbKey, gpu_hashNum, gpu_groupNum);
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbType));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbSize));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbIndex));

        gbCount = 1;

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbCount,sizeof(int)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuGbCount, 0, sizeof(int)));

        count_group_num<<<grid,block>>>(gpu_hashNum, HSIZE, gpuGbCount);
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        // copy groub by count back to host
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gbCount, gpuGbCount, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_psum,HSIZE*sizeof(int)));
        scanImpl(gpu_hashNum,HSIZE,gpu_psum,pp);

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbCount));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_hashNum));
    }

    if(gbConstant == 1)
        res->tupleNum = 1;
    else // query has group by keyword
        res->tupleNum = gbCount;

    printf("[INFO]Number of groupBy results: %ld\n",res->tupleNum);
    // after this point, computation occurs

    char ** gpuResult = NULL;
    char ** result = NULL;
    
    result = (char **)malloc(sizeof(char*)*res->totalAttr); // host stores data address on GPU
    CHECK_POINTER(result);
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuResult, sizeof(char *)* res->totalAttr));

    // copy thing to device memory for computation
    for(int i=0; i<res->totalAttr;i++){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&result[i], res->tupleNum * res->attrSize[i]));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(result[i], 0, res->tupleNum * res->attrSize[i]));
        res->content[i] = result[i]; 
        res->dataPos[i] = GPU;
        res->attrTotalSize[i] = res->tupleNum * res->attrSize[i];
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuResult[i], &result[i], sizeof(char *), cudaMemcpyHostToDevice));
    }


    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbType, sizeof(int)*res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbType, res->attrType, sizeof(int)*res->totalAttr, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbSize, sizeof(int)*res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbSize, res->attrSize, sizeof(int)*res->totalAttr, cudaMemcpyHostToDevice));

    struct groupByExp *gpuGbExp;

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuGbExp, sizeof(struct groupByExp)*res->totalAttr));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbExp, gb->gbExp, sizeof(struct groupByExp)*res->totalAttr, cudaMemcpyHostToDevice));
    for(int i=0;i<res->totalAttr;i++){
        struct mathExp * tmpMath;
        if(gb->gbExp[i].exp.opNum == 2){ // 2 operands
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&tmpMath, 2* sizeof(struct mathExp)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(tmpMath,(struct mathExp*)gb->gbExp[i].exp.exp,2*sizeof(struct mathExp), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&(gpuGbExp[i].exp.exp), &tmpMath, sizeof(struct mathExp *), cudaMemcpyHostToDevice));
        }
    }

    gpuGbColNum = res->totalAttr; // not sure why update second times
    //printf("2 gpuGbColNum: %d\n", res->totalAttr);

    //verify gpuContent
    /*
    float * tmp2 = (float *)malloc(64);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(tmp2,gb->table->content[0],64,cudaMemcpyDeviceToHost));
    for (int k = 0; k < 16; k++) {
        printf("tmp2 content: %f\n", tmp2[k]);
    }
    free(tmp2);
    */

    if(gbConstant !=1){ // query has group by keyword
        agg_cal<<<grid,block>>>(gpuContent, gpuGbColNum, gpuGbExp, gpuGbType, gpuGbSize, gpuTupleNum, gpuGbKey, gpu_psum, gpu_groupNum,gpuResult);

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbKey));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_groupNum));
    }else // query has no group by keyword
        agg_cal_cons<<<grid,block>>>(gpuContent, gpuGbColNum, gpuGbExp, gpuTupleNum,gpuResult, gpuPageRank);

    // verify result on host
    float *h_pageRank = (float *)malloc(gpuTupleNum*sizeof(float));
    clock_gettime(CLOCK_REALTIME,&cudaMemcpy_start);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(h_pageRank, gpuPageRank, gpuTupleNum*sizeof(float),cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME,&cudaMemcpy_end);
    /*
    for (int k = 0; k < gpuTupleNum; k++) {
        printf("PageRank[%d]: %.10f\n", k, h_pageRank[k]);
    }
    */

    /*
    float *h_res = (float *)malloc(sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(h_res, result[0], sizeof(float),cudaMemcpyDeviceToHost));
    printf("host res: %f\n", h_res[0]);
    */ 

    for(int i=0; i<gb->table->totalAttr;i++){
        if(gb->table->dataPos[i]==MEM)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[i]));
    }
    
    free(column);
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuContent));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbType));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbExp));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResult));

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    double cudaMemcpytime = (cudaMemcpy_end.tv_sec -  cudaMemcpy_start.tv_sec)* BILLION + cudaMemcpy_end.tv_nsec - cudaMemcpy_start.tv_nsec;
    printf("GroupBy Time: %lf\n", timeE/(1000*1000));
    printf("cudaMemcpy Time: %lf\n", cudaMemcpytime/(1000*1000));

    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();

    return res;
}
