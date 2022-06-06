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
#include "../include/common.h"
#include "../include/inviJoin.h"
#include "../include/gpuCudaLib.h"
#include "scanImpl.cu"

__global__ static void count_hash_num(char *dim, long dNum,int *num){
	int stride = blockDim.x * gridDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i=offset;i<dNum;i+=stride){
		int joinKey = ((int *)dim)[i]; 
		int hKey = joinKey % HSIZE;
		atomicAdd(&(num[hKey]),1);
	}
}

__global__ static void build_hash_table(char *dim, long dNum, int *psum, char * bucket){

	int stride = blockDim.x * gridDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i=offset;i<dNum;i+=stride){
		int joinKey = ((int *) dim)[i]; 
		int hKey = joinKey % HSIZE;
		int offset = atomicAdd(&psum[hKey],1) * 2;
		((int*)bucket)[offset] = joinKey;
		offset += 1;
		int dimId = i+1;
		((int*)bucket)[offset] = dimId;
	}
}


__global__ static void count_join_result(int* hashNum, int* psum, char* bucket, char* fact, long fNum, int * factFilter){
	int stride = blockDim.x * gridDim.x;
	long offset = blockIdx.x*blockDim.x + threadIdx.x;

	for(int i=offset;i<fNum;i+=stride){
		int fKey = ((int *)fact)[i];
		int hKey = fKey % HSIZE;
		int keyNum = hashNum[hKey];

		for(int j=0;j<keyNum;j++){
			int pSum = psum[hKey];
			int dimKey = ((int *)bucket)[2*j + 2*pSum];

			if(dimKey == fKey){
				int dimId = ((int *)bucket)[2*j + 2*pSum +1];
				factFilter[i] = dimId;
				break;
			}
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

__global__ void static joinFact_int(int *resPsum, char * fact,  int attrSize, long  num, int * filter, char * result){

        int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        long localCount = resPsum[startIndex];

        for(long i=startIndex;i<num;i+=stride){
                if(filter[i] != 0){
                        ((int*)result)[localCount] = ((int *)fact)[i];
                        localCount ++;
                }
        }
}

__global__ void static joinDim_int(int *resPsum, char * dim, int attrSize, long num, int *factF,int * filter, char * result){

        int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        long localCount = resPsum[startIndex];

        for(long i=startIndex;i<num;i+=stride){
                if( filter[i] != 0){
                	int dimId = factF[i];
                        ((int*)result)[localCount] = ((int*)dim)[dimId-1];
                        localCount ++;
                }
        }
}

__global__ void static joinDim_other(int *resPsum, char * dim, int attrSize, long num, int* factF,int * filter, char * result){

        int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        long localOffset = resPsum[startIndex] * attrSize;

        for(long i=startIndex;i<num;i+=stride){
                if( filter[i] != 0){
                	int dimId = factF[i];
                        memcpy(result + localOffset, dim + (dimId-1)* attrSize, attrSize);
                        localOffset += attrSize;
                }
        }
}



__global__ void static merge(int ** filter, long fNum, int dNum,int * result, int *count, int * totalCount){

	int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int lcount = 0;

	for(long i=startIndex; i<fNum; i+=stride){
		int tmp = 1;
		for(int j=0;j<dNum;j++){
			if(filter[j][i] == 0){
				tmp = 0;
				break;
			}
		}
		lcount += tmp;
		result[i] = tmp; 
	}
	count[startIndex] = lcount;
	atomicAdd(totalCount,lcount);
}

static void buildHashPlan(long size, int * pass){
	int gpuMem = getGpuGlobalMem(0);

	*pass = 3*size / gpuMem + 1; 
}


struct tableNode * inviJoin(struct joinNode *jNode, struct statistic *pp){

	struct tableNode * res = NULL;
	char ** gpu_fact;
	char ** gpu_hash;
	int ** gpuHashPsum;
	int ** gpuHashNum;
	int ** gpuFilter;

	struct timespec start,end;
	float gpuTime;

	cudaEvent_t startGPU, stopGPU;
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	clock_gettime(CLOCK_REALTIME,&start);

	dim3 grid(1024);
	dim3 block(256);

	int blockNum = jNode->factTable->tupleNum / block.x + 1;
	if(blockNum < 1024)
		grid = blockNum;

	int threadNum = grid.x * block.x;

	res = (struct tableNode *) malloc(sizeof(struct tableNode));
	res->tupleSize = jNode->tupleSize;
	res->totalAttr = jNode->totalAttr;
	res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataFormat = (int *) malloc(res->totalAttr * sizeof(int));

	res->content = (char **) malloc(sizeof(char *) * jNode->totalAttr );

	for(int i=0;i<res->totalAttr;i++){
		res->attrType[i] = jNode->attrType[i];
		res->attrSize[i] = jNode->attrSize[i];
		if(jNode->keepInGpu[i] == 1){
			res->dataPos[i] = GPU;
		}else{
			res->dataPos[i] = MEM;
		}
		res->dataFormat[i] = UNCOMPRESSED;
	}

	gpuHashPsum = (int **) malloc(sizeof(int *) * jNode->dimNum);
	gpuHashNum = (int **) malloc(sizeof(int *) * jNode->dimNum);
	gpu_hash = (char **)malloc(sizeof(char *) * jNode->dimNum);

	for(int k=0;k<jNode->dimNum;k++){
		char *gpu_dim;

		long primaryKeySize = sizeof(int) * jNode->dimTable[k]->tupleNum;

		int dimIndex = jNode->dimIndex[k];

/*
 * 	build hash table on GPU
 */


		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&(gpuHashNum[k]),HSIZE * sizeof(int)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&(gpuHashPsum[k]), sizeof(int) * HSIZE));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuHashNum[k],0,sizeof(int) * HSIZE));

		int pass = 0;
		int dimInGpu = 0;
		buildHashPlan(primaryKeySize,&pass);

		if (pass != 1){
			printf("Hash Table too large! not supported yet!");
			exit(-1);
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&(gpu_hash[k]), 2 * primaryKeySize));

		if(jNode->dimTable[k]->dataPos[dimIndex] == MEM){
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->dimTable[k]->content[dimIndex], primaryKeySize,cudaMemcpyHostToDevice));

		}else if(jNode->dimTable[k]->dataPos[dimIndex] == GPU){
			dimInGpu = 1 ;
			gpu_dim = jNode->dimTable[k]->content[dimIndex]; 
		}

		cudaEventRecord(startGPU,0);
		count_hash_num<<<grid,block>>>(gpu_dim,jNode->dimTable[k]->tupleNum,gpuHashNum[k]);

		CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

		cudaEventRecord(stopGPU,0);
		cudaEventSynchronize(stopGPU);
		cudaEventElapsedTime(&gpuTime,startGPU,stopGPU);

		pp->kernel += gpuTime;
		//printf("GPU count hash result time:%lf\n",gpuTime);

		scanImpl(gpuHashNum[k],HSIZE,gpuHashPsum[k], pp);

		int * gpu_psum ;

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_psum, sizeof(int) * HSIZE));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_psum, gpuHashPsum[k], sizeof(int) * HSIZE, cudaMemcpyDeviceToDevice));

		cudaEventRecord(startGPU,0);
		build_hash_table<<<grid,block>>>(gpu_dim,jNode->dimTable[k]->tupleNum,gpu_psum,gpu_hash[k]);

		CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

		CUDA_SAFE_CALL_NO_SYNC(cudaEventRecord(stopGPU,0));
		cudaEventSynchronize(stopGPU);
		cudaEventElapsedTime(&gpuTime,startGPU,stopGPU);
		pp->kernel += gpuTime;
		//printf("GPU build hash table time:%lf\n",gpuTime);

		if(dimInGpu == 0)
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_dim));

		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum));

	}

	int filterSize = jNode->factTable->tupleNum * sizeof(int);
	int *factInGpu = (int *) malloc(sizeof(int) * jNode->dimNum);

	gpuFilter = (int **) malloc(sizeof(int *) * jNode->dimNum);
	gpu_fact = (char **) malloc(sizeof(char *) * jNode->dimNum);

	for(int k=0;k<jNode->dimNum;k++){
		int index = jNode->factIndex[k];

		if(jNode->factTable->dataPos[index] == MEM){
			factInGpu[k] = 0;
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&(gpu_fact[k]), filterSize));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact[k],jNode->factTable->content[index], filterSize, cudaMemcpyHostToDevice));

		}else if(jNode->factTable->dataPos[index] == GPU){
			factInGpu[k] = 1;
			gpu_fact[k] = jNode->factTable->content[index];
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&(gpuFilter[k]), filterSize));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuFilter[k],0,filterSize));
	}

	cudaEventRecord(startGPU,0);

	for(int k=0;k<jNode->dimNum;k++)
		count_join_result<<<grid,block>>>(gpuHashNum[k], gpuHashPsum[k], gpu_hash[k], gpu_fact[k], jNode->factTable->tupleNum, gpuFilter[k]);

	CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

	cudaEventRecord(stopGPU,0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&gpuTime,startGPU,stopGPU);
	//printf("GPU generate filter time:%lf\n",gpuTime);
	pp->kernel += gpuTime;

	for(int k=0;k<jNode->dimNum;k++){
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_hash[k]));
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuHashPsum[k]));
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuHashNum[k]));

		if(factInGpu[k] == 0)
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact[k]));
	}

	free(gpu_hash);
	free(gpuHashPsum);
	free(gpuHashNum);
	free(gpu_fact);
	free(factInGpu);

	int * gpuFinalFilter;
	int * gpuCount, *gpuTotalCount;
	int * gpuResPsum;
	int ** gpuFilterAddr;

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &gpuFilterAddr, sizeof(int *) * jNode->dimNum));

	for(int k=0;k<jNode->dimNum;k++){
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&(gpuFilterAddr[k]), &(gpuFilter[k]), sizeof(int *), cudaMemcpyHostToDevice));
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFinalFilter,sizeof(int) * jNode->factTable->tupleNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuCount,sizeof(int) *  threadNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuTotalCount,sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuTotalCount,0,sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuResPsum,sizeof(int) * threadNum));

	cudaEventRecord(startGPU,0);
	merge<<<grid,block>>>(gpuFilterAddr,jNode->factTable->tupleNum,jNode->dimNum,gpuFinalFilter, gpuCount,gpuTotalCount);
	CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

	cudaEventRecord(stopGPU,0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&gpuTime,startGPU,stopGPU);
	//printf("GPU merge filter time:%lf\n",gpuTime);
	pp->kernel += gpuTime;

	int totalCount = 0;

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&totalCount,gpuTotalCount,sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuTotalCount));

	res->tupleNum = totalCount;

	for(int i=0;i<res->totalAttr;i++){
		res->attrTotalSize[i] = totalCount * res->attrSize[i];
	}

	scanImpl(gpuCount,threadNum,gpuResPsum, pp);

	gpu_fact = (char **) malloc(sizeof(char *) * jNode->totalAttr);
	factInGpu = (int *) malloc(sizeof(int) * jNode->totalAttr);
	char **gpuResult = (char **) malloc(sizeof(char *) * jNode->totalAttr);
	int *attrSize = (int *) malloc(sizeof(int) * jNode->totalAttr);
	int *attrType = (int *) malloc(sizeof(int) * jNode->totalAttr);

	for(int i=0; i< jNode->factOutputNum;i++){
		int index = jNode->factOutputIndex[i];
		int aSize = jNode->factTable->attrSize[index];
		int size = aSize * jNode->factTable->tupleNum;
		attrSize[i] = aSize;
		attrType[i] = jNode->factTable->attrType[index];
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&(gpuResult[i]),aSize * totalCount));
		if(jNode->factTable->dataPos[index] == MEM ){
			factInGpu[i] = 0;
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&(gpu_fact[i]),size));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact[i], jNode->factTable->content[index],aSize * jNode->factTable->tupleNum, cudaMemcpyHostToDevice));
		}else if(jNode->factTable->dataPos[index] == GPU){
			factInGpu[i] = 1;
			gpu_fact[i] = jNode->factTable->content[index];
		}
	}

	int k = jNode->factOutputNum;
	for(int i=0;i<jNode->dimNum;i++){
		for(int j=0;j<jNode->dimOutputNum[i]; j++){
			int index = jNode->dimOutputIndex[i][j];
			int aSize = jNode->dimTable[i]->attrSize[index];
			attrSize[k] = aSize;
			attrType[k] = jNode->dimTable[i]->attrType[index];
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&(gpuResult[k]),aSize * totalCount));
			if(jNode->dimTable[i]->dataPos[index] == MEM){
				factInGpu[k] = 0;
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&(gpu_fact[k]),aSize*jNode->dimTable[i]->tupleNum));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact[k],jNode->dimTable[i]->content[index], aSize*jNode->dimTable[i]->tupleNum,cudaMemcpyHostToDevice));
			}else if (jNode->dimTable[i]->dataPos[index] ==GPU){
				factInGpu[k] = 1;
				gpu_fact[k] = jNode->dimTable[i]->content[index];
			}
			k++;
		}
	}

	cudaEventRecord(startGPU,0);

	for(int i=0;i<jNode->factOutputNum;i++){
		if(attrType[i] != STRING)
			joinFact_int<<<grid,block>>>(gpuResPsum,gpu_fact[i],attrSize[i],jNode->factTable->tupleNum,gpuFinalFilter,gpuResult[i]);
		else
			joinFact_other<<<grid,block>>>(gpuResPsum,gpu_fact[i],attrSize[i],jNode->factTable->tupleNum,gpuFinalFilter,gpuResult[i]);

	}

	k = jNode->factOutputNum;
	for(int i=0;i<jNode->dimNum;i++){

		for(int j=0;j<jNode->dimOutputNum[i];j++){
			if (attrType[k] != STRING)
				joinDim_int<<<grid,block>>>(gpuResPsum,gpu_fact[k],attrSize[k],jNode->factTable->tupleNum,gpuFilter[k],gpuFinalFilter,gpuResult[k]);
			else
				joinDim_other<<<grid,block>>>(gpuResPsum,gpu_fact[k],attrSize[k],jNode->factTable->tupleNum,gpuFilter[k],gpuFinalFilter,gpuResult[k]);
			k++;
		}
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

	cudaEventRecord(stopGPU,0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&gpuTime,startGPU,stopGPU);

	pp->kernel += gpuTime;
	//printf("GPU filter fact result time:%lf\n",gpuTime);

	for(int i=0;i<jNode->dimNum;i++){
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuFilter[i]));
	}
	for(int i=0;i<jNode->totalAttr;i++){
		if(factInGpu[i] == 0)
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact[i]));
	}

	for(int i=0;i<jNode->factOutputNum;i++){
		int pos = jNode->factOutputPos[i];
		if(res->dataPos[pos] == MEM){
			res->content[pos] = (char *) malloc(res->tupleNum * res->attrSize[pos]);
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[pos], gpuResult[i], res->tupleNum * res->attrSize[pos],cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResult[i]));
		}else if(res->dataPos[pos] == GPU){
			res->content[pos] = gpuResult[i];
		}
	}
	for(int i=0;i<jNode->dimOutputTotal;i++){
		int pos = jNode->dimOutputPos[i];
		if(res->dataPos[pos] == MEM){
			res->content[pos] = (char *) malloc(res->tupleNum * res->attrSize[pos]);
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[pos], gpuResult[i+jNode->factOutputNum], res->tupleNum * res->attrSize[pos],cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResult[i+jNode->factOutputNum]));
		}else if(res->dataPos[pos] == GPU){
			res->content[pos] = gpuResult[i+jNode->factOutputNum];
		}
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuFinalFilter));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResPsum));

	clock_gettime(CLOCK_REALTIME,&end);
	double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
	pp->total += timeE / (1000 * 1000);
	return res;

}
