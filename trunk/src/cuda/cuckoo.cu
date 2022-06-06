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
#include "../include/cpuCudaLib.h"
#include "scanImpl.cu"

#define CUCKOO_SIZE	512
#define	SUBSIZE		(CUCKOO_SIZE)
#define SEED_NUM	256
#define PRIME		1900813

#define NP2(n)              do {                    \
    n--;                                            \
    n |= n >> 1;                                    \
    n |= n >> 2;                                    \
    n |= n >> 4;                                    \
    n |= n >> 8;                                    \
    n |= n >> 16;                                   \
    n ++; } while (0)


// if the foreign key is compressed using dict-encoding, call this method to generate dict filter first
__global__ static void count_join_result_dict(int *num, int* psum, char* bucket, char* fact, int dNum, int* dictFilter){

	int stride = blockDim.x * gridDim.x;
	int offset = blockIdx.x*blockDim.x + threadIdx.x;

	struct dictHeader *dheader;
	dheader = (struct dictHeader *) fact;
	
	for(int i=offset;i<dNum;i+=stride){
		int fkey = dheader->hash[i];
		int hkey = fkey &(HSIZE-1);
		int keyNum = num[hkey];

		for(int j=0;j<keyNum;j++){
			int pSum = psum[hkey];
			int dimKey = ((int *)(bucket))[2*j + 2*pSum];
			int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
			if( dimKey == fkey){
				dictFilter[i] = dimId;
				break;
			}
		}
	}

}

// transform the dictionary filter to the final filter than can be used to generate the result

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


// count the number that is not zero in the filter
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


// if the foreign key is compressed using rle, call this method to generate join filter
__global__ static void count_join_result_rle(int* num, int* psum, char* bucket, char* fact, long tupleNum, long tupleOffset,  int * factFilter){

	int stride = blockDim.x * gridDim.x;
	long offset = blockIdx.x*blockDim.x + threadIdx.x;

	struct rleHeader *rheader = (struct rleHeader *)fact;
	int dNum = rheader->dictNum;

	for(int i=offset; i<dNum; i += stride){
		int fkey = ((int *)(fact+sizeof(struct rleHeader)))[i];
		int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
		int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

		if((fcount + fpos) < tupleOffset)
			continue;

		if(fpos >= (tupleOffset + tupleNum))
			break;

		int hkey = fkey &(HSIZE-1);
		int keyNum = num[hkey];

		for(int j=0;j<keyNum;j++){

			int pSum = psum[hkey];
			int dimKey = ((int *)(bucket))[2*j + 2*pSum];
			int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];

			if( dimKey == fkey){

				if(fpos < tupleOffset){
					int tcount = fcount + fpos - tupleOffset;
					if(tcount > tupleNum)
						tcount = tupleNum;
					for(int k=0;k<tcount;k++)
						factFilter[k] = dimId;

				}else if((fpos + fcount) > (tupleOffset + tupleNum)){
					int tcount = tupleOffset + tupleNum - fpos ;
					for(int k=0;k<tcount;k++)
						factFilter[fpos+k-tupleOffset] = dimId;
				}else{
					for(int k=0;k<fcount;k++)
						factFilter[fpos+k-tupleOffset] = dimId;

				}

				break;
			}
		}
	}

}

// if the foreign key is not compressed at all, call this method to generate join filter
__global__ static void count_join_result(int * seeds, int* hashTable, int bucketNum,char* fact, long inNum, int* count, int * factFilter){
	int lcount = 0;
	int stride = blockDim.x * gridDim.x;
	long offset = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int con[6];

	for(int i=offset;i<inNum;i+=stride){
		int fkey = ((int *)(fact))[i];
		int bn = fkey % PRIME % bucketNum;
		int seed = seeds[bn];
		con[0] = seed ^ 0xffff;
                con[1] = seed ^ 0xcba9;
                con[2] = seed ^ 0x7531;
                con[3] = seed ^ 0xbeef;
                con[4] = seed ^ 0xd9f1;
                con[5] = seed ^ 0x337a;

		int htOffset = 3*2*SUBSIZE * bn;

		int index1 = (con[0] *fkey + con[1]) % PRIME % SUBSIZE;
		int index2 = (con[2] *fkey + con[3]) % PRIME % SUBSIZE;
		int index3 = (con[4] *fkey + con[5]) % PRIME % SUBSIZE;

		if(fkey == hashTable[htOffset+ 2*index1]){
			factFilter[i] = hashTable[htOffset+2*index1 +1];
			lcount ++;
		}else if(fkey == hashTable[htOffset+2*SUBSIZE + 2*index2]){
			factFilter[i] = hashTable[htOffset+2*SUBSIZE + 2*index2 +1];
			lcount ++;
		}else if(fkey == hashTable[htOffset+ 4*SUBSIZE +  2*index3]){
			factFilter[i] = hashTable[htOffset+4*SUBSIZE + 2*index3 +1];
			lcount ++;
		}

	}

	count[offset] = lcount;
}

// unpack the column that is compresses using Run Length Encoding

__global__ void static unpack_rle(char * fact, char * rle, long tupleNum, long tupleOffset, int dNum){

	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=offset; i<dNum; i+=stride){

		int fvalue = ((int *)(fact+sizeof(struct rleHeader)))[i];
		int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
		int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

		if((fcount + fpos) < tupleOffset)
			continue;

		if(fpos >= (tupleOffset + tupleNum))
			break;

		if(fpos < tupleOffset){
			int tcount = fcount + fpos - tupleOffset;
			if(tcount > tupleNum)
				tcount = tupleNum;
			for(int k=0;k<tcount;k++){
				((int*)rle)[k] = fvalue; 
			}

		}else if ((fpos + fcount) > (tupleOffset + tupleNum)){
			int tcount = tupleNum  + tupleOffset - fpos;
			for(int k=0;k<tcount;k++){
				((int*)rle)[fpos-tupleOffset + k] = fvalue;
			}

		}else{
			for(int k=0;k<fcount;k++){
				((int*)rle)[fpos-tupleOffset + k] = fvalue;
			}
		}
	}
}

// generate psum for RLE compressed column based on filter
// current implementaton: scan through rle element and find the correponsding element in the filter

__global__ void static rle_psum(int *count, char * fact,  long  tupleNum, long tupleOffset, int * filter){

	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	struct rleHeader *rheader = (struct rleHeader *) fact;
	int dNum = rheader->dictNum;

	for(int i= offset; i<dNum; i+= stride){

		int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
		int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];
		int lcount= 0;

		if((fcount + fpos) < tupleOffset)
			continue;

		if(fpos >= (tupleOffset + tupleNum))
			break;

		if(fpos < tupleOffset){
			int tcount = fcount + fpos - tupleOffset;
			if(tcount > tupleNum)
				tcount = tupleNum;
			for(int k=0;k<tcount;k++){
				if(filter[k]!=0)
					lcount++;
			}
			count[i] = lcount;

		}else if ((fpos + fcount) > (tupleOffset + tupleNum)){
			int tcount = tupleNum  + tupleOffset - fpos;
			for(int k=0;k<tcount;k++){
				if(filter[fpos-tupleOffset + k]!=0)
					lcount++;
			}
			count[i] = lcount;

		}else{
			for(int k=0;k<fcount;k++){
				if(filter[fpos-tupleOffset + k]!=0)
					lcount++;
			}
			count[i] = lcount;
		}
	}

}

//filter the column that is compressed using Run Length Encoding
//current implementation:

__global__ void static joinFact_rle(int *resPsum, char * fact,  int attrSize, long  tupleNum, long tupleOffset, int * filter, char * result){

	int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	struct rleHeader *rheader = (struct rleHeader *) fact;
	int dNum = rheader->dictNum;

	for(int i = startIndex; i<dNum; i += stride){
		int fkey = ((int *)(fact+sizeof(struct rleHeader)))[i];
		int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
		int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

		if((fcount + fpos) < tupleOffset)
			continue;

		if(fpos >= (tupleOffset + tupleNum))
			break;

		if(fpos < tupleOffset){
			int tcount = fcount + fpos - tupleOffset;
			int toffset = resPsum[i];
			for(int j=0;j<tcount;j++){
				if(filter[j] != 0){
					((int*)result)[toffset] = fkey ;
					toffset ++;
				}
			}

		}else if ((fpos + fcount) > (tupleOffset + tupleNum)){
			int tcount = tupleOffset + tupleNum - fpos;
			int toffset = resPsum[i];
			for(int j=0;j<tcount;j++){
				if(filter[fpos-tupleOffset+j] !=0){
					((int*)result)[toffset] = fkey ;
					toffset ++;
				}
			}

		}else{
			int toffset = resPsum[i];
			for(int j=0;j<fcount;j++){
				if(filter[fpos-tupleOffset+j] !=0){
					((int*)result)[toffset] = fkey ;
					toffset ++;
				}
			}
		}
	}

}

// filter the column in the fact table that is compressed using dictionary encoding
__global__ void static joinFact_dict_other(int *resPsum, char * fact,  char *dict, int byteNum,int attrSize, long  num, int * filter, char * result){

	int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	long localOffset = resPsum[startIndex] * attrSize;

	struct dictHeader *dheader = (struct dictHeader*)dict;

	for(long i=startIndex;i<num;i+=stride){
		if(filter[i] != 0){
			int key = 0;
			memcpy(&key, fact + i* byteNum, byteNum);
			memcpy(result + localOffset, &dheader->hash[key], attrSize);
			localOffset += attrSize;
		}
	}
}

__global__ void static joinFact_dict_int(int *resPsum, char * fact, char *dict, int byteNum, int attrSize, long  num, int * filter, char * result){

	int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	long localCount = resPsum[startIndex];

	struct dictHeader *dheader = (struct dictHeader*)dict;

	for(long i=startIndex;i<num;i+=stride){
		if(filter[i] != 0){
			int key = 0;
			memcpy(&key, fact + i* byteNum, byteNum);
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

__global__ void static joinDim_rle(int *resPsum, char * dim, int attrSize, long tupleNum, long tupleOffset, int * filter, char * result){

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

__global__ void static joinDim_dict_other(int *resPsum, char * dim, char *dict, int byteNum, int attrSize, long num,int * filter, char * result){

	int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	long localOffset = resPsum[startIndex] * attrSize;

	struct dictHeader *dheader = (struct dictHeader*)dict;

	for(long i=startIndex;i<num;i+=stride){
		int dimId = filter[i];
		if( dimId != 0){
			int key = 0;
			memcpy(&key, dim + (dimId-1) * byteNum, byteNum);
			memcpy(result + localOffset, &dheader->hash[key], attrSize);
			localOffset += attrSize;
		}
	}
}

__global__ void static joinDim_dict_int(int *resPsum, char * dim, char *dict, int byteNum, int attrSize, long num,int * filter, char * result){

	int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	long localCount = resPsum[startIndex];

	struct dictHeader *dheader = (struct dictHeader*)dict;

	for(long i=startIndex;i<num;i+=stride){
		int dimId = filter[i];
		if( dimId != 0){
			int key = 0;
			memcpy(&key, dim + (dimId-1) * byteNum, byteNum);
			((int*)result)[localCount] = dheader->hash[key];
			localCount ++;
		}
	}
}

__global__ void static joinDim_int(int *resPsum, char * dim, int attrSize, long num,int * filter, char * result){

	int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	long localCount = resPsum[startIndex];

	for(long i=startIndex;i<num;i+=stride){
		int dimId = filter[i];
		if( dimId != 0){
			((int*)result)[localCount] = ((int*)dim)[dimId-1];
			localCount ++;
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


__global__ void static preshuffle(char *dim, int tupleNum, int bucketNum, int * bucketID, int *offset, int *count){

	int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=startIndex;i<tupleNum; i+=stride){
		int key = ((int *)dim)[i]; 
		int bk = key % PRIME % bucketNum;

		offset[i] = atomicAdd(&count[bk],1);
		bucketID[i] = bk;
	}

}

__global__ void static shuffle_data(char *dim,int tupleNum,int bucketNum, int *start, int * offset, int *bucketStart, char *result, int * rid){

	int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=startIndex;i<tupleNum;i+=stride){
		int key = ((int *)dim)[i];
		int bk = start[i];
		int pos = offset[i] + bucketStart[bk];
		((int*)result)[pos] = key;
		rid[pos] = i;
	}
}


__global__ void static cuckooHash(char *dim, int * rid, int * bucketOffset,int* bucketCount,int* hashTable, int *seedInput, int seedNum,int * seeds){

	__shared__ int buf[CUCKOO_SIZE];
	__shared__ int bufId[CUCKOO_SIZE];
	__shared__ int sub1[2*SUBSIZE];
	__shared__ int sub2[2*SUBSIZE];
	__shared__ int sub3[2*SUBSIZE];
	__shared__ int buildFinish[1];

	int bkNum = bucketCount[blockIdx.x];
	int offset = bucketOffset[blockIdx.x];

	if(threadIdx.x<bkNum){
		buf[threadIdx.x] = ((int *)dim)[threadIdx.x+offset];
		bufId[threadIdx.x] = rid[threadIdx.x+offset];
	}

	int count = 0;
	unsigned int seed;

	if(threadIdx.x<bkNum){

		unsigned int con[6];
	
		int key = buf[threadIdx.x];
		int id = bufId[threadIdx.x];

		do{

			seed = seedInput[count];
			for(int i=threadIdx.x;i<SUBSIZE;i+=bkNum){

				sub1[2*i] = sub1[2*i+1] = -1;
				sub2[2*i] = sub2[2*i+1] = -1;
				sub3[2*i] = sub3[2*i+1] = -1;
			}
			__syncthreads();

			if(threadIdx.x == 0)
				*buildFinish = 1;

			int inSubTable = 0;
			con[0] = seed ^ 0xffff;
			con[1] = seed ^ 0xcba9;
			con[2] = seed ^ 0x7531;
			con[3] = seed ^ 0xbeef;
			con[4] = seed ^ 0xd9f1;
			con[5] = seed ^ 0x337a;

			int index1 = (con[0] *key + con[1]) % PRIME % SUBSIZE;
			int index2 = (con[2] *key + con[3]) % PRIME % SUBSIZE;
			int index3 = (con[4] *key + con[5]) % PRIME % SUBSIZE;

			if(inSubTable == 0 ){
				inSubTable = 1;
				sub1[2*index1] = key;
				sub1[2*index1+1] = id;
				__syncthreads();
			}

			if(inSubTable == 1 && sub1[2*index1] != key){
				inSubTable = 2;
				sub2[2*index2] = key;
				sub2[2*index2+1] = id; 
				__syncthreads();
			}


			if(inSubTable == 2 && sub2[2*index2] != key){
				inSubTable = 3;
				sub3[2*index3] = key;
				sub3[2*index3+1] = id;
				__syncthreads();
			}

			if(inSubTable == 3 && sub3[2*index3] != key){
				*buildFinish = 0;
			}
			__syncthreads();

			if(*buildFinish == 1)
				break;

			count ++;

		}while(count <seedNum);

	}

	offset = blockIdx.x * 3 * 2*SUBSIZE;
	for(int i=threadIdx.x;i<2*SUBSIZE;i += blockDim.x){
		hashTable[i+offset] = sub1[i];
	}

	__syncthreads();

	for(int i=threadIdx.x;i<2*SUBSIZE;i += blockDim.x){
		hashTable[i + offset + 2*SUBSIZE] = sub2[i];
	}

	__syncthreads();
	for(int i=threadIdx.x;i<2*SUBSIZE;i += blockDim.x){
		hashTable[i + offset + 4*SUBSIZE] = sub3[i];
	}

	__syncthreads();

	seeds[blockIdx.x] = seed;
	
}


/*
 * cuckooHashJoin implements the foreign key join between a fact table and dimension table.
 *
 * Prerequisites:
 *	1. the data to be joined can be fit into GPU device memory.
 *	2. dimension table is not compressed
 *	
 * Input:
 *	jNode: contains information about the two joined tables.
 *	pp: records statistics such as kernel execution time
 *
 * Output:
 * 	A new table node
 */

struct tableNode * cuckooHashJoin(struct joinNode *jNode, struct statistic *pp){
	struct tableNode * res = NULL;

	int *cpu_count, *resPsum;
	int count = 0;
	int i;

	int * gpu_hashNum;
	char * gpu_result;
	char  *gpu_bucket, *gpu_fact, * gpu_dim;
	int * gpu_count,  *gpu_psum, *gpu_resPsum;

	int defaultBlock = 2048;

	dim3 grid(defaultBlock);
	dim3 block(CUCKOO_SIZE);
	int blockNum;
	int threadNum;

	blockNum = jNode->leftTable->tupleNum / block.x + 1;
	if(blockNum < defaultBlock)
		grid = blockNum;
	else
		grid = defaultBlock;

	threadNum = grid.x * block.x;

	res = (struct tableNode*) malloc(sizeof(struct tableNode));
	initTable(res);
	res->totalAttr = jNode->totalAttr;
	res->tupleSize = jNode->tupleSize;
	res->attrType = (int *) malloc(res->totalAttr * sizeof(int));
	res->attrSize = (int *) malloc(res->totalAttr * sizeof(int));
	res->attrIndex = (int *) malloc(res->totalAttr * sizeof(int));
	res->attrTotalSize = (int *) malloc(res->totalAttr * sizeof(int));
	res->dataPos = (int *) malloc(res->totalAttr * sizeof(int));
	res->dataFormat = (int *) malloc(res->totalAttr * sizeof(int));
	res->content = (char **) malloc(res->totalAttr * sizeof(char *));

	for(i=0;i<jNode->leftOutputAttrNum;i++){
		int pos = jNode->leftPos[i];
		res->attrType[pos] = jNode->leftOutputAttrType[i];
		int index = jNode->leftOutputIndex[i];
		res->attrSize[pos] = jNode->leftTable->attrSize[index];
		res->dataFormat[pos] = UNCOMPRESSED;
	}

	for(i=0;i<jNode->rightOutputAttrNum;i++){
		int pos = jNode->rightPos[i];
		res->attrType[pos] = jNode->rightOutputAttrType[i];
		int index = jNode->rightOutputIndex[i];
		res->attrSize[pos] = jNode->rightTable->attrSize[index];
		res->dataFormat[pos] = UNCOMPRESSED;
	}

	long primaryKeySize = sizeof(int) * jNode->rightTable->tupleNum;

/*
 * 	build hash table on GPU
 */

	int *gpu_psum1;


	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_hashNum,sizeof(int)*HSIZE));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_hashNum,0,sizeof(int)*HSIZE));

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_count,sizeof(int)*threadNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_resPsum,sizeof(int)*threadNum));

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_psum,HSIZE*sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_bucket, 2*primaryKeySize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_psum1,HSIZE*sizeof(int)));

	int dataPos = jNode->rightTable->dataPos[jNode->rightKeyIndex];

	if(dataPos == MEM || dataPos == PINNED){
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));

	}else if (dataPos == GPU || dataPos == UVA){
		gpu_dim = jNode->rightTable->content[jNode->rightKeyIndex];
	}

	int bucketSize = CUCKOO_SIZE;
	int bucketNum = (jNode->rightTable->tupleNum + bucketSize -1 )/ bucketSize ;
	bucketNum *= 1.2;

	int * gpuBucketCount, *gpuBucketOffset, *gpuBucketID;
	int * gpuBucketPsum;

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuBucketCount, sizeof(int) * bucketNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuBucketCount,0, sizeof(int) * bucketNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuBucketID, jNode->rightTable->tupleNum * sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuBucketOffset,jNode->rightTable->tupleNum * sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuBucketPsum,sizeof(int) * bucketNum));

	preshuffle<<<grid,block>>>(gpu_dim,jNode->rightTable->tupleNum,bucketNum,gpuBucketID,gpuBucketOffset,gpuBucketCount);

	scanImpl(gpuBucketCount,bucketNum,gpuBucketPsum,pp);

	char *gpuBucketData;
	int * gpuRid;

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuBucketData,jNode->rightTable->tupleNum * sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRid,jNode->rightTable->tupleNum * sizeof(int)));

	shuffle_data<<<grid,block>>>(gpu_dim,jNode->rightTable->tupleNum,bucketNum,gpuBucketID,gpuBucketOffset,gpuBucketPsum, gpuBucketData, gpuRid);

	int * gpuHash;
	int * gpuSeeds;

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuHash, bucketNum*2*3*SUBSIZE*sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuSeeds, sizeof(int)*bucketNum));

	int inputSeed[SEED_NUM];
	struct timespec now;

        clock_gettime(CLOCK_REALTIME,&now);
	for(int i=0;i<SEED_NUM;i++){
		srand(i+1);
		inputSeed[i] = random();
	}

	int * gpuRandomSeed;
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuRandomSeed,sizeof(inputSeed)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuRandomSeed,inputSeed,sizeof(inputSeed),cudaMemcpyHostToDevice));

	cuckooHash<<<bucketNum,block>>>(gpu_dim,gpuRid, gpuBucketPsum,gpuBucketCount, gpuHash, gpuRandomSeed,SEED_NUM,gpuSeeds);

	if (dataPos == MEM || dataPos == PINNED)
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_dim));

	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRandomSeed));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum1));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuBucketID));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuBucketOffset));


/*
 *	join on GPU
 */

	int *gpuFactFilter;

	dataPos = jNode->leftTable->dataPos[jNode->leftKeyIndex];
	int format = jNode->leftTable->dataFormat[jNode->leftKeyIndex];

	long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
	long filterSize = jNode->leftTable->attrSize[jNode->leftKeyIndex] * jNode->leftTable->tupleNum;

	if(dataPos == MEM || dataPos == PINNED){
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact, foreignKeySize));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));

	}else if (dataPos == GPU || dataPos == UVA){
		gpu_fact = jNode->leftTable->content[jNode->leftKeyIndex];
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFactFilter,filterSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuFactFilter,0,filterSize));

	if(format == UNCOMPRESSED)
		count_join_result<<<grid,block>>>(gpuSeeds, gpuHash, bucketNum,gpu_fact, jNode->leftTable->tupleNum, gpu_count,gpuFactFilter);

	else if(format == DICT){
		int dNum;
		struct dictHeader * dheader;

		if(dataPos == MEM || dataPos == UVA || dataPos == PINNED){
			dheader = (struct dictHeader *) jNode->leftTable->content[jNode->leftKeyIndex];
			dNum = dheader->dictNum;

		}else if (dataPos == GPU){
			dheader = (struct dictHeader *) malloc(sizeof(struct dictHeader));
			memset(dheader,0,sizeof(struct dictHeader));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(dheader,gpu_fact,sizeof(struct dictHeader), cudaMemcpyDeviceToHost));
			dNum = dheader->dictNum;
		}
		free(dheader);

		int * gpuDictFilter;
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuDictFilter, 0 ,dNum * sizeof(int)));


		count_join_result_dict<<<grid,block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, dNum, gpuDictFilter);
		CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

		transform_dict_filter<<<grid,block>>>(gpuDictFilter, gpu_fact, jNode->leftTable->tupleNum, dNum, gpuFactFilter);
		CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));

		filter_count<<<grid,block>>>(jNode->leftTable->tupleNum, gpu_count, gpuFactFilter);

	}else if (format == RLE){

		count_join_result_rle<<<512,64>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum, 0,gpuFactFilter);
		CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

		filter_count<<<grid, block>>>(jNode->leftTable->tupleNum, gpu_count, gpuFactFilter);
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

	cpu_count = (int *) malloc(sizeof(int)*threadNum);
	memset(cpu_count,0,sizeof(int)*threadNum);
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(cpu_count,gpu_count,sizeof(int)*threadNum,cudaMemcpyDeviceToHost));
	resPsum = (int *) malloc(sizeof(int)*threadNum);
	memset(resPsum,0,sizeof(int)*threadNum);
	scanImpl(gpu_count,threadNum,gpu_resPsum, pp);

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(resPsum,gpu_resPsum,sizeof(int)*threadNum,cudaMemcpyDeviceToHost));

	count = resPsum[threadNum-1] + cpu_count[threadNum-1];
	res->tupleNum = count;
	printf("joinNum %d\n",count);

	if(dataPos == MEM || dataPos == PINNED){
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_bucket));
		
	free(resPsum);
	free(cpu_count);

	for(i=0; i<res->totalAttr; i++){
		int index, pos;
		long colSize = 0, resSize = 0;
		int leftRight = 0;

		int attrSize, attrType;
		char * table;
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
			index = jNode->leftOutputIndex[pos];
			dataPos = jNode->leftTable->dataPos[index];
			format = jNode->leftTable->dataFormat[index];

			table = jNode->leftTable->content[index];
			attrSize  = jNode->leftTable->attrSize[index];
			attrType  = jNode->leftTable->attrType[index];
			colSize = jNode->leftTable->attrTotalSize[index];

			resSize = res->tupleNum * attrSize;
		}else{
			index = jNode->rightOutputIndex[pos];
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

				if(dataPos == MEM || dataPos == PINNED){
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
				}else{
					gpu_fact = table;
				}

				if(attrSize == sizeof(int))
					joinFact_int<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);
				else
					joinFact_other<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

			}else if (format == DICT){
				struct dictHeader * dheader;
				int byteNum;
				char * gpuDictHeader;

				assert(dataPos == MEM || dataPos == PINNED);

				dheader = (struct dictHeader *)table;
				byteNum = dheader->bitNum/8;
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuDictHeader,sizeof(struct dictHeader)));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));

				if(dataPos == MEM || dataPos == PINNED){
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table + sizeof(struct dictHeader), colSize-sizeof(struct dictHeader),cudaMemcpyHostToDevice));
				}else{
					gpu_fact = table + sizeof(struct dictHeader);
				}

				if (attrSize == sizeof(int))
					joinFact_dict_int<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);
				else
					joinFact_dict_other<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

			}else if (format == RLE){

				if(dataPos == MEM || dataPos == PINNED){
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
				}else{
					gpu_fact = table;
				}

				int dNum = (colSize - sizeof(struct rleHeader))/(3*sizeof(int));

				char * gpuRle;
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRle, jNode->leftTable->tupleNum * sizeof(int)));

				unpack_rle<<<grid,block>>>(gpu_fact, gpuRle,jNode->leftTable->tupleNum, 0, dNum);

				CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

				joinFact_int<<<grid,block>>>(gpu_resPsum,gpuRle, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));

			}

		}else{
			if(format == UNCOMPRESSED){

				if(dataPos == MEM || dataPos == PINNED){
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
				}else{
					gpu_fact = table;
				}

				if(attrType == sizeof(int))
					joinDim_int<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
				else
					joinDim_other<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);

			}else if (format == DICT){
				struct dictHeader * dheader;
				int byteNum;
				char * gpuDictHeader;
				assert(dataPos == MEM || dataPos == PINNED);

				dheader = (struct dictHeader *)table;
				byteNum = dheader->bitNum/8;
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuDictHeader,sizeof(struct dictHeader)));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));
				if(dataPos == MEM || dataPos == PINNED){
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table + sizeof(struct dictHeader), colSize-sizeof(struct dictHeader),cudaMemcpyHostToDevice));
				}else{
					gpu_fact = table + sizeof(struct dictHeader);
				}

				if(attrType == sizeof(int))
					joinDim_dict_int<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
				else
					joinDim_dict_other<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader, byteNum, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

			}else if (format == RLE){

				if(dataPos == MEM || dataPos == PINNED){
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
				}else{
					gpu_fact = table;
				}

				joinDim_rle<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, 0,gpuFactFilter,gpu_result);
			}
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

		
		res->attrTotalSize[i] = resSize;
		res->dataFormat[i] = UNCOMPRESSED;
		if(res->dataPos[i] == MEM){
			res->content[i] = (char *) malloc(resSize);
			memset(res->content[i],0,resSize);
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i],gpu_result,resSize,cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_result));

		}else if(res->dataPos[i] == GPU){
			res->content[i] = gpu_result;
		}
		if(dataPos == MEM || dataPos == PINNED)
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));

	}

	CUDA_SAFE_CALL(cudaFree(gpuFactFilter));

	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_count));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_hashNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_resPsum));

	return res;

}
