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
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "scanImpl.cu"
#include "../include/common.h"
#include "../include/gpuCudaLib.h"

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)

/*
 * stringCmp: Compare two strings on GPU using one single GPU thread.
 * @buf1: the first input buffer
 * @buf2: the second input buffer
 * @size: the length of data to be compared
 *
 * Return 
 *  1 if buf1 is larger
 *  0 if they are equal
 *  -1 if buf2 is larger
 */

__device__ static inline int stringCmp(char* buf1, char *buf2, int size){
    int i;
    int res = 0;
    for(i=0;i<size;i++){
        if(buf1[i] > buf2[i]){
            res = 1;
            break;
        }else if (buf1[i] < buf2[i]){
            res = -1;
            break;
        }
        if(buf1[i] == 0 && buf2[i] == 0)
            break;
    }
    return res;
}

/*
 * testCon: evaluate one selection predicate using one GPU thread
 * @buf1: input data to be tested
 * @buf2: the test criterion, usually a number of a string.
 * @size: the size of the input data buf1
 * @type: the type of the input data buf1 
 * @rel: >,<, >=, <= or ==.
 *
 * Return:
 *  0 if the input data meets the criteria
 *  1 otherwise
 */

__device__ static inline int testCon(char *buf1, char* buf2, int size, int type, int rel){
    int res = 1;
    if (type == INT){
        if(rel == EQ){
            res = ( *((int*)buf1) == *(((int*)buf2)) );
        }else if (rel == GTH){
            res = ( *((int*)buf1) > *(((int*)buf2)) );
        }else if (rel == LTH){
            res = ( *((int*)buf1) < *(((int*)buf2)) );
        }else if (rel == GEQ){
            res = ( *((int*)buf1) >= *(((int*)buf2)) );
        }else if (rel == LEQ){
            res = ( *((int*)buf1) <= *(((int*)buf2)) );
        }

    }else if (type == FLOAT){
        if(rel == EQ){
            res = ( *((float*)buf1) == *(((float*)buf2)) );
        }else if (rel == GTH){
            res = ( *((float*)buf1) > *(((float*)buf2)) );
        }else if (rel == LTH){
            res = ( *((float*)buf1) < *(((float*)buf2)) );
        }else if (rel == GEQ){
            res = ( *((float*)buf1) >= *(((float*)buf2)) );
        }else if (rel == LEQ){
            res = ( *((float*)buf1) <= *(((float*)buf2)) );
        }

    }else{
        int tmp = stringCmp(buf1,buf2,size);
        if(rel == EQ){
            res = (tmp == 0);
        }else if (rel == GTH){
            res = (tmp > 0);
        }else if (rel == LTH){
            res = (tmp < 0);
        }else if (rel == GEQ){
            res = (tmp >= 0);
        }else if (rel == LEQ){
            res = (tmp <= 0);
        }
    }
    return res;
}


/*
 * transform_dict_filter_and: merge the filter for dictionary-compressed predicate into the final filter.
 * @dictFilter: the filter for the dictionary compressed data
 * @dictFact: the compressed fact table column
 * @tupleNum: the number of tuples in the column
 * @filter: the filter for the uncompressed data
 */

__global__ static void transform_dict_filter_and(int * dictFilter, char *dictFact, long tupleNum, int dNum,  int * filter, int byteNum){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    int * fact = (int*)(dictFact + sizeof(struct dictHeader));

    int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ; 

    for(long i=offset; i<numInt; i += stride){
        int tmp = fact[i];
        unsigned long bit = 1;

        for(int j=0; j< sizeof(int)/byteNum; j++){
            int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
            int fkey = (tmp & t)>> (j*byteNum*8) ;
            filter[i* sizeof(int)/byteNum + j] &= dictFilter[fkey];
        }
    }
}

__global__ static void transform_dict_filter_init(int * dictFilter, char *dictFact, long tupleNum, int dNum,  int * filter,int byteNum){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    int * fact = (int*)(dictFact + sizeof(struct dictHeader));
    int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

    for(long i=offset; i<numInt; i += stride){
        int tmp = fact[i];
        unsigned long bit = 1;

        for(int j=0; j< sizeof(int)/byteNum; j++){
            int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
            int fkey = (tmp & t)>> (j*byteNum*8) ;
            filter[i* sizeof(int)/byteNum + j] = dictFilter[fkey];
        }
    }
}

__global__ static void transform_dict_filter_or(int * dictFilter, char *fact, long tupleNum, int dNum,  int * filter,int byteNum){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

    for(long i=offset; i<numInt; i += stride){
        int tmp = ((int *)fact)[i];
        unsigned long bit = 1;

        for(int j=0; j< sizeof(int)/byteNum; j++){
            int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
            int fkey = (tmp & t)>> (j*byteNum*8) ;
            filter[i* sizeof(int)/byteNum + j] |= dictFilter[fkey];
        }
    }
}

/*
 * genScanFilter_dict_init: generate the filter for dictionary-compressed predicate
 */

__global__ static void genScanFilter_dict_init(struct dictHeader *dheader, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(int i=tid;i<dNum;i+=stride){
        int fkey = dheader->hash[i];
        con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
        dfilter[i] = con;
    }
}

__global__ static void genScanFilter_dict_or(struct dictHeader *dheader, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(int i=tid;i<dNum;i+=stride){
        int fkey = dheader->hash[i];
        con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
        dfilter[i] |= con;
    }
}

__global__ static void genScanFilter_dict_and(struct dictHeader *dheader, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(int i=tid;i<dNum;i+=stride){
        int fkey = dheader->hash[i];
        con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
        dfilter[i] &= con;
    }
}

__global__ static void genScanFilter_rle(char *col, int colSize, int colType, long tupleNum, struct whereExp *where, int andOr, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    struct rleHeader *rheader = (struct rleHeader *) col;
    int dNum = rheader->dictNum;

    for(int i = tid; i<dNum; i += stride){
        int fkey = ((int *)(col+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(col+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(col+sizeof(struct rleHeader)))[i + 2*dNum];

        con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
    
        for(int k=0;k<fcount;k++){
            if(andOr == AND)
                filter[fpos+k] &= con;
            else
                filter[fpos+k] |= con;
        }

    }
}

__global__ static void genScanFilter_and_eq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) == 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_gth(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) > 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_lth(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) < 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_geq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) >= 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_leq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) <= 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_init_int_eq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] == where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_float_eq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] == where; 
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_int_gth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] > where; 
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_float_gth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] > where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_int_lth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] < where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_float_lth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] < where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_int_geq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] >= where;
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_float_geq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] >= where;
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_int_leq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] <= where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_float_leq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] <= where; 
        filter[i] = con;
    }
}

__global__ static void genScanFilter_and_int_eq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] == where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_eq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] == where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_int_geq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] >= where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_geq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] >= where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_int_leq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] <= where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_leq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] <= where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_int_gth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] > where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_gth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] > where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_int_lth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] < where;
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_lth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] < where;
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_init_eq(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) == 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_gth(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) > 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_lth(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) < 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_geq(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) >= 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_leq(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) <= 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_or_eq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize) == 0);
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_gth(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize)> 0);
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_lth(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize) < 0);
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_geq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize) >= 0);
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_leq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize) <= 0);
        filter[i] |= con;
    }
}


__global__ static void genScanFilter_or_int_eq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] == where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_float_eq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] == where; 
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_int_gth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] > where; 
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_float_gth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] > where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_int_lth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] < where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_float_lth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] < where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_int_geq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] >= where;
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_float_geq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] >= where;
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_int_leq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] <= where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_float_leq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] <= where; 
        filter[i] |= con;
    }
}

/*
 * countScanNum: count the number of results that each thread generates.
 */

__global__ static void countScanNum(int *filter, long tupleNum, int * count){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localCount = 0;

    for(long i = tid; i<tupleNum; i += stride){
        localCount += filter[i];
    }

    count[tid] = localCount;

}

/*
 * scan_dict_other: generate the result for dictionary-compressed column.
 */

__global__ static void scan_dict_other(char *col, struct dictHeader * dheader, int byteNum,int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = psum[tid] * colSize;

    for(long i = tid; i<tupleNum; i+= stride){
        if(filter[i] == 1){
            int key = 0;
            memcpy(&key, col + sizeof(struct dictHeader) + i* dheader->bitNum/8, dheader->bitNum/8);
            memcpy(result+pos,&dheader->hash[key],colSize);
            pos += colSize;
        }
    }
}

__global__ static void scan_dict_int(char *col, struct dictHeader * dheader,int byteNum,int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localCount = psum[tid]; 

    for(long i = tid; i<tupleNum; i+= stride){
        if(filter[i] == 1){
            int key = 0;
            memcpy(&key, col + sizeof(struct dictHeader) + i*byteNum, byteNum);
            ((int *)result)[localCount] = dheader->hash[key];
            localCount ++;
        }
    }
}

/*
 * scan_other: generate scan result for uncompressed column.
 */

__global__ static void scan_other(char *col, int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = psum[tid]  * colSize;

    for(long i = tid; i<tupleNum;i+=stride){
        
        if(filter[i] == 1){
            memcpy(result+pos,col+i*colSize,colSize);
            pos += colSize;
        }
    }
}


__global__ static void scan_int(char *col, int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localCount = psum[tid] ; 

    for(long i = tid; i<tupleNum;i+=stride){
        
        if(filter[i] == 1){
            ((int*)result)[localCount] = ((int*)col)[i];
            localCount ++;
        }
    }
}

__global__ void static unpack_rle(char * fact, char * rle, long tupleNum, int dNum){

    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<dNum; i+=stride){

        int fvalue = ((int *)(fact+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

        for(int k=0;k<fcount;k++){
            ((int*)rle)[fpos+ k] = fvalue;
        }
    }
}

/*
 * tableScan Prerequisites:
 *  1. the input data can be fit into GPU device memory
 *  2. input data are stored in host memory
 * 
 * Input:
 *  sn: contains the data to be scanned and the predicate information
 *  pp: records statistics such kernel execution time and PCIe transfer time 
 *
 * Output:
 *  A new table node
 */

struct tableNode * tableScan(struct scanNode *sn, struct statistic *pp){

    struct timespec start,end;
    clock_gettime(CLOCK_REALTIME,&start);

    struct tableNode *res = NULL;
    int tupleSize = 0;

    res = (struct tableNode *) malloc(sizeof(struct tableNode));
    CHECK_POINTER(res);

    res->totalAttr = sn->outputNum;

    res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrType);
    res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrSize);
    res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrTotalSize);
    res->attrIndex = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->attrIndex);
    res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->dataPos);
    res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
    CHECK_POINTER(res->dataFormat);
    res->content = (char **) malloc(sizeof(char *) * res->totalAttr);
    CHECK_POINTER(res->content);

    for(int i=0;i<res->totalAttr;i++){
        int index = sn->outputIndex[i];
        res->attrType[i] = sn->tn->attrType[index];
        res->attrSize[i] = sn->tn->attrSize[index];
    }

    int *gpuCount = NULL, *gpuFilter = NULL, *gpuPsum = NULL;

    dim3 grid(2048);
    dim3 block(256);

    long totalTupleNum = sn->tn->tupleNum;
    int blockNum = totalTupleNum / block.x + 1;

    if(blockNum<2048)
        grid = blockNum;

    int threadNum = grid.x * block.x;
    int attrNum = sn->whereAttrNum;

    char ** column = (char **) malloc(attrNum * sizeof(char *));
    CHECK_POINTER(column);

    int * whereFree = (int *)malloc(attrNum * sizeof(int));
    CHECK_POINTER(whereFree);

    int * colWherePos = (int *)malloc(sn->outputNum * sizeof(int));
    CHECK_POINTER(colWherePos);

    for(int i=0;i<sn->outputNum;i++)
        colWherePos[i] = -1;

    for(int i=0;i<attrNum;i++){
        whereFree[i] = 1;
        for(int j=0;j<sn->outputNum;j++){
            if(sn->whereIndex[i] == sn->outputIndex[j]){
                whereFree[i] = -1;
                colWherePos[j] = i;
            }
        }
    }

    int count;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFilter,sizeof(int) * totalTupleNum));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuPsum,sizeof(int)*threadNum));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuCount,sizeof(int)*threadNum));

    assert(sn->hasWhere !=0);
    assert(sn->filter != NULL);

    struct whereCondition *where = sn->filter;

/*
 * The first step is to evaluate the selection predicates and generate a vetor to form the final results.
 */

    if(1){

        struct whereExp * gpuExp = NULL;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuExp, sizeof(struct whereExp)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[0], sizeof(struct whereExp), cudaMemcpyHostToDevice));

/*
 * Currently we evaluate the predicate one by one.
 * When consecutive predicates are accessing the same column, we don't release the GPU device memory
 * that store the accessed column until all these predicates have been evaluated. Otherwise the device
 * memory will be released immediately after the corresponding predicate has been evaluated.
 *
 * (@whereIndex, @prevWhere), (@index,  @prevIndex), (@format, @prevFormat) are used to decide
 * whether two consecutive predicates access the same column with the same format.
 */

        int whereIndex = where->exp[0].index;
        int index = sn->whereIndex[whereIndex];
        int prevWhere = whereIndex;
        int prevIndex = index;

        int format = sn->tn->dataFormat[index];
        int prevFormat = format;

/*
 * @dNum, @byteNum and @gpuDictFilter are for predicates that need to access dictionary-compressed columns.
 */
        int dNum;
        int byteNum;
        int *gpuDictFilter = NULL;

/*
 * We will allocate GPU device memory for a column if it is stored in the host pageable or pinned memory.
 * If it is configured to utilize the UVA technique, no GPU device memory will be allocated.
 */

        if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[whereIndex], sn->tn->attrTotalSize[index]));

        if(format == UNCOMPRESSED){
            if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
            else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                column[whereIndex] = sn->tn->content[index];

            int rel = where->exp[0].relation;
            if(sn->tn->attrType[index] == INT){
                int whereValue = *((int*) where->exp[0].content);
                if(rel==EQ)
                    genScanFilter_init_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == GTH)
                    genScanFilter_init_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == LTH)
                    genScanFilter_init_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == GEQ)
                    genScanFilter_init_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if (rel == LEQ)
                    genScanFilter_init_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);

            }else if (sn->tn->attrType[index] == FLOAT){
                float whereValue = *((float*) where->exp[0].content);
                if(rel==EQ)
                    genScanFilter_init_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == GTH)
                    genScanFilter_init_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == LTH)
                    genScanFilter_init_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == GEQ)
                    genScanFilter_init_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if (rel == LEQ)
                    genScanFilter_init_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);

            }else{
                if(rel == EQ)
                    genScanFilter_init_eq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == GTH)
                    genScanFilter_init_gth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == LTH)
                    genScanFilter_init_lth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == GEQ)
                    genScanFilter_init_geq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == LEQ)
                    genScanFilter_init_leq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
            }

        }else if(format == DICT){

/*Data are stored in the host memory for selection*/

            struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
            dNum = dheader->dictNum;
            byteNum = dheader->bitNum/8;

            struct dictHeader* gpuDictHeader = NULL;
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

            if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
            else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                column[whereIndex] = sn->tn->content[index];

            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

            genScanFilter_dict_init<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);

            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

        }else if(format == RLE){

            if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
            else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                column[whereIndex] = sn->tn->content[index];

            genScanFilter_rle<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, where->andOr, gpuFilter);
        }

        int dictFilter = 0;
        int dictFinal = OR;
        int dictInit = 1;

        for(int i=1;i<where->expNum;i++){
            whereIndex = where->exp[i].index;
            index = sn->whereIndex[whereIndex];
            format = sn->tn->dataFormat[index];

            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[i], sizeof(struct whereExp), cudaMemcpyHostToDevice));

            if(prevIndex != index){

/*When the two consecutive predicates access different columns*/

                if(prevFormat == DICT){
                    if(dictInit == 1){
                        transform_dict_filter_init<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);
                        dictInit = 0;
                    }else if(dictFinal == OR)
                        transform_dict_filter_or<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);
                    else
                        transform_dict_filter_and<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);

                    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
                    dictFinal = where->andOr;
                }

                if(whereFree[prevWhere] == 1 && (sn->tn->dataPos[prevIndex] == MEM || sn->tn->dataPos[prevIndex] == MMAP || sn->tn->dataPos[prevIndex] == PINNED))
                    CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[prevWhere]));

                if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[whereIndex] , sn->tn->attrTotalSize[index]));

                if(format == DICT){
                    if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
                    else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                        column[whereIndex] = sn->tn->content[index];

                    struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
                    dNum = dheader->dictNum;
                    byteNum = dheader->bitNum/8;

                    struct dictHeader * gpuDictHeader = NULL;
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

                    genScanFilter_dict_init<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
                    dictFilter= -1;
                    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

                }else{
                    if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
                    else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                        column[whereIndex] = sn->tn->content[index];
                }

                prevIndex = index;
                prevWhere = whereIndex;
                prevFormat = format;
            }


            if(format == UNCOMPRESSED){
                int rel = where->exp[i].relation;
                if(sn->tn->attrType[index] == INT){
                    int whereValue = *((int*) where->exp[i].content);
                    if(where->andOr == AND){
                        if(rel==EQ)
                            genScanFilter_and_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GTH)
                            genScanFilter_and_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == LTH)
                            genScanFilter_and_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GEQ)
                            genScanFilter_and_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_and_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                    }else{
                        if(rel==EQ)
                            genScanFilter_or_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GTH)
                            genScanFilter_or_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == LTH)
                            genScanFilter_or_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GEQ)
                            genScanFilter_or_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_or_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                    }

                } else if (sn->tn->attrType[index] == FLOAT){
                    float whereValue = *((float*) where->exp[i].content);
                    if(where->andOr == AND){
                        if(rel==EQ)
                            genScanFilter_and_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GTH)
                            genScanFilter_and_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == LTH)
                            genScanFilter_and_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GEQ)
                            genScanFilter_and_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_and_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                    }else{
                        if(rel==EQ)
                            genScanFilter_or_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GTH)
                            genScanFilter_or_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == LTH)
                            genScanFilter_or_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GEQ)
                            genScanFilter_or_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_or_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                    }
                }else{
                    if(where->andOr == AND){
                        if (rel == EQ)
                            genScanFilter_and_eq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GTH)
                            genScanFilter_and_gth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LTH)
                            genScanFilter_and_lth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GEQ)
                            genScanFilter_and_geq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_and_leq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                    }else{
                        if (rel == EQ)
                            genScanFilter_or_eq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GTH)
                            genScanFilter_or_gth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LTH)
                            genScanFilter_or_lth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GEQ)
                            genScanFilter_or_geq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_or_leq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                    }
                }

            }else if(format == DICT){

                struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
                dNum = dheader->dictNum;
                byteNum = dheader->bitNum/8;

                struct dictHeader * gpuDictHeader;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

                if(dictFilter != -1){
                    if(where->andOr == AND)
                        genScanFilter_dict_and<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
                    else
                        genScanFilter_dict_or<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
                }
                dictFilter = 0;

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

            }else if (format == RLE){
                genScanFilter_rle<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, where->andOr, gpuFilter);

            }

        }

        if(prevFormat == DICT){
            if (dictInit == 1){
                transform_dict_filter_init<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
                dictInit = 0;
            }else if(dictFinal == AND)
                transform_dict_filter_and<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
            else
                transform_dict_filter_or<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
        }
    
        if(whereFree[prevWhere] == 1 && (sn->tn->dataPos[prevIndex] == MEM || sn->tn->dataPos[prevIndex] == MMAP || sn->tn->dataPos[prevIndex] == PINNED))
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[prevWhere]));

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuExp));

    }


/* Count the number of tuples that meets the predicats for each thread
 * and calculate the prefix sum.
 */
    countScanNum<<<grid,block>>>(gpuFilter,totalTupleNum,gpuCount);
    scanImpl(gpuCount,threadNum,gpuPsum, pp);

    int tmp1, tmp2;

    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp1, &gpuCount[threadNum-1], sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp2, &gpuPsum[threadNum-1], sizeof(int), cudaMemcpyDeviceToHost));

    count = tmp1+tmp2;
    res->tupleNum = count;
    printf("[INFO]Number of selection results: %d\n",count);

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuCount));

    char **result = NULL, **scanCol = NULL;

    attrNum = sn->outputNum;

    scanCol = (char **) malloc(attrNum * sizeof(char *));
    CHECK_POINTER(scanCol);
    result = (char **) malloc(attrNum * sizeof(char *));
    CHECK_POINTER(result);
    
    for(int i=0;i<attrNum;i++){

        int pos = colWherePos[i];
        int index = sn->outputIndex[i];
        tupleSize += sn->tn->attrSize[index];

        if(pos != -1){
            scanCol[i] = column[pos];
        }else{
            if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &scanCol[i] , sn->tn->attrTotalSize[index]));

            if(sn->tn->dataFormat[index] != DICT){
                if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(scanCol[i], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
                else
                    scanCol[i] = sn->tn->content[index];

            }else{
                if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(scanCol[i], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
                else
                    scanCol[i] = sn->tn->content[index];
            }
        }

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &result[i], count * sn->tn->attrSize[index]));
    }

    if(1){

        for(int i=0; i<attrNum; i++){
            int index = sn->outputIndex[i];
            int format = sn->tn->dataFormat[index];
            if(format == UNCOMPRESSED){
                if (sn->tn->attrSize[index] == sizeof(int))
                    scan_int<<<grid,block>>>(scanCol[i], sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);
                else
                    scan_other<<<grid,block>>>(scanCol[i], sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

            }else if(format == DICT){
                struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
                int byteNum = dheader->bitNum/8;

                struct dictHeader * gpuDictHeader = NULL;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

                if (sn->tn->attrSize[i] == sizeof(int))
                    scan_dict_int<<<grid,block>>>(scanCol[i], gpuDictHeader, byteNum,sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);
                else
                    scan_dict_other<<<grid,block>>>(scanCol[i], gpuDictHeader,byteNum,sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

            }else if(format == RLE){
                int dNum = (sn->tn->attrTotalSize[index] - sizeof(struct rleHeader))/(3*sizeof(int));
                char * gpuRle = NULL;

                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRle, totalTupleNum * sizeof(int)));

                unpack_rle<<<grid,block>>>(scanCol[i], gpuRle,totalTupleNum, dNum);


                scan_int<<<grid,block>>>(gpuRle, sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));
            }

        }
    }


    res->tupleSize = tupleSize;

    for(int i=0;i<attrNum;i++){

        int index = sn->outputIndex[i];

        if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(scanCol[i]));

        int colSize = res->tupleNum * res->attrSize[i];

        res->attrTotalSize[i] = colSize;
        res->dataFormat[i] = UNCOMPRESSED;

        if(sn->keepInGpu == 1){
            res->dataPos[i] = GPU;
            res->content[i] = result[i];
        }else{
            res->dataPos[i] = MEM;
            res->content[i] = (char *)malloc(colSize);
            CHECK_POINTER(res->content[i]);
            memset(res->content[i],0,colSize);
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i],result[i],colSize ,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaFree(result[i]));
        }
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuPsum));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuFilter));

    free(column);
    free(scanCol);
    free(result);

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    printf("TableScan Time: %lf\n", timeE/(1000*1000));

    return res;

}


