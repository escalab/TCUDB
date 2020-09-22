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
#ifdef DEBUG
#include "../include/cuPrintf.cu"
#include "../include/cuPrintf.cuh"
#endif

using namespace nvcuda;

// For wmma API, these must be multiples fo 16
//#define MATRIX_M 16
//#define MATRIX_N 16
//#define MATRIX_K 16

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
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

#ifdef ITUNES
    struct timespec itunesFill_start, itunesFill_end;
#elif BEER
    struct timespec beerFill_start, beerFill_end;
#endif

#ifdef BEER
__host__ void static attrProjection(float * res, int height, int width, 
        struct joinNode *jNode, 
        short *A_id, short *B_id,
        short *A_beer, short *B_beer,
        short *A_factory, short *B_factory,
        short *A_style, short *B_style,
        short *A_abv, short *B_abv) {
    /* jNode->leftOutputIndex[index] 
     * index = 0 id
     *         1 beer_name
     *         2 factory
     *         3 style
     */
    int i, m, n; // m: A tuple# | n: B tuple#
    for (i = 0; i < height*width; i++) {
        // project non-zero elements
        if (res[i] != 0) {
            m = i / width; // A_tuple#
            n = i % width; // B_tuple#

            // project left table attribute
            for (int o = 0; o < jNode->leftOutputAttrNum; o++) {
                // get attribute index
                int leftAttrIndex = jNode->leftTable->attrIndex[o];
                switch(leftAttrIndex) {
                    case 0:
                        printf("Table A ID: %hu\n", A_id[m]);
                        break;
                    case 1:
                        printf("Table A Beer Name: %hu\n", A_beer[m]);
                        break;
                    case 2:
                        printf("Table A Factory: %hu\n", A_factory[m]);
                        break;
                    case 3:
                        printf("Table A Style: %hu\n", A_style[m]);
                        break;
                    case 4:
                        printf("Table A ABV: %hu\n", A_abv[m]);
                        break;
                    default:
                        printf("Invalid attribute index for Table A\n");
                }
            }

            // project right table attribute
            for (int o = 0; o < jNode->rightOutputAttrNum; o++) {
                // get attribute index
                int rightAttrIndex = jNode->rightTable->attrIndex[o];
                switch(rightAttrIndex) {
                    case 0:
                        printf("Table B ID: %hu\n", B_id[n]);
                        break;
                    case 1:
                        printf("Table B Beer Name: %hu\n", B_beer[n]);
                        break;
                    case 2:
                        printf("Table B Factory: %hu\n", B_factory[n]);
                        break;
                    case 3:
                        printf("Table B Style: %hu\n", B_style[n]);
                        break;
                    case 4:
                        printf("Table B ABV: %hu\n", B_abv[n]);
                        break;
                    default:
                        printf("Invalid attribute index for Table B\n");
                }
            }

        }
        printf("\n");
    }
    //TODO: create materialized view -- combined projected column into a temp table
    // e.g. beer2.sql -> 7x6 table, 7 rows(join count), 6 columns(projection)
    //TODO: further matching 

}
#elif ITUNES
__host__ void static attrProjection(float * res, int height, int width, 
        struct joinNode *jNode, 
        short *A_id, short *B_id,
        short *A_song, short *B_song,
        short *A_artist, short *B_artist,
        short *A_album, short *B_album,
        short *A_genre, short *B_genre,
        short *A_price, short *B_price,
        short *A_copyright, short *B_copyright,
        short *A_time, short *B_time,
        short *A_released, short *B_released) {

    printf("height: %d\twidth: %d\n",height, width);
    int i, m, n; // m: A tuple# | n: B tuple#
    for (i = 0; i < height*width; i++) {
        // project non-zero elements
        if (res[i] != 0) {
            m = i / width; // A_tuple#
            n = i % width; // B_tuple#

            // project left table attribute
            for (int o = 0; o < jNode->leftOutputAttrNum; o++) {
                // get attribute index
                int leftAttrIndex = jNode->leftTable->attrIndex[o];
                printf("leftAttrIndex: %d\n",leftAttrIndex);
                switch(leftAttrIndex) {
                    case 0:
                        printf("Table A ID: %hu\n", A_id[m]);
                        break;
                    case 1:
                        printf("Table A Song Name: %hu\n", A_song[m]);
                        break;
                    case 2:
                        printf("Table A Artist: %hu\n", A_artist[m]);
                        break;
                    case 3:
                        printf("Table A Album: %hu\n", A_album[m]);
                        break;
                    case 4:
                        printf("Table A Genre: %hu\n", A_genre[m]);
                        break;
                    case 5:
                        printf("Table A Price: %hu\n", A_price[m]);
                        break;
                    case 6:
                        printf("Table A CopyRight: %hu\n", A_copyright[m]);
                        break;
                    case 7:
                        printf("Table A Time: %hu\n", A_time[m]);
                        break;
                    case 8:
                        printf("Table A Released: %hu\n", A_released[m]);
                        break;
                    default:
                        printf("Invalid attribute index for Table A\n");
                }
            }

            // project right table attribute
            for (int o = 0; o < jNode->rightOutputAttrNum; o++) {
                // get attribute index
                int rightAttrIndex = jNode->rightTable->attrIndex[o];
                //printf("rightOutputIndex: %d\n", rightAttrIndex);
                switch(rightAttrIndex) {
                    case 0:
                        printf("Table B ID: %hu\n", B_id[n]);
                        break;
                    case 1:
                        printf("Table B Song Name: %hu\n", B_song[n]);
                        break;
                    case 2:
                        printf("Table B Artist: %hu\n", B_artist[n]);
                        break;
                    case 3:
                        printf("Table B Album: %hu\n", B_album[n]);
                        break;
                    case 4:
                        printf("Table B Genre: %hu\n", B_genre[n]);
                        break;
                    case 5:
                        printf("Table B Price: %hu\n", B_price[n]);
                        break;
                    case 6:
                        printf("Table B CopyRight: %hu\n", B_copyright[n]);
                        break;
                    case 7:
                        printf("Table B Time: %hu\n", B_time[n]);
                        break;
                    case 8:
                        printf("Table B Released: %hu\n", B_released[n]);
                        break;
                    default:
                        printf("Invalid attribute index for Table B\n");
                }
            }

        }
        printf("\n");
    }
}

#endif

/* Fill matrix on device memory */
// TODO: if attrType == 1, ptr cast to char type
__global__ void static gpu_fill(char *column, int matWidth, half *mat, size_t tupleNum, int attrType) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i > tupleNum) return;

    int index = i * attrType;
    //int value = (int)column[index]; // char -> int will lose 3 bytes
    int *value = (int*)&column[index];
    //mat[i*matWidth + (*value)] = __int2half_rd(1);
    mat[i*matWidth + (*value)] = __float2half(1.0f);
}

__host__ void static print_vector(float *vec, int n) {
    for(int i = 0; i < n; i++)
        printf("%.1f ", vec[i]);
}

__host__ void static verify_test(half *matrix, int height, int width) {
    int i;
    for (i = 0; i < height*width; i++) {
        printf("%.0f\t", __half2float(matrix[i]));
        if ((i+1) % width == 0)
            printf("\n\n");  
    }

}

__host__ void static verify_result(float * matrix, int height, int width) {
    int i;
    for (i = 0; i < height*width; i++) {
        printf("%.0f\t", matrix[i]);
        if ((i+1) % width == 0)
            printf("\n\n");  
    }

}

__host__ void static verify_result_short(short * matrix, int height, int width){
    int i;
    for (i = 0; i < height*width; i++) {
        printf("%hu\t", matrix[i]);
        if ((i+1) % width == 0)
            printf("\n\n");  
    }

}

__host__ void static verify_result_int(int * matrix, int height, int width) {
    int i;
    for (i = 0; i < height*width; i++) {
        printf("%d\t", matrix[i]);
        if ((i+1) % width == 0)
            printf("\n\n");  
    }

}

#ifdef CUBLAS_HALF
__global__ void gpu_transpose(half *odata, const half *idata, int row, int col) {
//__global__ void gpu_transpose(half *odata, const short *idata, int row, int col) {
//__global__ void gpu_transpose(short *odata, const short *idata, int row, int col) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index % col;
    int y = index / col;
    //int x = blockDim.x * blockIdx.x + threadIdx.x;
    //int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < col && y < row) {
        odata[x*row + y] = idata[y*col + x];
        //odata[x*row + y] = __short2half_rd(idata[y*col + x]);
        //odata[x*row + y] = idata[y*col + x];
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

/* Transpose the matrix on CPU */
#ifdef CUBLAS_HALF
__host__ void transpose(short *in, short *out, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            out[j*row+i] = in[i*col+j];
        }
    }
}
#else
__host__ void transpose(float *in, float *out, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            out[j*row+i] = in[i*col+j];
        }
    }
}
#endif

#ifdef WMMA_INT4
__host__ int sum_matrix_int(int *mat, int height, int width) {
    int i, sum = 0;
    for (i = 0; i < height*width; i++)
        sum += mat[i];
    return sum;
}
#else
__host__ int sum_matrix(float *mat, int height, int width) {
    int i, sum = 0;
    for (i = 0; i < height*width; i++)
        sum += mat[i];
    return sum;
}
#endif
/* Find the nearest multiple of N, check the width of matrix or tupleNum to form the matrices for MM */
__host__ int nearestMultipleN(int inNum, int n) {
    if (!n)
        return inNum;
    int remain = inNum % n;
    if (!remain)
        return inNum;
    return (inNum + n - remain);
}

/*
 *  If the query only need to return the count of join result.
 *  result t = mat1*mat.T
 *  count = t.size - sum(t) -- how many non-zero in t
 */
#ifdef WMMA_INT4
__host__ void static tcu_match(struct joinNode *jNode, int width,
         signed char *A, signed char *B, int attr_type1, int attr_type2) {

    int A_tupleNum = jNode->leftTable->tupleNum;
    int B_tupleNum = jNode->rightTable->tupleNum;

    // create first matrix
    int i, colContIdx; // index of column content
    colContIdx = 0;
    for (i = 0; i < A_tupleNum; i++) {
        int *colCont;   // content of column
        colCont = (int*)(&jNode->leftTable->content[jNode->leftKeyIndex][colContIdx]);
        colContIdx += attr_type1; // 4 because of INT type
        A[i*width+(*colCont)] = 1; // mark as 1 if appear in the matrix
    }

    // create second matrix
    colContIdx = 0;
    for (i = 0; i < B_tupleNum; i++) {
        int *colCont;
        colCont = (int*)(&jNode->rightTable->content[jNode->rightKeyIndex][colContIdx]);
        colContIdx += attr_type1;
        B[i*width+(*colCont)] = 1;
    }

    // transpose second matrix
    //transpose(B, B_T, B_tupleNum, width);

    // perform MM & return count on device
}
#elif CUBLAS_HALF
// host version
__host__ void static tcu_match(struct joinNode *jNode, int width,
         short *A, short *B, int attr_type1, int attr_type2) {

    int A_tupleNum = jNode->leftTable->tupleNum;
    int B_tupleNum = jNode->rightTable->tupleNum;

    // create first matrix
    int i, colContIdx; // index of column content
    colContIdx = 0;
    for (i = 0; i < A_tupleNum; i++) {
        int *colCont;   // content of column
        colCont = (int*)(&jNode->leftTable->content[jNode->leftKeyIndex][colContIdx]);
        colContIdx += attr_type1; // 4 because of INT type
        A[i*width+(*colCont)] = (short)1; // mark as 1 if appear in the matrix
    }

    // create second matrix
    colContIdx = 0;
    for (i = 0; i < B_tupleNum; i++) {
        int *colCont;
        colCont = (int*)(&jNode->rightTable->content[jNode->rightKeyIndex][colContIdx]);
        colContIdx += attr_type1;
        B[i*width+(*colCont)] = (short)1;
    }

    // transpose second matrix
    //transpose(B, B_T, B_tupleNum, width);

    // perform MM & return count on device
}
#elif CUBLAS || WMMA_HALF 
__host__ void static tcu_match(struct joinNode *jNode, int width,
         float *A, float *B, int attr_type1, int attr_type2) {

    int A_tupleNum = jNode->leftTable->tupleNum;
    int B_tupleNum = jNode->rightTable->tupleNum;

    // create first matrix
    int i, colContIdx; // index of column content
    colContIdx = 0;
    for (i = 0; i < A_tupleNum; i++) {
        int *colCont;   // content of column
        colCont = (int*)(&jNode->leftTable->content[jNode->leftKeyIndex][colContIdx]);
        colContIdx += attr_type1; // 4 because of INT type
        A[i*width+(*colCont)] = 1; // mark as 1 if appear in the matrix
    }

    // create second matrix
    colContIdx = 0;
    for (i = 0; i < B_tupleNum; i++) {
        int *colCont;
        colCont = (int*)(&jNode->rightTable->content[jNode->rightKeyIndex][colContIdx]);
        colContIdx += attr_type1;
        B[i*width+(*colCont)] = 1;
    }

    // transpose second matrix
    //transpose(B, B_T, B_tupleNum, width);

    // perform MM & return count on device
}
#endif

#ifdef ITUNES
/* iTunes dataset with iTunes.sql */
//TODO: if materialized view is required
__host__ void static itunes_match(struct joinNode *jNode, int width,
         short *A, short *B, int attr_type1, int attr_type2, 
         int attr_num1, int attr_num2,
         short *A_id, short *B_id,
         short *A_song, short *B_song,
         short *A_artist, short *B_artist,
         short *A_album, short *B_album,
         short *A_genre, short *B_genre,
         short *A_price, short *B_price,
         short *A_copyright, short *B_copyright,
         short *A_time, short *B_time,
         short *A_released, short *B_released
         ) {

    int A_tupleNum = jNode->leftTable->tupleNum;
    int B_tupleNum = jNode->rightTable->tupleNum;

    //printf("attr_num1: %d\n", attr_num1);
    //printf("attr_num2: %d\n", attr_num2);

    int m, n;
    for (m = 0; m < attr_num1; m++) {
        int A_col_idx = jNode->leftTable->attrIndex[m];
        int k = 0;

        for (n = 0; n < A_tupleNum*attr_type1; n+=attr_type1) {
            int *temp;
            temp = (int*)(&jNode->leftTable->content[m][n]);

            switch(A_col_idx) {
                case 0:
                    //printf("A id: %d\n", *temp);
                    A_id[k] = (short)*temp;
                    break;
                case 1:
                    //printf("A song: %d\n", *temp);
                    A_song[k] = (short)*temp;
                    break;
                case 2:
                    A_artist[k] = (short)*temp;
                    break;
                case 3:
                    A_album[k] = (short)*temp;
                    break;
                case 4:
                    A_genre[k] = (short)*temp;
                    break;
                case 5:
                    A_price[k] = (short)*temp;
                    break;
                case 6:
                    A_copyright[k] = (short)*temp;
                    break;
                case 7:
                    //printf("A time: %d\n", *temp);
                    A_time[k] = (short)*temp;
                    break;
                case 8:
                    A_released[k] = (short)*temp;
                    break;
                default:
                    printf("Invalid attribute index for Table A\n");
            } 

            k++;
        }
    }

    for (m = 0; m < attr_num2; m++) {
        int B_col_idx = jNode->rightTable->attrIndex[m];
        int k = 0;

        for (n = 0; n < B_tupleNum*attr_type2; n+=attr_type2) {
            int *temp;
            temp = (int*)(&jNode->rightTable->content[m][n]);
            
            switch(B_col_idx) {
                case 0:
                    B_id[k] = (short)*temp;
                    break;
                case 1:
                    B_song[k] = (short)*temp;
                    break;
                case 2:
                    B_artist[k] = (short)*temp;
                    break;
                case 3:
                    B_album[k] = (short)*temp;
                    break;
                case 4:
                    B_genre[k] = (short)*temp;
                    break;
                case 5:
                    B_price[k] = (short)*temp;
                    break;
                case 6:
                    B_copyright[k] = (short)*temp;
                    break;
                case 7:
                    B_time[k] = (short)*temp;
                    break;
                case 8:
                    B_released[k] = (short)*temp;
                    break;
                default:
                    printf("Invalid attribute index for Table B\n");
            }

            k++;
        }
    }
   
    clock_gettime(CLOCK_REALTIME, &itunesFill_start);
    // create first matrix
    int i, colContIdx; // index of column content
    colContIdx = 0;
    for (i = 0; i < A_tupleNum; i++) {
        int *colCont;   // content of column
        colCont = (int*)(&jNode->leftTable->content[jNode->leftKeyIndex][colContIdx]);
        colContIdx += attr_type1; // 4 because of INT type
        A[i*width+(*colCont)] = (short)1; // mark as 1 if appear in the matrix
    }

    // create second matrix
    colContIdx = 0;
    for (i = 0; i < B_tupleNum; i++) {
        int *colCont;
        colCont = (int*)(&jNode->rightTable->content[jNode->rightKeyIndex][colContIdx]);
        colContIdx += attr_type1;
        B[i*width+(*colCont)] = (short)1;
    } 
    clock_gettime(CLOCK_REALTIME, &itunesFill_end);
}
#endif

#ifdef BEER
//TODO: if materialized view is required
/* correspond to beer.sql 
 * SELECT TABLEA.BEER, TABLEA.ABV, TABLEB.STYLE
 * FROM TABLEA, TABLEB
 * WHERE TABLEA.ABV = TABLEB.ABV;
 */
// #attributes to store depends on SELECT statements
// iterate sequence depends on SELECT sequence; however,
// col_idx sequence is fixed, depends on schema
__host__ void static beer_match(struct joinNode *jNode, int width,
         short *A, short *B, int attr_type1, int attr_type2, 
         int attr_num1, int attr_num2,
         short *A_id, short *B_id,
         short *A_beer, short *B_beer,
         short *A_factory, short *B_factory,
         short *A_style, short *B_style,
         short *A_abv, short *B_abv) {

    int A_tupleNum = jNode->leftTable->tupleNum;
    int B_tupleNum = jNode->rightTable->tupleNum;
    //for (int o = 0; o < jNode->leftOutputAttrNum; o++)
    //    printf("left output index: %d\n", jNode->leftOutputIndex[o]);
    //for (int o = 0; o < jNode->rightOutputAttrNum; o++)
    //    printf("right output index: %d\n", jNode->rightOutputIndex[o]);
    //printf("attr_num1: %d\n", attr_num1);
    //printf("attr_num2: %d\n", attr_num2);
    int m, n;
    for (m = 0; m < attr_num1; m++) {
        int A_col_idx = jNode->leftTable->attrIndex[m];
        int k = 0;

        for (n = 0; n < A_tupleNum*attr_type1; n+=attr_type1) {
            int *temp;
            temp = (int*)(&jNode->leftTable->content[m][n]);

            switch(A_col_idx) { 
                case 0:
                    A_id[k] = (short)*temp;
                    //printf("A col_idx[0]: %d\n", *temp);
                    break;
                case 1:
                    A_beer[k] = (short)*temp;
                    //printf("A col_idx[1]: %d\n", *temp);
                    break;
                case 2:
                    A_factory[k] = (short)*temp;
                    //printf("A col_idx[2]: %d\n", *temp);
                    break;
                case 3:
                    A_style[k] = (short)*temp;
                    //printf("A col_idx[3]: %d\n", *temp);
                    break;
                case 4:
                    A_abv[k] = (short)*temp;
                    //printf("A col_idx[4]: %d\n", *temp);
                    break;
                default:
                    printf("Invalid attribute index for Table A\n");
            }

            k++;
        }
    }

    for (m = 0; m < attr_num2; m++) {
        int B_col_idx = jNode->rightTable->attrIndex[m];
        int k = 0;

        for (n = 0; n < B_tupleNum*attr_type2; n+=attr_type2) {
            int *temp;
            temp = (int*)(&jNode->rightTable->content[m][n]);

            switch(B_col_idx) { 
                case 0:
                    B_id[k] = (short)*temp;
                    //printf("B col_idx[0]: %d\n", *temp);
                    break;
                case 1:
                    B_beer[k] = (short)*temp;
                    //printf("B col_idx[1]: %d\n", *temp);
                    break;
                case 2:
                    B_factory[k] = (short)*temp;
                    //printf("B col_idx[2]: %d\n", *temp);
                    break;
                case 3:
                    B_style[k] = (short)*temp;
                    //printf("B col_idx[3]: %d\n", *temp);
                    break;
                case 4:
                    B_abv[k] = (short)*temp;
                    //printf("B col_idx[4]: %d\n", *temp);
                    break;
                default:
                    printf("Invalid attribute index for Table B\n");
            }

            k++;
        }
    }

    clock_gettime(CLOCK_REALTIME, &beerFill_start);
    // create first matrix
    int i, colContIdx; // index of column content
    colContIdx = 0;
    for (i = 0; i < A_tupleNum; i++) {
        int *colCont;   // content of column
        colCont = (int*)(&jNode->leftTable->content[jNode->leftKeyIndex][colContIdx]);
        colContIdx += attr_type1; // 4 because of INT type
        A[i*width+(*colCont)] = (short)1; // mark as 1 if appear in the matrix
    }

    // create second matrix
    colContIdx = 0;
    for (i = 0; i < B_tupleNum; i++) {
        int *colCont;
        colCont = (int*)(&jNode->rightTable->content[jNode->rightKeyIndex][colContIdx]);
        colContIdx += attr_type1;
        B[i*width+(*colCont)] = (short)1;
    }
    clock_gettime(CLOCK_REALTIME, &beerFill_end);
}
#endif

/* Map the table entires into matrix for tensor core to use 
 * Assum both matrix have the same dimension and the value in INT type for now, e.g., both 16x16 dim
 * To support multiple types, this function need to be modified
 */

// micro benchmark for simple matrix multiplication query
__host__ void static micro_mm(struct joinNode *jNode, float * matrix1, float * matrix2, int width,
        int attr_num1, int attr_num2, int attr_type1, int attr_type2) {
    int *mat1_i, *mat1_j, *mat1_val; // row index, col index, value
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
        int k = 0; // k is row-index of the table (tupleNum index)
        
        for (j = 0; j < leftTupleNum * attr_type1; j+=attr_type1) {
            int *temp;
            temp = (int*)(&jNode->leftTable->content[i][j]);
            
            if (left_col_idx == 0) { // match to schema's i
                mat1_i[k] = *temp;
            }
            else if (left_col_idx == 1) { // match to schema's j
                mat1_j[k] = *temp;
            }
            else { // match to schema's val
                // read 4 bytes at once because the type is int
                mat1_val[k] = *temp;
            }
            k++;
        }
    }

    
    for (i = 0; i < attr_num2; i++) {
        int right_col_idx = jNode->rightTable->attrIndex[i];
        int k = 0;
        
        for (j = 0; j < rightTupleNum * attr_type2; j+=attr_type2) {
            int *temp;
            temp = (int*)(&jNode->rightTable->content[i][j]);
            
            if (right_col_idx == 0) {
                mat2_i[k] = *temp;
            }
            else if (right_col_idx == 1) {
                mat2_j[k] = *temp;
            }
            else {
                mat2_val[k] = *temp;
            }
            k++;
        }
    }

    // map index to array[width * i + j] = val
    // prepare two matrices (1-D array format) for WMMA
    int m;
    for (m = 0; m < leftTupleNum; m++) {
        matrix1[width * mat1_i[m] + mat1_j[m]] = (float)mat1_val[m];
        //printf("%.2f\t", matrix1[width * mat1_i[m] + mat1_j[m]]);
    }
    //printf("\n");

    //printf("rightTupleNum: %d\n", rightTupleNum);
    for (m = 0; m < rightTupleNum; m++) {
        //printf("mat2 val: %d\tn: %d\n", mat2_val[n], n);
        matrix2[width * mat2_i[m] + mat2_j[m]] = (float)mat2_val[m];
        //printf("%.2f\t", matrix2[width * mat2_i[m] + mat2_j[m]]);
    }
    //printf("\n");

    free(mat1_i);
    free(mat1_j);
    free(mat1_val);
    free(mat2_i);
    free(mat2_j);
    free(mat2_val);
}

/* Print matrix content in device memory */
#ifdef DEBUG
__global__ void static verify_gpuResult(half * matrix, int width) {
    int i;
    for (i = 0; i < width*width; i++) {
        cuPrintf("%.1f\t", __half2float(matrix[i]));
        if ((i+1) % width == 0)
          cuPrintf("\n");  
    }

}
#endif

/* Convert input data from half to float type */
__global__ void static convertFp16ToFp32(float *out, half *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

/* Convert matrix from int to half type */
__global__ void static convertFp32ToFp16(half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

/* Convert matrix from short to half type */
__global__ void static convertShortToFp16(half *out, short *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __short2half_rd(in[idx]);
    }
}

/* Convert matrix from int to float type */
__host__ void static convertIntToFp32(float *out, int *in, int width) {
    int i;
    for (i = 0; i < width * width; i++) {
        out[i] = (float)in[i]; 
    }
}

/* Check whether the tupleNum is multiple of 16 because the WMMA requires the width of matrix be multiple of 16 */
__host__ int static findMatWidth(int tupleNum) {
    if (tupleNum <= 256)
        return 16;
    else {
        int tmp = ceil(sqrt(tupleNum));
        return (int)(ceil(tmp/(float)16)*16);
    }
}

__device__ static float getVal(char **content, struct mathExp exp, int pos) {
    float res;
    if (exp.opType == CONS)
        res = exp.opValue;
    else {
        int index = exp.opValue;
        res = ((int *)(content[index]))[pos];
    }

    return res;
}

// since WMMA perform C = alpha*A*B+beta*C, here we just fill operator MULTIPLY
__device__ static void fillMathExp(char **content, struct mathExp exp, int pos, float * A, float * B) {

    if (exp.op == MULTIPLY) {
        if (((struct mathExp*)exp.exp)[0].op == NOOP)
            A[pos] = getVal(content, ((struct mathExp*)exp.exp)[0], pos);
        if (((struct mathExp*)exp.exp)[1].op == NOOP)
            B[pos] = getVal(content, ((struct mathExp*)exp.exp)[1], pos);
    }
        
    return;
}

//#ifdef WMMA_HALF
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
//#endif

#if defined(RED) || defined(REDHALF)
__host__ static void setVector(float *vec, int n) {
    for (int i = 0; i < n; i++)
        vec[i] = 1.0;
}
/*
#elif REDHALF
__host__ static void setVector(half *vec, int n) {
    for (int i = 0; i < n; i++)
        vec[i] = __float2half(1.0f);
}
*/
#endif

__host__ static void setRed(short *red, int n) {
    for (int i = 0; i < n; i++)
        red[i] = (short)1;
}

__global__ static void agg_cal_cons(char ** content, int colNum, struct groupByExp* exp, long tupleNum, float * A, float * B) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=index;i<tupleNum;i+=stride){
        for(int j=0;j<colNum;j++){
            int func = exp[j].func;
            // for now, we only care about SUM
            if (func == SUM) {
                // 1. fill data into two matrices
                //transform_data(content, exp[j].exp, i, A, B);
                fillMathExp(content, exp[j].exp, i, A, B);
                // maybe the order does not important if we can get relative ranking

                // 2. copy data into device using cudaMemcpy (if directly assign in device memory, can avoid this step)

            } else if (func == AVG) {
                // not the main point now
            }
        }
    }
} 

/* Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
 *  1) Matrices are packed in memory.
 *  2) M, N and K are multiples of 16.
 *  3) Neither A nor B are transposed.
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
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
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
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);

        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}

#ifdef WMMA_INT4
__global__ void wmma_int(signed char *a, signed char *b, int *c, int M, int N, int K, int alpha, int beta) {
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    wmma::fill_fragment(acc_frag, 0);

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
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);

        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}
#endif

/*
 * tcuJoinn using NVIDIA's WMMA lib to perform matrix multiplication can aggregation..
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
struct tableNode * tcuJoin(struct joinNode *jNode, struct statistic *pp, int *matrix_dim){
#ifdef DEBUG
    cudaPrintfInit();
#endif
    int leftTupleNum = jNode->leftTable->tupleNum;
    int rightTupleNum = jNode->rightTable->tupleNum;

    // parse user input dimension from command line
    //int MATRIX_M, MATRIX_N, MATRIX_K;
    uint64_t MATRIX_M, MATRIX_N, MATRIX_K; // use uint64_t to avoid overflow
#if defined(WMMA_HALF) || defined(WMMA_INT4)
    // WMMA requires the dimension to be multiple of 16
    MATRIX_M = (uint64_t)nearestMultipleN(leftTupleNum, 16);
    MATRIX_N = (uint64_t)nearestMultipleN(rightTupleNum, 16);
#else
    MATRIX_M = (uint64_t)leftTupleNum;
    MATRIX_N = (uint64_t)rightTupleNum;
#endif
    MATRIX_K = *matrix_dim; // user input, matrix width (if WMMA, multiple of 16)

#ifdef DEBUG
    printf("left  tuple #: %d\n", leftTupleNum);
    printf("right tuple #: %d\n", rightTupleNum);
    printf("MATRIX_M: %lu\n", MATRIX_M);
    printf("MATRIX_N: %lu\n", MATRIX_N);
    printf("MATRIX_K: %lu\n", MATRIX_K);
#endif

#if defined(CUBLAS_HALF) || defined(CUBLAS)
    struct timespec debug_start, debug_end; // cublasCreate has init overhead
    struct timespec count_start, count_end;
    struct timespec transpose_start, transpose_end;
#endif
    struct timespec tcu_start, tcu_end;
    struct timespec init_start, init_end;
    struct timespec fill_start, fill_end;
    //struct timespec convert_start, convert_end;
    struct timespec cuMemcpy_start, cuMemcpy_end;
    clock_gettime(CLOCK_REALTIME, &tcu_start);
    clock_gettime(CLOCK_REALTIME, &init_start);

#ifdef WMMA_INT4
    signed char *h_int_A, *h_int_B; // host int4 array
    signed char *d_int_A, *d_int_B; // device int4 array
    int *c_int_wmma, *c_host_int_wmma;
    int alpha = 1;
    int beta = 0;
#elif CUBLAS_HALF
    //short *h_short_A, *h_short_B;
    half *d_fp16_A, *d_fp16_B, *d_fp16_BT;
    char *gpu_fact, *gpu_dim; // raw data
    float *c_cublas;
    half *c_fp16_cublas;
    float *c_host_cublas;

    //float *h_vec, *d_vec, *d_temp; // SGEMV
#if defined(RED)|| defined(REDHALF)
    float *h_red, *d_red;
    float *h_red2, *d_red2;
    /*
#elif REDHALF
    half *h_red, *d_red;
    half *h_red2, *d_red2;
    */
#endif
    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_fp16 = __float2half(1.0f);
    half beta_fp16 = __float2half(1.0f);

    // use tensor core or cublas
    cublasHandle_t cublasHandle; // cublas tcu
    cudaEvent_t startcublasEX;
    cudaEvent_t stopcublasEX;

    cudaErrCheck(cudaEventCreate(&startcublasEX));
    cudaErrCheck(cudaEventCreate(&stopcublasEX));

    clock_gettime(CLOCK_REALTIME, &debug_start);
    cublasErrCheck(cublasCreate(&cublasHandle));
    clock_gettime(CLOCK_REALTIME, &debug_end);
    // enable tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

#ifdef BEER
    /* Beer dataset */
    short *h_id_A, *h_id_B;
    short *h_beer_A, *h_beer_B;
    short *h_factory_A, *h_factory_B;
    short *h_style_A, *h_style_B;
    short *h_abv_A, *h_abv_B;
#elif ITUNES
    /* iTunes dataset */
    short *h_id_A, *h_id_B;
    short *h_song_A, *h_song_B;
    short *h_artist_A, *h_artist_B;
    short *h_album_A, *h_album_B;
    short *h_genre_A, *h_genre_B;
    short *h_price_A, *h_price_B;
    short *h_copyright_A, *h_copyright_B;
    short *h_time_A, *h_time_B;
    short *h_released_A, *h_released_B;
#endif

#elif WMMA_HALF
    float *h_fp32_A, *h_fp32_B; // host float32 array
    float *d_fp32_A, *d_fp32_B; // device float32 array
    half *d_fp16_A, *d_fp16_B;
    float *c_wmma, *c_wmma_sum1, *c_wmma_sum2, *c_host_wmma;
    float alpha = 1.0f;
    float beta = 0.0f;

    //mask for reduction
    float *d_fp32_mask, *h_fp32_mask;
    float *d_fp32_mask2, *h_fp32_mask2;
    half *d_fp16_mask, *d_fp16_mask2;

#else // CUBLAS SGEMM
    float *h_fp32_A, *h_fp32_B; // host float32 array
    float *d_fp32_A, *d_fp32_B, *d_fp32_BT; // device float32 array
    float *c_sgemm, *c_host_sgemm;
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t cublasHandle_default; // cublas default
    cudaEvent_t startcublas; // for sgemm (FP32)
    cudaEvent_t stopcublas;

    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));
    clock_gettime(CLOCK_REALTIME, &debug_start);
    cublasErrCheck(cublasCreate(&cublasHandle_default));
    clock_gettime(CLOCK_REALTIME, &debug_end);
    cublasErrCheck(cublasSetMathMode(cublasHandle_default,CUBLAS_DEFAULT_MATH));
#endif

#if defined(WMMA_HALF) || defined(WMMA_INT4)
    cudaEvent_t startWMMA;
    cudaEvent_t stopWMMA;
    cudaErrCheck(cudaEventCreate(&startWMMA));
    cudaErrCheck(cudaEventCreate(&stopWMMA)); 

    dim3 gridDim;
    dim3 blockDim;
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
#endif

#ifdef WMMA_INT4
    h_int_A = (signed char*)calloc(MATRIX_M*MATRIX_K, sizeof(signed char));
    h_int_B = (signed char*)calloc(MATRIX_K*MATRIX_N, sizeof(signed char));
    c_host_int_wmma = (int*)calloc(MATRIX_M*MATRIX_N, sizeof(int));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_int_A, MATRIX_M * MATRIX_K * sizeof(signed char)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_int_B, MATRIX_K * MATRIX_N * sizeof(signed char)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_int_wmma, MATRIX_M * MATRIX_N * sizeof(int)));
#elif CUBLAS_HALF

#if defined(BEER) || defined(ITUNES)
    // if materialized view is required
    short *h_short_A, *h_short_B;
    h_short_A = (short*)calloc(MATRIX_M*MATRIX_K, sizeof(short));
    h_short_B = (short*)calloc(MATRIX_N*MATRIX_K, sizeof(short));
#endif

    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
    long primaryKeySize = sizeof(int) * jNode->rightTable->tupleNum;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact,foreignKeySize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));
    c_host_cublas = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_cublas, (uint64_t)MATRIX_M * (uint64_t)MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_A, (uint64_t)MATRIX_M * (uint64_t)MATRIX_K * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_B, (uint64_t)MATRIX_N * (uint64_t)MATRIX_K * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_BT, (uint64_t)MATRIX_K * (uint64_t)MATRIX_N * sizeof(half)));
    // SGEMV
//    h_vec = (float*)calloc(MATRIX_M, sizeof(float));
//    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_vec, MATRIX_M * sizeof(float)));
//    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_temp, MATRIX_N * sizeof(float)));

#ifdef BEER
    /* Beer dataset */
    h_id_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_beer_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_factory_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_style_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_abv_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_id_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_beer_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_factory_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_style_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_abv_B = (short*)calloc(MATRIX_N, sizeof(short));
#elif ITUNES
    h_id_A = (short*)calloc(MATRIX_M, sizeof(short)); 
    h_id_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_song_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_song_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_artist_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_artist_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_album_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_album_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_genre_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_genre_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_price_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_price_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_copyright_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_copyright_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_time_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_time_B = (short*)calloc(MATRIX_N, sizeof(short));
    h_released_A = (short*)calloc(MATRIX_M, sizeof(short));
    h_released_B = (short*)calloc(MATRIX_N, sizeof(short));
#endif

#if defined(RED)|| defined(REDHALF)
    h_red = (float*)calloc(MATRIX_N, sizeof(float));
    h_red2 = (float*)calloc(MATRIX_M, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_red, MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_red2, MATRIX_M * sizeof(float)));
    /*
#elif REDHALF
    h_red = (half*)calloc(MATRIX_N, sizeof(half));
    h_red2 = (half*)calloc(MATRIX_M, sizeof(half));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_red, MATRIX_N * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_red2, MATRIX_M * sizeof(half)));
    */
#endif
#elif CUBLAS
    h_fp32_A =     (float*)calloc(MATRIX_M*MATRIX_K, sizeof(float));
    h_fp32_B =     (float*)calloc(MATRIX_N*MATRIX_K, sizeof(float));
    c_host_sgemm = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_A, MATRIX_M * MATRIX_K * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_B, MATRIX_N * MATRIX_K * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_BT, MATRIX_K * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_sgemm, MATRIX_M * MATRIX_N * sizeof(float)));
#else  //WMMA_HALF
    h_fp32_A = (float*)calloc(MATRIX_M*MATRIX_K, sizeof(float));
    h_fp32_B = (float*)calloc(MATRIX_K*MATRIX_N, sizeof(float));
    c_host_wmma = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_A, MATRIX_M * MATRIX_K * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_fp32_B, MATRIX_K * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_A, MATRIX_M * MATRIX_K * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_fp16_B, MATRIX_K * MATRIX_N * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_wmma_sum1, MATRIX_M * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_wmma_sum2, MATRIX_M * MATRIX_N * sizeof(float)));

    // mask
    h_fp32_mask = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    h_fp32_mask2 = (float*)calloc(MATRIX_M*MATRIX_N, sizeof(float));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_mask, MATRIX_M * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp32_mask2, MATRIX_M * MATRIX_N * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_mask, MATRIX_M * MATRIX_N * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_fp16_mask2, MATRIX_M * MATRIX_N * sizeof(half)));
    set_mask(h_fp32_mask, MATRIX_M, MATRIX_N);
    set_mask2(h_fp32_mask2, MATRIX_M, MATRIX_N);
#endif
    clock_gettime(CLOCK_REALTIME, &init_end);

//    clock_gettime(CLOCK_REALTIME, &fill_start); 
#ifdef WMMA_INT4
    tcu_match(jNode, MATRIX_K, h_int_A, h_int_B, jNode->leftTable->attrType[0], jNode->rightTable->attrType[0]);
#elif CUBLAS_HALF

#ifdef BEER
    beer_match(jNode, MATRIX_K,
         h_short_A, h_short_B, 
         jNode->leftTable->attrType[0], jNode->rightTable->attrType[0], 
         jNode->leftTable->totalAttr, jNode->rightTable->totalAttr,
         h_id_A, h_id_B,
         h_beer_A, h_beer_B,
         h_factory_A, h_factory_B,
         h_style_A, h_style_B,
         h_abv_A, h_abv_B);
#elif ITUNES
    //clock_gettime(CLOCK_REALTIME, &itunesFill_start);
    itunes_match(jNode, MATRIX_K,
         h_short_A, h_short_B, 
         jNode->leftTable->attrType[0], jNode->rightTable->attrType[0], 
         jNode->leftTable->totalAttr, jNode->rightTable->totalAttr,
         h_id_A, h_id_B,
         h_song_A, h_song_B,
         h_artist_A, h_artist_B,
         h_album_A, h_album_B,
         h_genre_A, h_genre_B,
         h_price_A, h_price_B,
         h_copyright_A, h_copyright_B,
         h_time_A, h_time_B,
         h_released_A, h_released_B
         );
    //clock_gettime(CLOCK_REALTIME, &itunesFill_end);
#else // matrix-multiplication join count
    //tcu_match(jNode, MATRIX_K, h_short_A, h_short_B, jNode->leftTable->attrType[0], jNode->rightTable->attrType[0]);
    int A_tupleNum = jNode->leftTable->tupleNum;
    int B_tupleNum = jNode->rightTable->tupleNum;
    // cudaMemcpyHostToDevice raw data->char *column
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_start);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);
    //TODO: test result by copy D->H
    clock_gettime(CLOCK_REALTIME, &fill_start); 
    gpu_fill<<<(MAX_THREADS + A_tupleNum -1)/MAX_THREADS, MAX_THREADS>>> (gpu_fact,
            MATRIX_K,
            d_fp16_A,
            A_tupleNum,
            jNode->leftTable->attrType[0]);
    gpu_fill<<<(MAX_THREADS + B_tupleNum -1)/MAX_THREADS, MAX_THREADS>>> (gpu_dim,
            MATRIX_K,
            d_fp16_B,
            B_tupleNum,
            jNode->rightTable->attrType[0]);
    clock_gettime(CLOCK_REALTIME, &fill_end); 

#endif

#else  //WMMA_HALF or CUBLAS    
    tcu_match(jNode, MATRIX_K, h_fp32_A, h_fp32_B, jNode->leftTable->attrType[0], jNode->rightTable->attrType[0]);
#ifdef CUBLAS
    /*    
    micro_mm(jNode, h_fp32_A, h_fp32_B, MATRIX_M,
            jNode->leftTable->totalAttr, jNode->rightTable->totalAttr, 
            jNode->leftTable->attrType[0], jNode->rightTable->attrType[0]);
    */
#endif
#endif
//    clock_gettime(CLOCK_REALTIME, &fill_end);

#ifdef CUBLAS_HALF
#if defined(RED) || defined(REDHALF)
    setVector(h_red, MATRIX_N);
    setVector(h_red2, MATRIX_M);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_red, h_red, sizeof(float) * MATRIX_N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_red2, h_red2, sizeof(float) * MATRIX_M, cudaMemcpyHostToDevice));
    /*
#elif REDHALF
    setVector(h_red, MATRIX_N);
    setVector(h_red2, MATRIX_M);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_red, h_red, sizeof(half) * MATRIX_N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_red2, h_red2, sizeof(half) * MATRIX_M, cudaMemcpyHostToDevice));
    */
#endif
#endif

//    clock_gettime(CLOCK_REALTIME, &cuMemcpy_start);
#ifdef WMMA_INT4
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_int_A, h_int_A, sizeof(signed char) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_int_B, h_int_B, sizeof(signed char) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));
#elif WMMA_HALF
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_A, h_fp32_A, sizeof(float) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
    convertFp32ToFp16<<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_fp16_A, d_fp32_A, MATRIX_M * MATRIX_K);
    cudaErrCheck(cudaFree(d_fp32_A));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_B, h_fp32_B, sizeof(float) * MATRIX_K * MATRIX_N, cudaMemcpyHostToDevice));
    convertFp32ToFp16<<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (d_fp16_B, d_fp32_B, MATRIX_K * MATRIX_N);
    cudaErrCheck(cudaFree(d_fp32_B));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_mask, h_fp32_mask, sizeof(float) * MATRIX_M * MATRIX_N, cudaMemcpyHostToDevice));
    convertFp32ToFp16<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp16_mask, d_fp32_mask, MATRIX_M * MATRIX_N);
    cudaErrCheck(cudaFree(d_fp32_mask));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_mask2, h_fp32_mask2, sizeof(float) * MATRIX_M * MATRIX_N, cudaMemcpyHostToDevice));
    convertFp32ToFp16<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp16_mask2, d_fp32_mask2, MATRIX_M * MATRIX_N);
    cudaErrCheck(cudaFree(d_fp32_mask2));
#elif CUBLAS_HALF
    //directly fill in half format on gpu
//    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp16_A, h_short_A, sizeof(half) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
//    convertShortToFp16<<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_fp16_A, (short*)d_fp16_A, MATRIX_M * MATRIX_K);
//    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_short_B, h_short_B, sizeof(short) * MATRIX_N * MATRIX_K, cudaMemcpyHostToDevice));
    /*
    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 32;
    gridDim.x = (MATRIX_M + (blockDim.x - 1)) / (blockDim.x);
    gridDim.y = (MATRIX_K + blockDim.y - 1) / (blockDim.y);
    gpu_transpose<<< gridDim, blockDim >>> (d_short_BT, d_short_B, MATRIX_N, MATRIX_K);
    */
    clock_gettime(CLOCK_REALTIME, &transpose_start);
    //gpu_transpose<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_short_BT, d_short_B, MATRIX_N, MATRIX_K);
    gpu_transpose<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp16_BT, d_fp16_B, MATRIX_N, MATRIX_K);

    //gpu_transpose<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp16_BT, d_short_B, MATRIX_N, MATRIX_K);
    clock_gettime(CLOCK_REALTIME, &transpose_end);
//    cudaErrCheck(cudaFree(d_short_B));

    //CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp16_BT, h_short_BT, sizeof(half) * MATRIX_N * MATRIX_K, cudaMemcpyHostToDevice));
    //convertShortToFp16<<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (d_fp16_BT, (short*)d_fp16_BT, MATRIX_N * MATRIX_K);
#else // CUBLAS
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_A, h_fp32_A, sizeof(float) * MATRIX_M * MATRIX_K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_fp32_B, h_fp32_B, sizeof(float) * MATRIX_N * MATRIX_K, cudaMemcpyHostToDevice));
    gpu_transpose<<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (d_fp32_BT, d_fp32_B, MATRIX_N, MATRIX_K);
    cudaErrCheck(cudaFree(d_fp32_B));
#endif
//    clock_gettime(CLOCK_REALTIME, &cuMemcpy_end);

#ifdef WMMA_HALF 
    printf("\nM = %d, N = %d, K = %d. alpha = %.2f, beta = %.2f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

    printf("Running with wmma...\n");
    cudaErrCheck(cudaEventRecord(startWMMA));
    wmma_example <<< gridDim, blockDim >>> (d_fp16_A, d_fp16_B, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta); 
    cudaErrCheck(cudaFree(d_fp16_A));
    cudaErrCheck(cudaFree(d_fp16_B));
    // TODO: mask for WMMA has some bugs 
    // perform additional two wmma for reduction, sum will be c_host_wmma[0]
    /*
    half *c_wmma_reduction1, *c_wmma_reduction2;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&c_wmma_reduction1, MATRIX_M * MATRIX_N * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&c_wmma_reduction2, MATRIX_M * MATRIX_N * sizeof(half)));
    convertFp32ToFp16<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_wmma_reduction1, c_wmma, MATRIX_M * MATRIX_N);

    wmma_example <<< gridDim, blockDim >>> (d_fp16_mask2, c_wmma_reduction1, c_wmma_sum1, MATRIX_M, MATRIX_N, MATRIX_M, alpha, beta); 
    convertFp32ToFp16<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_wmma_reduction2, c_wmma_sum1, MATRIX_M * MATRIX_N);

    wmma_example <<< gridDim, blockDim >>> (d_fp16_mask2, c_wmma_reduction2, c_wmma_sum2, MATRIX_M, MATRIX_N, MATRIX_M, alpha, beta); 
    */
    cudaErrCheck(cudaFree(d_fp16_mask2));
    cudaErrCheck(cudaEventRecord(stopWMMA));

#elif WMMA_INT4
    printf("\nM = %d, N = %d, K = %d. alpha = %d, beta = %d\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

    printf("Running with wmma_INT4...\n");
    cudaErrCheck(cudaEventRecord(startWMMA));
    wmma_int <<< gridDim, blockDim >>> (d_int_A, d_int_B, c_int_wmma, MATRIX_M, MATRIX_N, MATRIX_K, 1, 0); 
    cudaErrCheck(cudaFree(d_int_A));
    cudaErrCheck(cudaFree(d_int_B));
    cudaErrCheck(cudaEventRecord(stopWMMA));

#elif CUBLAS_HALF
    //setVector(h_vec, MATRIX_M);
    //cublasSetVector(MATRIX_M, sizeof(float), &(h_vec[0]), 1, d_vec, 1);
    //float *temp;
    //temp = (float *)malloc(MATRIX_N*sizeof(float));

#if defined(RED)||defined(REDHALF)
    float *red_sum, *red_sum2;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&red_sum, MATRIX_M * sizeof(float)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&red_sum2, 1 * sizeof(float)));
    /*
#elif REDHALF
    half *red_sum, *red_sum2;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&red_sum, MATRIX_M * sizeof(half)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&red_sum2, 1 * sizeof(half)));
    */
#endif
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&c_fp16_cublas, (uint64_t)MATRIX_M * (uint64_t)MATRIX_N * sizeof(half)));
    printf("Running with cuBLAS on TCUs...\n");
    cudaErrCheck(cudaEventRecord(startcublasEX));
/*    cublasErrCheck(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha_fp16,
                d_fp16_BT,MATRIX_N,
                d_fp16_A,MATRIX_K,
                &beta_fp16,
                c_fp16_cublas, MATRIX_N));*/
#ifdef RED    
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha,
                d_fp16_BT, CUDA_R_16F, MATRIX_N,
                d_fp16_A, CUDA_R_16F, MATRIX_K,
                &beta,
                //c_fp16_cublas, CUDA_R_16F, MATRIX_N,
                //CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
                c_cublas, CUDA_R_32F, MATRIX_N,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
#else
    cublasErrCheck(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M, MATRIX_K,
                &alpha_fp16,
                d_fp16_BT,MATRIX_N,
                d_fp16_A,MATRIX_K,
                &beta_fp16,
                c_fp16_cublas, MATRIX_N));
#endif

    cudaErrCheck(cudaFree(d_fp16_A));
    cudaErrCheck(cudaFree(d_fp16_BT));

#ifdef REDHALF
    convertFp16ToFp32<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_cublas, c_fp16_cublas, MATRIX_M * MATRIX_N);
#endif
#if defined(RED) || defined(REDHALF)
    // 1st reduction
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, MATRIX_M, MATRIX_N,
                &alpha,
                d_red, CUDA_R_32F, 1,
                c_cublas, CUDA_R_32F, MATRIX_N,
                &beta,
                red_sum, CUDA_R_32F, 1,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu

    // 2nd reduction
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, 1, MATRIX_M,
                &alpha,
                red_sum, CUDA_R_32F, 1,
                d_red2, CUDA_R_32F, MATRIX_M,
                &beta,
                red_sum2, CUDA_R_32F, 1,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP)); // tcu
    
//#elif REDHALF
//    convertFp16ToFp32<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_cublas, c_fp16_cublas, MATRIX_M * MATRIX_N);
    /*
    cublasErrCheck(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, MATRIX_M, MATRIX_N,
                &alpha_fp16,
                d_red,1,
                c_fp16_cublas,MATRIX_N,
                &beta_fp16,
                red_sum, 1));

    cublasErrCheck(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                1, 1, MATRIX_M,
                &alpha_fp16,
                red_sum,1,
                d_red2,MATRIX_M,
                &beta_fp16,
                red_sum2, 1));
                */
#endif
    /*   
    // SGEMV (FP32)
    cublasErrCheck(cublasSgemv(cublasHandle, CUBLAS_OP_N,
                MATRIX_N, MATRIX_M,
                &alpha
                , c_cublas, MATRIX_N,
                d_vec, 1,
                &beta
                , d_temp, 1));
    cublasErrCheck(cublasGetVector(MATRIX_N, sizeof(float), d_temp, 1,temp,1));
    print_vector(temp, MATRIX_N);
    printf("\n");
    */

    cudaErrCheck(cudaEventRecord(stopcublasEX));
#elif CUBLAS
    printf("Running with sgemm...\n");
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

#ifdef WMMA_HALF
    struct timespec tmp_start, tmp_end;
    clock_gettime(CLOCK_REALTIME, &tmp_start);
    cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaFree(c_wmma));
#ifdef DEBUG
    printf("c_host_wmma:\n");
    verify_result(c_host_wmma, MATRIX_M, MATRIX_N);
#endif
    printf("Number of join results (MM reduction): %.0f\n", c_host_wmma[0]);
    printf("Number of join results (CPU count): %d\n", sum_matrix(c_host_wmma, MATRIX_M, MATRIX_N));
    clock_gettime(CLOCK_REALTIME, &tmp_end);
#elif WMMA_INT4
    struct timespec tmp_start, tmp_end;
    clock_gettime(CLOCK_REALTIME, &tmp_start);
    cudaErrCheck(cudaMemcpy(c_host_int_wmma, c_int_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaFree(c_int_wmma));
#ifdef DEBUG
    printf("c_host_int_wmma:\n");
    verify_result_int(c_host_int_wmma, MATRIX_M, MATRIX_N);
#endif
    printf("Number of join results (CPU count): %d\n", sum_matrix_int(c_host_int_wmma, MATRIX_M, MATRIX_N));
    clock_gettime(CLOCK_REALTIME, &tmp_end);

#elif CUBLAS_HALF

#elif CUBLAS
#ifdef DEBUG
    cudaErrCheck(cudaMemcpy(c_host_sgemm, c_sgemm, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
#endif
#endif

    // print error checking, cublasGemmEx and cublas
    //printf("\nChecking results with tensor cores...\n");

    // 0.01% relative tolerance. 1e-5 absolute tolerance.
    /*
    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = c_host_sgemm[i];
        float v2 = c_host_cublas[i];

        // TODO: abs diff failed due to precision loss
        // current fix: range value less than 2^10 (IEEE half type)
        if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-3) {
            errors++;
            if (errors < 10) printf("%.1f %.1f diff:%.1f\n", v1, v2, abs(v1-v2));
        }
    }

    if (errors > 0) {
        printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
    }
    */
#ifdef CUBLAS_HALF
    float cublasEXTime;

    cudaErrCheck(cudaEventSynchronize(stopcublasEX));
    cudaErrCheck(cudaEventElapsedTime(&cublasEXTime, startcublasEX, stopcublasEX));
    // retrieve other attributes from c_host_cublas given indices

#ifdef BEER
    /*
    attrProjection(c_host_cublas, MATRIX_M, MATRIX_N,
            jNode,
            h_id_A, h_id_B,
            h_beer_A, h_beer_B,
            h_factory_A, h_factory_B,
            h_style_A, h_style_B,
            h_abv_A, h_abv_B);
            */
#elif ITUNES
   /* 
    attrProjection(c_host_cublas, MATRIX_M, MATRIX_N,
            jNode,
            h_id_A, h_id_B,
            h_song_A, h_song_B,
            h_artist_A, h_artist_B,
            h_album_A, h_album_B,
            h_genre_A, h_genre_B,
            h_price_A, h_price_B,
            h_copyright_A, h_copyright_B,
            h_time_A, h_time_B,
            h_released_A, h_released_B);
            */
#endif

#if defined(RED) || defined(REDHALF)

    /*
#ifdef REDHALF
    float *red_sum2_fp32;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&red_sum2_fp32, 1 * sizeof(float)));
    convertFp16ToFp32<<< (1 + 255) / 256, 256 >>> (red_sum2_fp32, red_sum2, 1);
#endif
    */
    float *ans;
    ans = (float*)calloc(1, sizeof(float));
    clock_gettime(CLOCK_REALTIME, &count_start);
#if defined(RED) || defined(REDHALF)
    cudaErrCheck(cudaMemcpy(ans, red_sum2, 1 * sizeof(float), cudaMemcpyDeviceToHost));
//#elif REDHALF
//    cudaErrCheck(cudaMemcpy(ans, red_sum2_fp32, 1 * sizeof(float), cudaMemcpyDeviceToHost));
#endif
    clock_gettime(CLOCK_REALTIME, &count_end);
    printf("c_host_cublas reduction sum: %.0f\n", ans[0]);
    free(ans);
    cudaErrCheck(cudaFree(red_sum));
    cudaErrCheck(cudaFree(red_sum2));
#else

    convertFp16ToFp32<<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_cublas, c_fp16_cublas, MATRIX_M * MATRIX_N);

    uint64_t input_len = MATRIX_M*MATRIX_N;
    int asum_len = 200000000; // Sasum addition per section
    clock_gettime(CLOCK_REALTIME, &count_start);

    cublasStatus_t ret;
    ret = cublasCreate(&cublasHandle);
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
        }
        ret = cublasSasum(cublasHandle, remain, c_cublas+pos, 1, &cb_res);
        sum_res += (uint64_t)cb_res;
        clock_gettime(CLOCK_REALTIME, &count_end);
        printf("c_host_cublas sum: %lu\n", sum_res);
    }
#endif
    printf("cublasEX tensor cores (FP16) took %fms\n", cublasEXTime);
    
    cudaErrCheck(cudaEventDestroy(startcublasEX));
    cudaErrCheck(cudaEventDestroy(stopcublasEX));
    free(c_host_cublas);
    cudaErrCheck(cudaFree(c_cublas));
#elif CUBLAS
    float cublasTime;

    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    clock_gettime(CLOCK_REALTIME, &count_start);
    cublasStatus_t sgemm_ret;
    sgemm_ret = cublasCreate(&cublasHandle_default);
    float *cbsgemm_res = (float*)malloc(sizeof(float));
    sgemm_ret = cublasSasum(cublasHandle_default, MATRIX_M*MATRIX_N, c_sgemm, 1, cbsgemm_res);
    clock_gettime(CLOCK_REALTIME, &count_end);
    printf("c_host_sgemm sum: %.0f\n", *cbsgemm_res);
    printf("cublas sgemm (FP32) took %fms\n", cublasTime);

    cudaErrCheck(cudaEventDestroy(startcublas));
    cudaErrCheck(cudaEventDestroy(stopcublas));
    free(c_host_sgemm);
    cudaErrCheck(cudaFree(c_sgemm));
#else
    float wmmaTime;

    cudaErrCheck(cudaEventSynchronize(stopWMMA));
    cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    printf("wmma took %fms\n", wmmaTime);

    cudaErrCheck(cudaEventDestroy(startWMMA));
    cudaErrCheck(cudaEventDestroy(stopWMMA));
#endif

// free those data structures
#ifdef WMMA_HALF
    free(h_fp32_A);
    free(h_fp32_B);
    free(h_fp32_mask2);
    free(c_host_wmma);
#elif WMMA_INT4
    free(h_int_A);
    free(h_int_B);
    free(c_host_int_wmma);
#elif CUBLAS_HALF

#if defined(BEER) || defined(ITUNES)
    free(h_short_A);
    free(h_short_B);
#endif

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
    double debug_elapse = (debug_end.tv_sec -  debug_start.tv_sec)* BILLION + debug_end.tv_nsec - debug_start.tv_nsec;
    double transpose_elapse = (transpose_end.tv_sec -  transpose_start.tv_sec)* BILLION + transpose_end.tv_nsec - transpose_start.tv_nsec;
#endif
#ifdef BEER
    double beerFill_elapse = (beerFill_end.tv_sec -  beerFill_start.tv_sec)* BILLION + beerFill_end.tv_nsec - beerFill_start.tv_nsec;
#elif ITUNES
    double itunesFill_elapse = (itunesFill_end.tv_sec -  itunesFill_start.tv_sec)* BILLION + itunesFill_end.tv_nsec - itunesFill_start.tv_nsec;
#endif
#ifdef WMMA_HALF
    double tmp_elapse = (tmp_end.tv_sec -  tmp_start.tv_sec)* BILLION + tmp_end.tv_nsec - tmp_start.tv_nsec;
#endif
    
    printf("Initialization: %lf(ms)\n", init_elapse/(1000*1000));
    printf("Matrices filling: %lf(ms)\n", tcu_fill/(1000*1000));
    //printf("Data type convertion: %lf(ms)\n", tcu_convert/(1000*1000));
    printf("cudaMemcpy: %lf(ms)\n", cuMemcpy_elapse/(1000*1000));
    printf("MMA total time: %lf(ms)\n", tcu_elapse/(1000*1000));
#ifdef CUBLAS_HALF
    printf("cublasEX sum counting: %lf(ms)\n", count_elapse/(1000*1000));
    printf("debug (cublasCreate): %lf(ms)\n", debug_elapse/(1000*1000));
    printf("gpu transpose: %lf(ms)\n", transpose_elapse/(1000*1000));
#ifdef BEER
    printf("Beer filling time: %lf(ms)\n", beerFill_elapse/(1000*1000));
#elif ITUNES
    printf("iTunes filling time: %lf(ms)\n", itunesFill_elapse/(1000*1000));
#endif
#elif CUBLAS
    printf("cublasSGEMM sum counting: %lf(ms)\n", count_elapse/(1000*1000));
    printf("debug (cublasCreate): %lf(ms)\n", debug_elapse/(1000*1000));
#elif WMMA_HALF
    printf("Result verification: %lf(ms)\n", tmp_elapse/(1000*1000));
#endif
#ifdef DEBUG
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif
    return 0; // non-void function

}
