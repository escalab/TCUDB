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

#ifndef __GPU_CUDALIB_H_
#define __GPU_CUDALIB_H_

#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                                \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                        \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


#define CUDA_SAFE_CALL_NO_SYNC_RETRY( call) do {                                \
    cudaError err = call;                                                    \
    while( cudaSuccess != err) {                                                \
    err = call;                             \
    } } while (0)

static size_t getGpuGlobalMem(int deviceID){
        size_t free = 0, total = 0;

        cudaMemGetInfo(&free,&total);
        return total;
}

static long getAvailMem(void){
    long mem = sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    return mem;
}

static double getCurrentTime(void){
    struct timeval tv;
        gettimeofday(&tv, NULL);
        double curr  = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000 ;
    return curr;
}


/*
 * Transform integer to string using one single gpu thread.
 */

__device__ static char * gpuItoa(int value, char* result, int base){

        if (base < 2 || base > 36) {
                *result = '\0';
                return result;
        }

        char* ptr = result, *ptr1 = result, tmp_char;
        int tmp_value;

        do {
                tmp_value = value;
                value /= base;
                *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
        } while ( value );

        if (tmp_value < 0)
                *ptr++ = '-';

        *ptr-- = '\0';

        while(ptr1 < ptr) {
                tmp_char = *ptr;
                *ptr--= *ptr1;
                *ptr1++ = tmp_char;
        }
        return result;

}

 
/* 
 *  * string copy using one gpu thread. 
 *   */ 
 
__device__ static char * gpuStrcpy(char * dst, const char * src){ 
 
    char * orig = dst; 
    while(*src) 
            *dst++ = *src++; 
    *dst = '\0'; 
 
    return orig; 
} 
 
__device__ static char* gpuStrncat(char *dest, const char *src, size_t n) 
{ 
    int dest_len = 0; 
    int i; 
 
    char * tmp = dest; 
    while(*tmp != '\0'){ 
        tmp++; 
        dest_len ++;
    }

    for (i = 0 ; i < n && src[i] != '\0' ; i++)
        dest[dest_len + i] = src[i];
    dest[dest_len + i] = '\0';
    return dest;
}


__device__ static char * gpuStrcat(char * dest, const char * src){
    char *tmp =dest;
    int dest_len = 0;
    int i;

    while (*tmp!= '\0'){
        tmp++ ;
        dest_len ++;
    }

    for(i=0; src[i] !='\0'; i++){
        dest[dest_len + i] = src[i];
    }

    dest[dest_len + i] = '\0';

    return dest;
}


#endif
