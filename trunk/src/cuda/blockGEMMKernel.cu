#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <pthread.h>
#include <unistd.h>
#include "../include/common.h"
#include "../include/blockJoin.h"
#include "msplitm.h"
#include "blockGEMMKernel.h"

#define NUM_SUBMATRIX 4UL
#define NUM_STREAMS 4UL

#ifdef BLOCK_HALF
void msplitm(char transa, char transb, 
	unsigned long m, unsigned long n, unsigned long k, 
	float alpha, 
	const float *A, int lda, 
	const float *B, int ldb, 
	float beta, 
	float *C, int ldc)
{
	cudaStream_t streams[NUM_STREAMS];

	float* b = NULL;
	float* b_h = NULL;
	
	float* a[NUM_STREAMS];
	float* a_h[NUM_STREAMS];

	float* c[NUM_STREAMS];
	float* c_h[NUM_STREAMS];

	cublasHandle_t handles[NUM_STREAMS];
    printf("entering msplitm \n");
    unsigned long A_sz = m * k;
    unsigned long B_sz = k * n;
    unsigned long MAX =  (unsigned long )m* (unsigned long) n / NUM_SUBMATRIX;
    
	MAX -= MAX % k;
	unsigned long numSubMatrixB = B_sz / MAX;
	unsigned long SMB_sz = B_sz / numSubMatrixB;
	unsigned long subCols = B_sz / (numSubMatrixB * k);
	unsigned long numSubMatrixA = A_sz / MAX;
	unsigned long SMA_sz = A_sz / numSubMatrixA;
	unsigned long subRows = A_sz / (numSubMatrixA * k);
	unsigned long overflowA = m % subRows;
	unsigned long overflowB = n % subCols;

	printf("MAX: %lu\n", MAX);
	printf("B_sz: %lu\n",B_sz);
	printf("SubmatriciesB: %lu\n", numSubMatrixB);
	printf("SMB_sz: %lu\n", SMB_sz);
	printf("subCols: %lu\n", subCols);
	printf("subrows: %lu\n", subRows);
	printf("SMA_sz: %lu\n", SMA_sz);
	printf("submatriciesA: %lu\n", numSubMatrixA);
	printf("overflowB: %lu\n", overflowB);
	printf("overflowA: %lu\n", overflowA);

	cudaMalloc((void**) &b, sizeof(float) * subCols * k);
	cudaMallocHost((void**) &b_h, sizeof(float)*subCols * k );

	for(int i = 0; i < NUM_STREAMS; ++i){
		cublasCreate(&handles[i]);
		cudaStreamCreate(&streams[i]);
		cudaMalloc((void**) &a[i], sizeof(float) * subRows * k);
		cudaMalloc((void**) &c[i], sizeof(float) * subCols * subRows);
		cudaMallocHost((void**) &a_h[i], sizeof(float) * subRows * k);
		cudaMallocHost((void**) &c_h[i], sizeof(float) * subCols * subRows);
	}

	// read the whole column stripe of B
	// upper bound is numSubMatrixB + 1 because we might have overflow case
	for(unsigned long i = 0; i < numSubMatrixB + 1; ++i){
		int count = 0;
		if(overflowB == 0 && i == numSubMatrixB){
			break;
		}
	
		// B data assignment, k rows * subCols columns
		memset(b_h, 0, sizeof(float) * subCols * k);
		for(int j = 0; j < k; ++j){
			if (i < numSubMatrixB) {
				memcpy(b_h + j * subCols, B + j * n + i * subCols, sizeof(float) * subCols);
			} else {
				memcpy(b_h + j * subCols, B + j * n + i * subCols, sizeof(float) * overflowB);
			}
		}
	
		cudaMemcpyAsync(b, b_h, sizeof(float)*subCols*k, cudaMemcpyHostToDevice, streams[0]);
		unsigned long streamsActive = 0;

		// read the whole row stripe of A
		// upper bound is numSubMatrixA + 1 because we might have overflow case
		for(unsigned long y = 0; y < numSubMatrixA + 1; ++y) {
			if(overflowA == 0 && y == numSubMatrixA){
				break;
			}

			// A data assignment, subRows rows * k columns
			memset(a_h[y % NUM_STREAMS], 0, sizeof(float) * subRows * k);
			if (y < numSubMatrixA) {
				for(int j = 0; j < subRows; ++j){
					memcpy((a_h[y % NUM_STREAMS]) + j * k, A + y*subRows*k + j*k, sizeof(float) * k);		
				}
			} else {
				for(int j = 0; j < overflowA; ++j){
					memcpy((a_h[y % NUM_STREAMS]) + j * k, A + y * subRows * k + j * k, sizeof(float) * k);		
				}
			}
			
			cudaMemcpyAsync(a[y % NUM_STREAMS], a_h[y % NUM_STREAMS], sizeof(float)*subRows*k, cudaMemcpyHostToDevice, streams[y % NUM_STREAMS]);
			printf("sending multiply %lu,%lu to stream %lu\n", y, i, y % NUM_STREAMS);
            
            // /*
			printf("perform cublasGemmEx w/ half type inputs\n");
			doHalfMultiply2MatricesStreaming(subRows, k, a[y % NUM_STREAMS], k, subCols, b, c[y % NUM_STREAMS], streams[y % NUM_STREAMS], handles[y % NUM_STREAMS], alpha); 	
            // */
			cudaMemcpyAsync(c_h[y % NUM_STREAMS], c[y % NUM_STREAMS], sizeof(float)*subRows*subCols, cudaMemcpyDeviceToHost, streams[y % NUM_STREAMS]);
						
			streamsActive++;
			if(y % NUM_STREAMS == NUM_STREAMS - 1){
				for(int s = 0; s < NUM_STREAMS; ++s){
					cudaStreamSynchronize(streams[s]);
					int currWork = count * NUM_STREAMS + s;
					if(i == numSubMatrixB && currWork == numSubMatrixA){
						copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, overflowB, beta);
					}else if(i == numSubMatrixB){
						copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, 0, overflowB, beta);
					}else if(currWork == numSubMatrixA){
						copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, 0, beta);
					}else{
						copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, 0, beta);
					}
					streamsActive--;
				}
				++count;
			}
		}

		// PrintMatrix("C", m, n, C);
		printf("%lu Streams Active Left over\n", streamsActive);
		for(int s = 0; s < streamsActive; ++s){
			cudaStreamSynchronize(streams[s]);
			int currWork = count * NUM_STREAMS + s;
			if(i == numSubMatrixB && currWork == numSubMatrixA){
				copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, overflowB, beta);
			}else if(i == numSubMatrixB){
				copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, 0, overflowB, beta);
			}else if(currWork == numSubMatrixA){
				copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, 0, beta);
			}else{
				copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, 0, beta);
			}
		}
	}

	for(int i = 0; i < NUM_STREAMS; ++i){
		cudaFree(a[i]);
		cudaFree(c[i]);
		cudaFreeHost(a_h[i]);
		cudaFreeHost(c_h[i]);
		cudaStreamDestroy(streams[i]);
	}
	cudaFree(b);
	cudaFreeHost(b_h);
}
#else

void msplitm(char transa, char transb, 
	unsigned long m, unsigned long n, unsigned long k, 
	float alpha, 
	const float *A, int lda, 
	const float *B, int ldb, 
	float beta, 
	float *C, int ldc)
{
    cudaStream_t streams[NUM_STREAMS];

	float* b = NULL;
	float* b_h = NULL;
	
	float* a[NUM_STREAMS];
	float* a_h[NUM_STREAMS];

	float* c[NUM_STREAMS];
	float* c_h[NUM_STREAMS];

	cublasHandle_t handles[NUM_STREAMS];
    printf("entering msplitm \n");
    unsigned long A_sz = m * k;
    unsigned long B_sz = k * n;
    unsigned long MAX =  (unsigned long )m* (unsigned long) n / NUM_SUBMATRIX;
    
	MAX -= MAX % k;
	unsigned long numSubMatrixB = B_sz / MAX;
	unsigned long SMB_sz = B_sz / numSubMatrixB;
	unsigned long subCols = B_sz / (numSubMatrixB * k);
	unsigned long numSubMatrixA = A_sz / MAX;
	unsigned long SMA_sz = A_sz / numSubMatrixA;
	unsigned long subRows = A_sz / (numSubMatrixA * k);
	unsigned long overflowA = m % subRows;
	unsigned long overflowB = n % subCols;

	printf("MAX: %lu\n", MAX);
	printf("B_sz: %lu\n",B_sz);
	printf("SubmatriciesB: %lu\n", numSubMatrixB);
	printf("SMB_sz: %lu\n", SMB_sz);
	printf("subCols: %lu\n", subCols);
	printf("subrows: %lu\n", subRows);
	printf("SMA_sz: %lu\n", SMA_sz);
	printf("submatriciesA: %lu\n", numSubMatrixA);
	printf("overflowB: %lu\n", overflowB);
	printf("overflowA: %lu\n", overflowA);

	cudaMalloc((void**) &b, sizeof(float) * subCols * k);
	cudaMallocHost((void**) &b_h, sizeof(float)*subCols * k );

	for(int i = 0; i < NUM_STREAMS; ++i){
		cublasCreate(&handles[i]);
		cudaStreamCreate(&streams[i]);
		cudaMalloc((void**) &a[i], sizeof(float) * subRows * k);
		cudaMalloc((void**) &c[i], sizeof(float) * subCols * subRows);
		cudaMallocHost((void**) &a_h[i], sizeof(float) * subRows * k);
		cudaMallocHost((void**) &c_h[i], sizeof(float) * subCols * subRows);
	}

	// read the whole column stripe of B
	// upper bound is numSubMatrixB + 1 because we might have overflow case
	for(unsigned long i = 0; i < numSubMatrixB + 1; ++i){
		int count = 0;
		if(overflowB == 0 && i == numSubMatrixB){
			break;
		}
	
		// B data assignment, k rows * subCols columns
		memset(b_h, 0, sizeof(float) * subCols * k);
		for(int j = 0; j < k; ++j){
			if (i < numSubMatrixB) {
				memcpy(b_h + j * subCols, B + j * n + i * subCols, sizeof(float) * subCols);
			} else {
				memcpy(b_h + j * subCols, B + j * n + i * subCols, sizeof(float) * overflowB);
			}
		}
	
		cudaMemcpyAsync(b, b_h, sizeof(float)*subCols*k, cudaMemcpyHostToDevice, streams[0]);
		unsigned long streamsActive = 0;

		// read the whole row stripe of A
		// upper bound is numSubMatrixA + 1 because we might have overflow case
		for(unsigned long y = 0; y < numSubMatrixA + 1; ++y) {
			if(overflowA == 0 && y == numSubMatrixA){
				break;
			}

			// A data assignment, subRows rows * k columns
			memset(a_h[y % NUM_STREAMS], 0, sizeof(float) * subRows * k);
			if (y < numSubMatrixA) {
				for(int j = 0; j < subRows; ++j){
					memcpy((a_h[y % NUM_STREAMS]) + j * k, A + y*subRows*k + j*k, sizeof(float) * k);		
				}
			} else {
				for(int j = 0; j < overflowA; ++j){
					memcpy((a_h[y % NUM_STREAMS]) + j * k, A + y * subRows * k + j * k, sizeof(float) * k);		
				}
			}
			
			cudaMemcpyAsync(a[y % NUM_STREAMS], a_h[y % NUM_STREAMS], sizeof(float)*subRows*k, cudaMemcpyHostToDevice, streams[y % NUM_STREAMS]);
			printf("sending multiply %lu,%lu to stream %lu\n", y, i, y % NUM_STREAMS);
            

			// printf("perform cublasGemmEx FAST_16F\n");
            
            // doMMStreaming(subRows, k, a[y % NUM_STREAMS], k,
            //     subCols, b, c[y % NUM_STREAMS], streams[y % NUM_STREAMS],
            //     handles[y % NUM_STREAMS]);


			printf("perform cublasSgemm\n");
			doMultiply2MatricesStreaming(subRows, k, a[y % NUM_STREAMS], k, subCols, b, c[y % NUM_STREAMS], streams[y % NUM_STREAMS], handles[y % NUM_STREAMS], alpha); 	

			cudaMemcpyAsync(c_h[y % NUM_STREAMS], c[y % NUM_STREAMS], sizeof(float)*subRows*subCols, cudaMemcpyDeviceToHost, streams[y % NUM_STREAMS]);
						
			streamsActive++;
			if(y % NUM_STREAMS == NUM_STREAMS - 1){
				for(int s = 0; s < NUM_STREAMS; ++s){
					cudaStreamSynchronize(streams[s]);
					int currWork = count * NUM_STREAMS + s;
					if(i == numSubMatrixB && currWork == numSubMatrixA){
						copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, overflowB, beta);
					}else if(i == numSubMatrixB){
						copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, 0, overflowB, beta);
					}else if(currWork == numSubMatrixA){
						copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, 0, beta);
					}else{
						copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, 0, beta);
					}
					streamsActive--;
				}
				++count;
			}
		}

		// PrintMatrix("C", m, n, C);
		printf("%lu Streams Active Left over\n", streamsActive);
		for(int s = 0; s < streamsActive; ++s){
			cudaStreamSynchronize(streams[s]);
			int currWork = count * NUM_STREAMS + s;
			if(i == numSubMatrixB && currWork == numSubMatrixA){
				copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, overflowB, beta);
			}else if(i == numSubMatrixB){
				copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, 0, overflowB, beta);
			}else if(currWork == numSubMatrixA){
				copyElements(C,  c_h[s], subRows, subCols, m, n, currWork, i, overflowA, 0, beta);
			}else{
				copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, 0, beta);
			}
		}
	}

	for(int i = 0; i < NUM_STREAMS; ++i){
		cudaFree(a[i]);
		cudaFree(c[i]);
		cudaFreeHost(a_h[i]);
		cudaFreeHost(c_h[i]);
		cudaStreamDestroy(streams[i]);
	}
	cudaFree(b);
	cudaFreeHost(b_h);
}

#endif