#ifndef SCAN_IMPL_CU
#define SCAN_IMPL_CU

#include "scan.cu"
#include "../include/common.h"

static void scanImpl(int *d_input, int rLen, int *d_output, struct statistic * pp)
{
	int len = 2;
	if(rLen < len){
		int *input, *output;
		cudaMalloc((void**)&input,len*sizeof(int));
		cudaMalloc((void**)&output, len*sizeof(int));
		cudaMemset(input, 0, len*sizeof(int));
		cudaMemcpy(input, d_input, rLen*sizeof(int), cudaMemcpyDeviceToDevice);
		preallocBlockSums(len);
		prescanArray(output, input, len, pp);
		deallocBlockSums();
		cudaMemcpy(d_output,output,rLen*sizeof(int),cudaMemcpyDeviceToDevice);
		cudaFree(input);
		cudaFree(output);
		return;
	}else{
		preallocBlockSums(rLen);
		prescanArray(d_output, d_input, rLen, pp);
		deallocBlockSums();
	}
}


#endif

