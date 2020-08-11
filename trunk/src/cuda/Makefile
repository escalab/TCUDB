
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
endif

NVCC            ?= $(CUDA_BIN_PATH)/nvcc

GENCODE_SM70    := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
GENCODE_FLAGS   := $(GENCODE_SM70) -O3

INCLUDES   := -I../include

ifeq ($(OS_SIZE),32)
    LDFLAGS   := -lcudart -lrt
else
    LDFLAGS   := -lcudart -lrt
endif

TCGemm: simpleTensorCoreGEMM.cu
	$(NVCC) -o TCGemm -arch=sm_70 -lcublas -lcurand simpleTensorCoreGEMM.cu

gpudb: gpu cpu
	 $(NVCC) -pg -O3 tableScan.o inviJoin.o hashJoin.o tcuJoin.o cuckoo.o materialize.o groupby.o orderBy.o driver.o $(LDFLAGS) -o GPUDATABASE -lcurand -lcublas

cpu: driver.cu 
	$(NVCC) -c -g $(GENCODE_FLAGS) driver.cu

gpu: tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DWMMA_HALF tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 
	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS_HALF tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 

all: gpudb

clean:
	rm -rf *.o GPUDATABASE cudaTensorCoreGemm cudaTensorCoreGemm.o 
