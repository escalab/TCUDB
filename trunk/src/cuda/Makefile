OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")

CUDA_HOME       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_HOME)/include
CUDA_BIN_PATH   ?= $(CUDA_HOME)/bin
CUDA_LIB		:= -lcublas_static -lculibos -lcudart_static -ldl -lcublasLt_static 

ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
endif

NVCC            ?= $(CUDA_BIN_PATH)/nvcc

GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_FLAGS   := $(GENCODE_SM80)

INCLUDES   := -I../include
CFLAGS	:= -g -O3
NVCCFLAGS	:= -pg -O3

ifeq ($(OS_SIZE),32)
    LDFLAGS   := -lcudart -lrt
else
    LDFLAGS   := -lcudart -lrt
endif

tcudb: tcu cpu
	 $(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) tableScan.o tcuJoin.o driver.o $(LDFLAGS) -o TCUDB $(CUDA_LIB)

gpudb: gpu cpu
	 $(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) tableScan.o inviJoin.o hashJoin.o cuckoo.o materialize.o groupby.o orderBy.o driver.o $(LDFLAGS) -o GPUDATABASE

cpu: driver.cu 
	$(NVCC) -c $(CFLAGS) $(GENCODE_FLAGS) driver.cu

gpu: tableScan.cu hashJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu
	$(NVCC) -c -g $(GENCODE_FLAGS) tableScan.cu hashJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 

tcu: tableScan.cu tcuJoin.cu cuckoo.cu
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DWMMA_HALF tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 
	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS_HALF -DMICRO tableScan.cu tcuJoin.cu
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS_HALF -DITUNES -DRED tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS_HALF -DPAGERANK -DTHREAD tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS_HALF -DRED tableScan.cu tcuJoin.cu 
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS_HALF tableScan.cu tcuJoin.cu 
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS_HALF -DOUTDEGREE tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 
#	$(NVCC) -c -g $(GENCODE_FLAGS) -DCUBLAS -DMICRO tableScan.cu hashJoin.cu tcuJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 

all: tcudb

clean:
	rm -rf *.o GPUDATABASE TCUDB
