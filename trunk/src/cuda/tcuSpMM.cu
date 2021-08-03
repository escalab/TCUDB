#include <cusparse.h>
#include <stdio.h>

#include "tcuSpMM.h"

#define MAX_THREADS 1024
#define BILLION 1000000000
#define MILLION 1000000


#define cusparseCheck(stat) { cusparseCheck_((stat), __FILE__, __LINE__); }
void cusparseCheck_(cusparseStatus_t stat, const char *file, int line) {
    if (stat != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE API failed: %d : %s %s %d\n", stat,
                cusparseGetErrorString(stat), file, line);
    }
}


void printIndices(int *mat, int sz)
{
    for (int i = 0; i < sz; i++) {
        printf("%d\t", mat[i]);
    }
    printf("\n");
}

void printValues(float *mat, int sz)
{
    for (int i = 0; i < sz; i++) {
        printf("%f\t", mat[i]);
    }
    printf("\n");
}

int groupByRows(int *rowOffsets, int sz)
{
    int res = 0;
    for (int i = 1; i < sz; i++) {
        if (rowOffsets[i] != rowOffsets[i-1]) {
            res++;
        }
    }
    return res;
}

int groupByColumns(int *columns, int sz, int columnWidth)
{
    int res = 0;
    int *lookup = (int*)calloc(columnWidth, sizeof(int));
    for (int i = 0; i < sz; i++) {
        if (lookup[columns[i]] == 0) {
            lookup[columns[i]] = 1;
            res++;
        }
    }
    return res; 
}

/* num_rows = left_gbWidth, not orig leftTupleNum */
void tbl2csr_gbA(int tupleNum, char *joinKey, char *Xdata,
                int *csrOffsets, int *csrColumns, float *csrValues,
                int num_rows, int fillOne, char *gbColumn,
                int num_cols, int &nnz)
{
    int *lookup = (int *)malloc(num_rows * num_cols * sizeof(int));
    memset(lookup, -1, num_rows * num_cols * sizeof(int));

    int *num_elems_per_row = (int*)calloc(num_rows, sizeof(int));

    // count num of elements per row
    for (int i = 0; i< tupleNum; i++) {
        // row index is determined by groupBy value
        int *key = (int*)&joinKey[i * sizeof(int)];
        int *gbVal = (int*)&gbColumn[i * sizeof(int)];
        if (lookup[(*gbVal)*num_cols + (*key)] == -1) {
            lookup[(*gbVal)*num_cols + (*key)] = 1;
            num_elems_per_row[*gbVal]++;
        } 
        else {
            nnz--;
        }
    }

    // prefixsum
    for (int i = 0; i < num_rows; i++) {
        csrOffsets[i+1] = num_elems_per_row[i] + csrOffsets[i];
    }

    memset(lookup, -1, num_rows * num_cols * sizeof(int));
    for (int i = 0; i < tupleNum; i++) {
        int *key = (int*)&joinKey[i * sizeof(int)];
        int *gbVal = (int*)&gbColumn[i * sizeof(int)];
        if (lookup[(*gbVal)*num_cols + (*key)] == -1) {
            lookup[(*gbVal)*num_cols + (*key)] = 1;
            num_elems_per_row[*gbVal]--;
            int offset = csrOffsets[*gbVal] + num_elems_per_row[*gbVal];
            csrColumns[offset] = *key;

            if (fillOne) {
                csrValues[i] = 1.0f;
            } else {
                int *data = (int*)&Xdata[i * sizeof(int)];
                csrValues[offset] = (float)*data;
            }
        }
    }
}

void tbl2csr_transpose_gbB(int tupleNum, char *joinKey, char *Xdata,
                           int *csrOffsets, int *csrColumns, float *csrValues,
                           int num_rows, int fillOne, char *gbColumn,
                           int num_cols, int &nnz)
{
    // num_rows = MATRIX_K
    // num_cols = right_gbWidth
    int *lookup = (int *)malloc(num_rows * num_cols * sizeof(int));
    memset(lookup, -1, num_rows * num_cols * sizeof(int));

    int *num_elems_per_row = (int*)calloc(num_rows, sizeof(int));

    // count num of elements per row
    for (int i = 0; i< tupleNum; i++) {
        // row index is determined by join key
        int *key = (int*)&joinKey[i * sizeof(int)];
        int *gbVal = (int*)&gbColumn[i * sizeof(int)];
        if (lookup[(*key)*num_cols + (*gbVal)] == -1) {
            lookup[(*key)*num_cols + (*gbVal)] = 1;
            num_elems_per_row[*key]++;
        } else {
            nnz--;
        }
    }

    // prefixsum
    for (int i = 0; i < num_rows; i++) {
        csrOffsets[i+1] = num_elems_per_row[i] + csrOffsets[i];
    }

    memset(lookup, -1, num_rows * num_cols * sizeof(int));
    for (int i = 0; i < tupleNum; i++) {
        int *key = (int*)&joinKey[i * sizeof(int)];
        int *gbVal = (int*)&gbColumn[i * sizeof(int)];
        if (lookup[(*key)*num_cols + (*gbVal)] == -1) {
            lookup[(*key)*num_cols + (*gbVal)] = 1;
            num_elems_per_row[*key]--;
            int offset = csrOffsets[*key] + num_elems_per_row[*key];
            // col index is determined by groupBy value
            csrColumns[offset] = *gbVal;

            if (fillOne) {
                csrValues[i] = 1.0f;
            } else {
                int *data = (int*)&Xdata[i * sizeof(int)];
                csrValues[offset] = (float)*data;
            }
        }
    }
}

/* Prepare CSR format in transpose from table entries. */
void tbl2csr_transpose(int tupleNum, char *joinKey, char *Xdata,
                       int *csrOffsets, int *csrColumns, float *csrValues,
                       int num_rows, int fillOne)
{
    int *num_elems_per_row = (int*)calloc(num_rows, sizeof(int));

    // count num of elements per row
    for (int i = 0; i< tupleNum; i++) {
        int *key = (int*)&joinKey[i * sizeof(int)];
        num_elems_per_row[*key]++;
    }

    // prefixsum
    for (int i = 0; i < num_rows; i++) {
        csrOffsets[i+1] = num_elems_per_row[i] + csrOffsets[i];
    }

    for (int i = 0; i < tupleNum; i++) {
        int *key = (int*)&joinKey[i * sizeof(int)];
        num_elems_per_row[*key]--;
        int offset = csrOffsets[*key] + num_elems_per_row[*key];
        csrColumns[offset] = i;

        if (fillOne) {
            csrValues[i] = 1.0f;
        } else {
            //float *data = (float*)&Xdata[i * sizeof(float)];
            int *data = (int*)&Xdata[i * sizeof(int)];
            //printf("B val: %f\n", *data);
            csrValues[offset] = (float)*data;
        }
    }
}

__global__ void gpu_tbl2csr(int tupleNum, char *joinKey, char *Xdata,
                            int *csrOffsets, int *csrColumns, float *csrValues,
                            int fillOne) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= tupleNum) return;

    int *key = (int*)&joinKey[i*sizeof(int)];
    csrOffsets[i+1] = i+1;
    csrColumns[i] = *key;

    if (fillOne) {
        csrValues[i] = 1.0f;
    } else {
        //float *data = (float*)&Xdata[i * sizeof(float)];
        int *data = (int*)&Xdata[i * sizeof(int)];
        csrValues[i] = (float)*data;
    }

}

/* For the case groupBy A and B, groupBy count is the join count. */
void tcuspmm_gbAB(int Annz, int A_num_rows, int A_num_cols,
                int Bnnz, int B_num_rows, int B_num_cols,
                int MATRIX_K,
                int leftTupleNum, char *fact, char *ldata,
                int left_gbWidth, char *gbAColumn,
                int right_gbWidth, char *gbBColumn,
                int rightTupleNum, char *dim, char *rdata)
{
    struct timespec fill_start, fill_end;
    struct timespec cusp_start, cusp_end;
    struct timespec gb_start, gb_end;

    clock_gettime(CLOCK_REALTIME, &fill_start);
    float alpha = 1.0f, beta = 0.0f;

    A_num_rows = left_gbWidth;
    B_num_cols = right_gbWidth;

    int *hA_csrOffsets = (int*)calloc((A_num_rows + 1), sizeof(int));
    int *hA_csrColumns = (int*)calloc(Annz, sizeof(int));
    float *hA_csrValues  = (float*)calloc(Annz, sizeof(float));
    int *hB_csrOffsets = (int*)calloc((B_num_rows + 1), sizeof(int));
    int *hB_csrColumns = (int*)calloc(Bnnz, sizeof(int));
    float *hB_csrValues  = (float*)calloc(Bnnz, sizeof(float));

    tbl2csr_gbA(leftTupleNum, fact, ldata,
                hA_csrOffsets, hA_csrColumns, hA_csrValues,
                A_num_rows, 0, gbAColumn,
                MATRIX_K, Annz);
#ifdef DEBUG
    printIndices(hA_csrOffsets,(A_num_rows + 1));
    printIndices(hA_csrColumns, Annz);
    printValues(hA_csrValues, Annz);
#endif

    tbl2csr_transpose_gbB(rightTupleNum, dim, rdata,
                          hB_csrOffsets, hB_csrColumns, hB_csrValues,
                          MATRIX_K, 1, gbBColumn,
                          B_num_cols, Bnnz);
#ifdef DEBUG
    printIndices(hB_csrOffsets,(B_num_rows + 1));
    printIndices(hB_csrColumns, Bnnz);
    printValues(hB_csrValues, Bnnz);
#endif

    printf("Annz: %d A_num_rows: %d A_num_cols: %d\n", Annz, A_num_rows, 
            A_num_cols);
    printf("Bnnz: %d B_num_rows: %d B_num_cols: %d\n", Bnnz, B_num_rows, 
            B_num_cols);

    cusparseOperation_t opA  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    cudaEvent_t startcusparse;
    cudaEvent_t stopcusparse;

    cudaEventCreate(&startcusparse);
    cudaEventCreate(&stopcusparse);

    // Dev memory -- allocate and copy Amat, Bmat
    int *dA_rows, *dA_columns;
    int *dB_rows, *dB_columns;
    int *dC_rows, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    int *dA_csrOffsets, *dA_csrColumns;
    int *dB_csrOffsets, *dB_csrColumns;
    float *dA_csrValues, *dB_csrValues;
    int *dC_csrOffsets, *dC_csrColumns;

    cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dA_csrColumns, Annz * sizeof(int));
    cudaMalloc((void**) &dA_csrValues,  Annz * sizeof(float));

    cudaMalloc((void**) &dB_csrOffsets, (B_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dB_csrColumns, Bnnz * sizeof(int));
    cudaMalloc((void**) &dB_csrValues,  Bnnz * sizeof(float));

    cudaMalloc((void**) &dC_csrOffsets, (A_num_rows + 1) * sizeof(int));

    cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrColumns, hA_csrColumns, Annz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrValues, hA_csrValues, Annz * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrOffsets, hB_csrOffsets, (B_num_rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrColumns, hB_csrColumns, Bnnz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrValues, hB_csrValues, Bnnz * sizeof(float),
               cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_REALTIME, &fill_end);

    // call CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseCreate(&handle);
    
    clock_gettime(CLOCK_REALTIME, &cusp_start);
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;

    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, Annz,
                      dA_csrOffsets, dA_csrColumns, dA_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, B_num_rows, B_num_cols, Bnnz,
                      dB_csrOffsets, dB_csrColumns, dB_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    clock_gettime(CLOCK_REALTIME, &cusp_end);

    cudaEventRecord(startcusparse, 0);
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void**) &dBuffer1, bufferSize1);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);

    cudaMalloc((void**) &dBuffer2, bufferSize2);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    cudaEventRecord(stopcusparse, 0);

    clock_gettime(CLOCK_REALTIME, &gb_start);
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
    cudaMalloc((void**) &dC_csrColumns, C_nnz1 * sizeof(int));
    cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float));
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_csrColumns, dC_values);

    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    // device result check
    int   hC_csrOffsets_tmp[A_num_rows + 1];
    cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
               (A_num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    int groupByCount = 0;
    groupByCount = C_nnz1;
//    groupByCount = groupByRows(hC_csrOffsets_tmp, (A_num_rows + 1));
    clock_gettime(CLOCK_REALTIME, &gb_end);

    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);

    float cusparseSpGEMM_time;
    cudaEventElapsedTime(&cusparseSpGEMM_time, startcusparse, stopcusparse);
    double data_preparation = (fill_end.tv_sec-fill_start.tv_sec) * BILLION + 
        fill_end.tv_nsec - fill_start.tv_nsec;
    double cusp_preparation = (cusp_end.tv_sec-cusp_start.tv_sec) * BILLION + 
        cusp_end.tv_nsec - cusp_start.tv_nsec;
    double gb_elapse = (gb_end.tv_sec-gb_start.tv_sec) * BILLION + 
        gb_end.tv_nsec - gb_start.tv_nsec;
    printf("Join counts: %d\n", C_nnz1);
    printf("GroupBy counts: %d\n", groupByCount);
    printf("Data preparation time: %lf ms\n", 
            data_preparation / MILLION);
    printf("cusp init time: %lf ms\n", cusp_preparation / MILLION);
    printf("cusparseSpGEMM took %f ms\n", cusparseSpGEMM_time);
    printf("groupBy time: %lf ms\n", gb_elapse / MILLION);
}

/* groupByColumns */
void tcuspmm_gbB(int Annz, int A_num_rows, int A_num_cols,
                int Bnnz, int B_num_rows, int B_num_cols,
                int MATRIX_K, int foreignKeySize,
                int leftTupleNum, char *gpu_fact, char *ldata,
                int right_gbWidth, char *gbColumn,
                int rightTupleNum, char *dim, char *rdata)
{
    struct timespec fill_start, fill_end;
    struct timespec cusp_start, cusp_end;
    struct timespec gb_start, gb_end;

    clock_gettime(CLOCK_REALTIME, &fill_start);
    float alpha = 1.0f, beta = 0.0f;
    B_num_cols = right_gbWidth;

    /*
    int *hA_csrOffsets = (int*)calloc((A_num_rows + 1), sizeof(int));
    int *hA_csrColumns = (int*)calloc(Annz, sizeof(int));
    float *hA_csrValues  = (float*)calloc(Annz, sizeof(float));
    */
    int *hB_csrOffsets = (int*)calloc((B_num_rows + 1), sizeof(int));
    int *hB_csrColumns = (int*)calloc(Bnnz, sizeof(int));
    float *hB_csrValues  = (float*)calloc(Bnnz, sizeof(float));

    /*
    tbl2csr_gbA(leftTupleNum, fact, ldata,
                hA_csrOffsets, hA_csrColumns, hA_csrValues,
                left_gbWidth, 0, gbColumn,
                MATRIX_K, Annz);
    */
    tbl2csr_transpose_gbB(rightTupleNum, dim, rdata,
                          hB_csrOffsets, hB_csrColumns, hB_csrValues,
                          MATRIX_K, 1, gbColumn,
                          B_num_cols, Bnnz);

//    printf("tbl2csr_transpose finished\n");

    printf("Annz: %d A_num_rows: %d A_num_cols: %d\n", Annz, A_num_rows, 
            A_num_cols);
    printf("Bnnz: %d B_num_rows: %d B_num_cols: %d\n", Bnnz, B_num_rows, 
            B_num_cols);

    cusparseOperation_t opA  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    cudaEvent_t startcusparse;
    cudaEvent_t stopcusparse;

    cudaEventCreate(&startcusparse);
    cudaEventCreate(&stopcusparse);
    
    // Dev memory -- allocate and copy Amat, Bmat
    int *dA_rows, *dA_columns;
    int *dB_rows, *dB_columns;
    int *dC_rows, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    int *dA_csrOffsets, *dA_csrColumns;
    int *dB_csrOffsets, *dB_csrColumns;
    float *dA_csrValues, *dB_csrValues;
    int *dC_csrOffsets, *dC_csrColumns;

    char *gpu_ldata;

    // allocate
    cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dA_csrColumns, Annz * sizeof(int));
    cudaMalloc((void**) &dA_csrValues,  Annz * sizeof(float));
    
    cudaMalloc((void**) &gpu_ldata, foreignKeySize);

    cudaMalloc((void**) &dB_csrOffsets, (B_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dB_csrColumns, Bnnz * sizeof(int));
    cudaMalloc((void**) &dB_csrValues,  Bnnz * sizeof(float));

    cudaMalloc((void**) &dC_csrOffsets, (A_num_rows + 1) * sizeof(int));

#ifdef DEBUG 
    printIndices(hB_csrOffsets,(B_num_rows + 1));
    printIndices(hB_csrColumns, Bnnz);
    printValues(hB_csrValues, Bnnz);
#endif
    
    cudaMemcpy(gpu_ldata, ldata, foreignKeySize, cudaMemcpyHostToDevice);

    cudaMemcpy(dB_csrOffsets, hB_csrOffsets, (B_num_rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrColumns, hB_csrColumns, Bnnz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrValues, hB_csrValues, Bnnz * sizeof(float),
               cudaMemcpyHostToDevice);

    gpu_tbl2csr<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS, MAX_THREADS>>> (
            leftTupleNum, gpu_fact, gpu_ldata,
            dA_csrOffsets, dA_csrColumns, dA_csrValues, 0); 

    clock_gettime(CLOCK_REALTIME, &fill_end);

    // call CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseCreate(&handle);
    
    clock_gettime(CLOCK_REALTIME, &cusp_start);
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;

    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, Annz,
                      dA_csrOffsets, dA_csrColumns, dA_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, B_num_rows, B_num_cols, Bnnz,
                      dB_csrOffsets, dB_csrColumns, dB_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    clock_gettime(CLOCK_REALTIME, &cusp_end);

    cudaEventRecord(startcusparse, 0);
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void**) &dBuffer1, bufferSize1);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);

    cudaMalloc((void**) &dBuffer2, bufferSize2);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    cudaEventRecord(stopcusparse, 0);

    clock_gettime(CLOCK_REALTIME, &gb_start);
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
    cudaMalloc((void**) &dC_csrColumns, C_nnz1 * sizeof(int));
    cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float));
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_csrColumns, dC_values);

    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    // device result check
    int *hC_columns_tmp = (int*)calloc(C_nnz1, sizeof(int));
    int groupByCount = 0;
    cudaMemcpy(hC_columns_tmp, dC_csrColumns, C_nnz1 * sizeof(int),
               cudaMemcpyDeviceToHost);
    groupByCount = groupByColumns(hC_columns_tmp, C_nnz1, C_num_cols1);
    /*
    for (int i = 0; i < C_nnz1; i++) {
        printf("hC_columns_tmp[%d]: %d\n", i, hC_columns_tmp[i]);
    }
    */
    clock_gettime(CLOCK_REALTIME, &gb_end);

    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);

    float cusparseSpGEMM_time;
    cudaEventElapsedTime(&cusparseSpGEMM_time, startcusparse, stopcusparse);
    double data_preparation = (fill_end.tv_sec-fill_start.tv_sec) * BILLION + 
        fill_end.tv_nsec - fill_start.tv_nsec;
    double cusp_preparation = (cusp_end.tv_sec-cusp_start.tv_sec) * BILLION + 
        cusp_end.tv_nsec - cusp_start.tv_nsec;
    double gb_elapse = (gb_end.tv_sec-gb_start.tv_sec) * BILLION + 
        gb_end.tv_nsec - gb_start.tv_nsec;
    printf("Join counts: %d\n", C_nnz1);
    printf("GroupBy counts: %d\n", groupByCount);
    printf("Data preparation time: %lf ms\n", 
            data_preparation / MILLION);
    printf("cusp init time: %lf ms\n", cusp_preparation / MILLION);
    printf("cusparseSpGEMM took %f ms\n", cusparseSpGEMM_time);
    printf("groupBy time: %lf ms\n", gb_elapse / MILLION);
}

/* groupByRows */
void tcuspmm_gbA(int Annz, int A_num_rows, int A_num_cols,
                int Bnnz, int B_num_rows, int B_num_cols,
                int MATRIX_K,
                int leftTupleNum, char *fact, char *ldata,
                int left_gbWidth, char *gbColumn,
                int rightTupleNum, char *dim, char *rdata)
{
    struct timespec fill_start, fill_end;
    struct timespec cusp_start, cusp_end;
    struct timespec gb_start, gb_end;

    clock_gettime(CLOCK_REALTIME, &fill_start);
    float alpha = 1.0f, beta = 0.0f;
    A_num_rows = left_gbWidth;

    int *hA_csrOffsets = (int*)calloc((A_num_rows + 1), sizeof(int));
    int *hA_csrColumns = (int*)calloc(Annz, sizeof(int));
    float *hA_csrValues  = (float*)calloc(Annz, sizeof(float));
    int *hB_csrOffsets = (int*)calloc((B_num_rows + 1), sizeof(int));
    int *hB_csrColumns = (int*)calloc(Bnnz, sizeof(int));
    float *hB_csrValues  = (float*)calloc(Bnnz, sizeof(float));

    tbl2csr_gbA(leftTupleNum, fact, ldata,
                hA_csrOffsets, hA_csrColumns, hA_csrValues,
                left_gbWidth, 0, gbColumn,
                MATRIX_K, Annz);
    tbl2csr_transpose(rightTupleNum, dim, rdata,
                      hB_csrOffsets, hB_csrColumns, hB_csrValues,
                      MATRIX_K, 1);

    printf("Annz: %d A_num_rows: %d A_num_cols: %d\n", Annz, A_num_rows, 
            A_num_cols);
    printf("Bnnz: %d B_num_rows: %d B_num_cols: %d\n", Bnnz, B_num_rows, 
            B_num_cols);

    cusparseOperation_t opA  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    cudaEvent_t startcusparse;
    cudaEvent_t stopcusparse;

    cudaEventCreate(&startcusparse);
    cudaEventCreate(&stopcusparse);
    
    // Dev memory -- allocate and copy Amat, Bmat
    int *dA_rows, *dA_columns;
    int *dB_rows, *dB_columns;
    int *dC_rows, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    int *dA_csrOffsets, *dA_csrColumns;
    int *dB_csrOffsets, *dB_csrColumns;
    float *dA_csrValues, *dB_csrValues;
    int *dC_csrOffsets, *dC_csrColumns;

    // allocate
    cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dA_csrColumns, Annz * sizeof(int));
    cudaMalloc((void**) &dA_csrValues,  Annz * sizeof(float));

    cudaMalloc((void**) &dB_csrOffsets, (B_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dB_csrColumns, Bnnz * sizeof(int));
    cudaMalloc((void**) &dB_csrValues,  Bnnz * sizeof(float));

    cudaMalloc((void**) &dC_csrOffsets, (A_num_rows + 1) * sizeof(int));

#ifdef DEBUG
    printIndices(hA_csrOffsets,(A_num_rows + 1));
    printIndices(hA_csrColumns, Annz);
    printValues(hA_csrValues, Annz);
#endif

    cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrColumns, hA_csrColumns, Annz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrValues, hA_csrValues, Annz * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrOffsets, hB_csrOffsets, (B_num_rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrColumns, hB_csrColumns, Bnnz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrValues, hB_csrValues, Bnnz * sizeof(float),
               cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_REALTIME, &fill_end);

    // call CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseCreate(&handle);
    
    clock_gettime(CLOCK_REALTIME, &cusp_start);
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;

    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, Annz,
                      dA_csrOffsets, dA_csrColumns, dA_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, B_num_rows, B_num_cols, Bnnz,
                      dB_csrOffsets, dB_csrColumns, dB_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    clock_gettime(CLOCK_REALTIME, &cusp_end);

    cudaEventRecord(startcusparse, 0);
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void**) &dBuffer1, bufferSize1);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);

    cudaMalloc((void**) &dBuffer2, bufferSize2);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    cudaEventRecord(stopcusparse, 0);

    clock_gettime(CLOCK_REALTIME, &gb_start);
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
    cudaMalloc((void**) &dC_csrColumns, C_nnz1 * sizeof(int));
    cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float));
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_csrColumns, dC_values);

    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    // device result check
    int   hC_csrOffsets_tmp[A_num_rows + 1];
    cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
               (A_num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    int groupByCount = 0;
    groupByCount = groupByRows(hC_csrOffsets_tmp, (A_num_rows + 1));
    clock_gettime(CLOCK_REALTIME, &gb_end);

    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);

    float cusparseSpGEMM_time;
    cudaEventElapsedTime(&cusparseSpGEMM_time, startcusparse, stopcusparse);
    double data_preparation = (fill_end.tv_sec-fill_start.tv_sec) * BILLION + 
        fill_end.tv_nsec - fill_start.tv_nsec;
    double cusp_preparation = (cusp_end.tv_sec-cusp_start.tv_sec) * BILLION + 
        cusp_end.tv_nsec - cusp_start.tv_nsec;
    double gb_elapse = (gb_end.tv_sec-gb_start.tv_sec) * BILLION + 
        gb_end.tv_nsec - gb_start.tv_nsec;
    printf("Join counts: %d\n", C_nnz1);
    printf("GroupBy counts: %d\n", groupByCount);
    printf("Data preparation time: %lf ms\n", 
            data_preparation / MILLION);
    printf("cusp init time: %lf ms\n", cusp_preparation / MILLION);
    printf("cusparseSpGEMM took %f ms\n", cusparseSpGEMM_time);
    printf("groupBy time: %lf ms\n", gb_elapse / MILLION);
}

/* Amat key requires to cudaMemcpy first since filling Amat is done on GPU. */
void tcuspmm(int Annz, int A_num_rows, int A_num_cols,
             int Bnnz, int B_num_rows, int B_num_cols,
             int MATRIX_K, int foreignKeySize,
             int leftTupleNum, char *gpu_fact, char *ldata,
             int rightTupleNum, char *dim, char *rdata)
{
    //cudaDeviceReset();
    struct timespec fill_start, fill_end;
    struct timespec cusp_start, cusp_end;

    clock_gettime(CLOCK_REALTIME, &fill_start);
    float alpha = 1.0f, beta = 0.0f;
    int *hB_csrOffsets = (int*)calloc((B_num_rows + 1), sizeof(int));
    int *hB_csrColumns = (int*)calloc(Bnnz, sizeof(int));
    float *hB_csrValues  = (float*)calloc(Bnnz, sizeof(float));

    tbl2csr_transpose(rightTupleNum, dim, rdata,
                      hB_csrOffsets, hB_csrColumns, hB_csrValues,
                      MATRIX_K, 1);

    printf("Annz: %d A_num_rows: %d A_num_cols: %d\n", Annz, A_num_rows, 
            A_num_cols);
    printf("Bnnz: %d B_num_rows: %d B_num_cols: %d\n", Bnnz, B_num_rows, 
            B_num_cols);
                       
    cusparseOperation_t opA  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    cudaEvent_t startcusparse;
    cudaEvent_t stopcusparse;

    cudaEventCreate(&startcusparse);
    cudaEventCreate(&stopcusparse);
    
    // Dev memory -- allocate and copy Amat, Bmat
    int *dA_rows, *dA_columns;
    int *dB_rows, *dB_columns;
    int *dC_rows, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    int *dA_csrOffsets, *dA_csrColumns;
    int *dB_csrOffsets, *dB_csrColumns;
    float *dA_csrValues, *dB_csrValues;
    int *dC_csrOffsets, *dC_csrColumns;

    char *gpu_ldata;

    // allocate
    cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dA_csrColumns, Annz * sizeof(int));
    cudaMalloc((void**) &dA_csrValues,  Annz * sizeof(float));

    cudaMalloc((void**) &dB_csrOffsets, (B_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dB_csrColumns, Bnnz * sizeof(int));
    cudaMalloc((void**) &dB_csrValues,  Bnnz * sizeof(float));

    cudaMalloc((void**) &dC_csrOffsets, (A_num_rows + 1) * sizeof(int));

    cudaMalloc((void**) &gpu_ldata, foreignKeySize);

    // copy A
    cudaMemcpy(gpu_ldata, ldata, foreignKeySize, cudaMemcpyHostToDevice);
    
    // copy B
    cudaMemcpy(dB_csrOffsets, hB_csrOffsets, (B_num_rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrColumns, hB_csrColumns, Bnnz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrValues, hB_csrValues, Bnnz * sizeof(float),
               cudaMemcpyHostToDevice);

    gpu_tbl2csr<<<(MAX_THREADS+leftTupleNum-1)/MAX_THREADS, MAX_THREADS>>> (
            leftTupleNum, gpu_fact, gpu_ldata,
            dA_csrOffsets, dA_csrColumns, dA_csrValues, 0); 
    clock_gettime(CLOCK_REALTIME, &fill_end);

    cudaFree(gpu_fact);
    cudaFree(gpu_ldata);

    // call CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseCreate(&handle);
    
    clock_gettime(CLOCK_REALTIME, &cusp_start);
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;

    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, Annz,
                      dA_csrOffsets, dA_csrColumns, dA_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, B_num_rows, B_num_cols, Bnnz,
                      dB_csrOffsets, dB_csrColumns, dB_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    clock_gettime(CLOCK_REALTIME, &cusp_end);

    cudaEventRecord(startcusparse, 0);
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void**) &dBuffer1, bufferSize1);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);

    cudaMalloc((void**) &dBuffer2, bufferSize2);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    cudaEventRecord(stopcusparse, 0);

    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
    cudaMalloc((void**) &dC_csrColumns, C_nnz1 * sizeof(int));
    cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float));
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_csrColumns, dC_values);
    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    float cusparseSpGEMM_time;
    cudaEventElapsedTime(&cusparseSpGEMM_time, startcusparse, stopcusparse);
    double data_preparation = (fill_end.tv_sec-fill_start.tv_sec) * BILLION + 
        fill_end.tv_nsec - fill_start.tv_nsec;
    double cusp_preparation = (cusp_end.tv_sec-cusp_start.tv_sec) * BILLION + 
        cusp_end.tv_nsec - cusp_start.tv_nsec;
    printf("Join counts: %d\n", C_nnz1);
    printf("Data preparation time: %lf ms\n", 
            data_preparation / MILLION);
    printf("cusp init time: %lf ms\n", cusp_preparation / MILLION);
    printf("cusparseSpGEMM took %f ms\n", cusparseSpGEMM_time);
}  
