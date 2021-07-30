void tbl2csr_transpose(int tupleNum, char *joinKey, char *Xdata,
        int *csrOffsets, int *csrColumns, float *csrValues,
        int num_rows, int fillOne);

__global__ void gpu_tbl2csr(int tupleNum, char *joinKey, char *Xdata,
        int *csrOffsets, int *csrColumns, float *csrValues,
        int fillOne);

void tbl2csr_transpose_gbB(int tupleNum, char *joinKey, char *Xdata,
                           int *csrOffsets, int *csrColumns, float *csrValues,
                           int num_rows, int fillOne, char *gbColumn,
                           int num_cols, int &nnz);

void tbl2csr_gbA(int tupleNum, char *joinKey, char *Xdata,
                 int *csrOffsets, int *csrColumns, float *csrValues,
                 int num_rows, int fillOne, char *gbColumn,
                 int num_cols, int &nnz);

void tcuspmm_gbA(int Annz, int A_num_rows, int A_num_cols,
                 int Bnnz, int B_num_rows, int B_num_cols,
                 int MATRIX_K,
                 int leftTupleNum, char *fact, char *ldata,
                 int left_gbWidth, char *gbColumn,
                 int rightTupleNum, char *dim, char *rdata);

void tcuspmm_gbB(int Annz, int A_num_rows, int A_num_cols,
                 int Bnnz, int B_num_rows, int B_num_cols,
                 int MATRIX_K, int foreignKeySize,
                 int leftTupleNum, char *fact, char *ldata,
                 int right_gbWidth, char *gbColumn,
                 int rightTupleNum, char *dim, char *rdata);

void tcuspmm(int Annz, int A_num_rows, int A_num_cols,
             int Bnnz, int B_num_rows, int B_num_cols,
             int MATRIX_K, int foreignKeySize,
             int letfTupleNum, char *gpu_fact, char *ldata,
             int rightTupleNum, char *dim, char *rdata);
