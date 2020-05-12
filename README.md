# TCUDB

## Data generator

Use SSBM plain data generation as an example.

Generate data loader (build xml tree), enter `trunk` directory.

`./translate.py <table_schema>`

e.g.,

```
./translate.py test/ssb_test/ssb.schema`
cd test/dbgen
make
```

Create data with scale factor 1 GB

`./dbgen -vfF -T a -s 1`

Enter `src/utility` directory, and load SSBM data.

Generate "gpuDBLoader" to transform the original data

`make loader` 

## Load data from tables 


**`./gpuDBLoader --<table_name> <path_to_table>`**


e.g.,

```
./gpuDBLoader --lineorder ../../test/dbgen/lineorder.tbl \
               --ddate ../../test/dbgen/date.tbl \
               --customer ../../test/dbgen/customer.tbl \
               --supplier ../../test/dbgen/supplier.tbl \
               --part ../../test/dbgen/part.tbl
```
or

```
./gpuDBLoader --mat3 ../../test/dbgen/mat3.tbl \
              --mat4 ../../test/dbgen/mat4.tbl
```

## CUDA code generation (e.g., driver.cu)

**`./translate.py <query> <table_schema>`**

e.g.,

`./translate.py test/ssb_test/q1_1.sql test/ssb_test/ssb.schema`

or

`./translate.py test/simple_test/test4.sql test/simple_test/simple.schema`

Enter `src/cuda` or `src/opencl` directory
(make sure the gencode arch for the GPU device, modify the makefile in `src/cuda`, ).

Note: This experiment was conducted using NVIDIA RTX2080: `-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75`

Generate an executable program `GPUDATABASE` (main program).

`make gpudb`

Run query (e.g., datadir is where binary table files such as MATRICES0, MATRICES1...etc are located).

**`./GPUDATABASE --datadir dir`**

e.g., 

`./GPUDATABASE --datadir ../utility/`
