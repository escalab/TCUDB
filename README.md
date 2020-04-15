# TCUDB

### Generate data loader (build xml tree), enter `trunk` directory
`./translate.py test/ssb_test/ssb.schema`

### SSBM plain data generation
```
cd test/dbgen
make
```

Create data with scale factor 1 GB

`./dbgen -vfF -T a -s 1`

### Enter src/utility directory, load SSBM data
generate "gpuDBLoader" to transform the original data (`gcc -o gpuDBLoader load.c`)

`make loader` 
load data 

`./gpuDBLoader --lineorder ../../test/dbgen/lineorder.tbl --ddate ../../test/dbgen/date.tbl --customer ../../test/dbgen/customer.tbl --supplier ../../test/dbgen/supplier.tbl --part ../../test/dbgen/part.tbl`


### Code generation (cuda code, e.g., driver.cu)
`./translate.py test/ssb_test/q1_1.sql test/ssb_test/ssb.schema`
```
--------------------------------------------------------------------
Generating XML tree ...
Generating GPU Codes ...
Done
--------------------------------------------------------------------
```

Enter `src/cuda` or `src/opencl` directory
(make sure the gencode arch for the GPU device, modify the makefile in `src/cuda`)

generate an executable file named `GPUDATABASE`

`make gpudb`

Run query (e.g., datadir is where MATRICES0, MATRICES1... are located)

`./GPUDATABASE --datadir dir`

e.g., ./GPUDATABASE --datadir ./GPUDATABASE --datadir ../utility/

```
# example output for ssb_test/q1_1.sql
[INFO]Number of groupBy results: 1
GroupBy Time: 0.913627
Materialization Time: 0.062024
Disk Load Time: 0.000000
PCIe Time: 0.000000
Kernel Time: 0.000000
Total Time: 1.018521
```
