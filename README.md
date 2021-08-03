# TCUDB

# How to run

Use script in `trunk` directory to run TCUDB.

```
./run_ssb_q1_1.sh -q test/emulate/q1_1b1.sql
```

# Manually run queries

### XML to GPU code.
```
./translate.py test/emulate/ssb.schema
```

### Load data. (Need to know what tables are going to be used)

The following code snippet uses SSB q1_1.sql as an example.

For TCUDB, both tables require the join keys to be encoded in `integer` type instead of the original `date` type.
```
cd src/utility/
make loader
./gpuDBLoader --lineorder ../../test/dbgen/lo_q1_1_sf1_enc.tbl --ddate ../../test/dbgen/d_q1_1_sf1_enc.tbl
```
### Parse query and generate CUDA driver
Back to `trunk` directory.
```
cd ../../
./translate.py test/emulate/q1_1b1.sql test/emulate/ssb.schema
```

### Compile TCU node
Through the `Makefile` in `src/cuda`, there are some compile flags provided.

`-DCUSPARSE` -- Using cuSPARSE APIs. (w/o this falg, TCUDB will use dense-filling method)
`-DDEBUG` -- print some statistics for debugging

```
cd src/cuda
make clean && make tcudb
```
### Run a query
```
./TCUDB --datadir ../utility/
```

Note: This experiment was conducted using NVIDIA RTX390: `-gencode arch=compute_80,code=sm_80`
