# TCUDB

# Requirements

Some Python modules are require for `translate.py`(Python2.7) and `ssb_q1_enc.py`(Python3.7).
```
config
pandas
numpy
sklearn.preprocessing
os
```
# How to generate SSBM data files
In directory `trunk/test/dbgen`, using makefile.suite as a basis.

Type `make` to compile and to generate the SSBM dbgen executable.

To generate the tables (SF=1):
```shell
# customer.tbl
dbgen -s 1 -T c

# part.tbl
dbgen -s 1 -T p

# supplier.tbl
dbgen -s 1 -T s

# date.tbl
dbgen -s 1 -T d

# lineorder.tbl
dbgen -s 1 -T l
```
For TCUDB, both tables require the join keys to be encoded in `integer` type instead of the original `date` type.
We provide `ssb_q1_enc.py` to encode the joined column for an example.

# Generate shrunken graph from SNAP-penn_road
We provide `penn_road.ipynb` to generate `node.tbl`, `edge.tbl`, `outdegree.tbl` and `pagerank.tbl`.


# How to run

Use scripts in `trunk` directory to run.

```
./run_em_beer.sh
```

# Manually run queries

### XML to GPU code.
```
./translate.py test/entity_match/beer.schema
```

### Load data.
```
cd src/utility/
make loader
./gpuDBLoader --tablea ../../test/dbgen/beer_tableA.tbl --tableb ../../test/dbgen/beer_tableB.tbl
```

### Parse query and generate CUDA driver
Back to `trunk` directory.
```
cd ../../
./translate.py test/entity_match/beer.sql test/entity_match/beer.schema
```

### Compile TCU node
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

Note: This experiment was conducted using NVIDIA RTX390: `-gencode arch=compute_80,code=sm_80` with Driver Version: 460.32.03    CUDA Version: 11.2
