# TCUDB

# Requirements

Some Python modules are require for `translate.py`(need to use Python2.7 to run).
```
config
pybind11
pandas
scipy
numpy
sklearn
os
```

# Generate shrunken graph from SNAP-penn_road
We provide `penn_road.ipynb` to generate `node.tbl`, `edge.tbl`, `outdegree.tbl` and `pagerank.tbl`.


# How to run

Use scripts in `trunk` directory to run.

```
./run_em_beer.sh
```

# Manually run queries

### SQL to XML.
```
./translate.py test/entity_match/beer.schema
```

### Load data.
```
cd src/utility/
make loader
./gpuDBLoader --tablea ../../test/dbgen/beer_tableA.tbl --tableb ../../test/dbgen/beer_tableB.tbl
```

### Compute metadata and generate CUDA driver code.
Back to `trunk` directory.
```
cd ../../
./translate.py test/entity_match/beer.sql test/entity_match/beer.schema test/dbgen/beer_tableA.tbl test/dbgen/beer_tableB.tbl
```

### Run a query.
```
src/cuda/TCUDB --datadir src/utility/
```

Note: This experiment was conducted using NVIDIA RTX3090: `-gencode arch=compute_80,code=sm_80` with Driver Version: 460.32.03    CUDA Version: 11.2

- `trunk/test/dbgen/denseTblGen.py` use to generate tables for `msplit` -- `python denseTblGen.py --dim 8192`
- `trunk/metaGen.py` precompute metadata for the estimation