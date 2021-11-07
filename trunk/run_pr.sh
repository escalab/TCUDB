#!/bin/bash
SIZE=4096

./translate.py test/snap/snap.schema
cd src/utility
make loader
./gpuDBLoader --pagerank ../../test/dbgen/pagerank_${SIZE}.tbl --outdegree ../../test/dbgen/outdegree_${SIZE}.tbl
echo "Finished loading pagerank_${SIZE}.tbl"
echo "Finished loading outdegree_${SIZE}.tbl"
cd ../../
./translate.py test/snap/pr_q3.sql test/snap/snap.schema
cd src/cuda
make clean && make tcudb
./TCUDB --datadir ../utility/