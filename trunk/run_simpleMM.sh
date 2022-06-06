#!/bin/bash
# DIM=16
DIM=256

./translate.py test/simple_test/simple.schema
cd src/utility
make loader
./gpuDBLoader --mat1 ../../test/dbgen/A_${DIM}x${DIM}.tbl --mat2 ../../test/dbgen/B_${DIM}x${DIM}.tbl
echo "Finished loading A_${DIM}x${DIM}.tbl"
echo "Finished loading B_${DIM}x${DIM}.tbl"
cd ../../
./translate.py test/simple_test/simpleMM.sql test/simple_test/simple.schema test/dbgen/A_${DIM}x${DIM}.tbl test/dbgen/B_${DIM}x${DIM}.tbl
# cd src/cuda
# make clean && make tcudbdense
src/cuda/TCUDB --datadir src/utility/
