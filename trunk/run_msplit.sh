#!/bin/bash
DIM=8192

if [ ! -f test/dbgen/A_${DIM}x${DIM}.tbl ]; then
    echo "A_${DIM}x${DIM}.tbl not found!"
    echo "Using denseTblGen.py to create A_${DIM}x${DIM}.tbl first." && exit 0;
fi

if [ ! -f test/dbgen/B_${DIM}x${DIM}.tbl ]; then
    echo "B_${DIM}x${DIM}.tbl not found!"
    echo "Using denseTblGen.py to create B_${DIM}x${DIM}.tbl first." && exit 0;
fi

./translate.py test/simple_test/simple.schema
cd src/utility
make loader
./gpuDBLoader --mat1 ../../test/dbgen/A_${DIM}x${DIM}.tbl --mat2 ../../test/dbgen/B_${DIM}x${DIM}.tbl
cd ../../
echo "Generating metadata..."
./translate.py test/simple_test/simpleMM.sql test/simple_test/simple.schema test/dbgen/A_${DIM}x${DIM}.tbl test/dbgen/B_${DIM}x${DIM}.tbl
src/cuda/TCUDB --datadir src/utility/
