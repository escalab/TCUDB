#!/bin/bash
DISTINCT=4096
SIZE=4096

./translate.py test/micro_test/micro.schema
cd src/utility
make loader
./gpuDBLoader --mata ../../test/dbgen/micro_k${DISTINCT}_${SIZE}A.tbl --matb ../../test/dbgen/micro_k${DISTINCT}_${SIZE}B.tbl
echo "Finished loading micro_k${DISTINCT}_${SIZE}A.tbl"
echo "Finished loading micro_k${DISTINCT}_${SIZE}B.tbl"
cd ../../
./translate.py test/micro_test/micro_q1.sql test/micro_test/micro.schema
cd src/cuda
make clean && make tcudb
echo "run micro_q1.sql"
./TCUDB --datadir ../utility/
cd ../../
./translate.py test/micro_test/micro_q3.sql test/micro_test/micro.schema
cd src/cuda
make clean && make tcudb
echo "run micro_q3.sql"
./TCUDB --datadir ../utility/
cd ../../
./translate.py test/micro_test/micro_q4.sql test/micro_test/micro.schema
cd src/cuda
make clean && make tcudb_dense
echo "run micro_q4.sql"
./TCUDB --datadir ../utility/