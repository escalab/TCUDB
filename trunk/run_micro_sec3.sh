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
./translate.py test/micro_test/micro_q1.sql test/micro_test/micro.schema test/dbgen/micro_k${DISTINCT}_${SIZE}A.tbl test/dbgen/micro_k${DISTINCT}_${SIZE}B.tbl
echo "run micro_q1.sql"
src/cuda/TCUDB --datadir src/utility/


./translate.py test/micro_test/micro_q3.sql test/micro_test/micro.schema test/dbgen/micro_k${DISTINCT}_${SIZE}A.tbl test/dbgen/micro_k${DISTINCT}_${SIZE}B.tbl
echo "run micro_q3.sql"
src/cuda/TCUDB --datadir src/utility/


./translate.py test/micro_test/micro_q4.sql test/micro_test/micro.schema test/dbgen/micro_k${DISTINCT}_${SIZE}A.tbl test/dbgen/micro_k${DISTINCT}_${SIZE}B.tbl
echo "run micro_q4.sql"
src/cuda/TCUDB --datadir src/utility/