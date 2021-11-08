#!/bin/bash
# execute ssb_q1_enc.py first, to encode joined columns into int type
# and filter data for the core emulation for SSB Q1 
python3 ssb_q1_enc.py

./translate.py test/emulate/ssb.schema
cd src/utility
make loader
./gpuDBLoader --lineorder ../../test/dbgen/lo_q1_test.tbl --ddate ../../test/dbgen/d_q1_test.tbl
cd ../../
./translate.py test/emulate/q1_1b1.sql test/emulate/ssb.schema
cd src/cuda
make clean && make tcudb
./TCUDB --datadir ../utility/
