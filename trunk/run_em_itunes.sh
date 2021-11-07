#!/bin/bash

./translate.py test/entity_match/itunes.schema
cd src/utility
make loader
./gpuDBLoader --tablea ../../test/dbgen/iTunes_tableA.tbl --tableb ../../test/dbgen/iTunes_tableB.tbl
echo "Finished loading iTunes_tableA.tbl"
echo "Finished loading iTunes_tableB.tbl"
cd ../../
./translate.py test/entity_match/itunes.sql test/entity_match/itunes.schema
cd src/cuda
make clean && make tcudb
./TCUDB --datadir ../utility/