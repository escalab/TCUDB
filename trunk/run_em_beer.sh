#!/bin/bash

./translate.py test/entity_match/beer.schema
cd src/utility
make loader
# make all
./gpuDBLoader --tablea ../../test/dbgen/beer_tableA.tbl --tableb ../../test/dbgen/beer_tableB.tbl
echo "Finished loading beer_tableA.tbl"
echo "Finished loading beer_tableB.tbl"
cd ../../
./translate.py test/entity_match/beer.sql test/entity_match/beer.schema test/dbgen/beer_tableA.tbl test/dbgen/beer_tableB.tbl
src/cuda/TCUDB --datadir src/utility/