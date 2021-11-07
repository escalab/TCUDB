#!/bin/bash

./translate.py test/entity_match/beer.schema
cd src/utility
make loader
./gpuDBLoader --tablea ../../test/dbgen/beer_tableA.tbl --tableb ../../test/dbgen/beer_tableB.tbl
echo "Finished loading beer_tableA.tbl"
echo "Finished loading beer_tableB.tbl"
cd ../../
./translate.py test/entity_match/beer.sql test/entity_match/beer.schema
cd src/cuda
make clean && make tcudb
./TCUDB --datadir ../utility/