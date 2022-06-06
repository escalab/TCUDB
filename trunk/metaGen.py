#!/usr/bin/env python
# coding: utf-8

# In[9]:


import argparse
import pandas as pd
import sys

def getColumnHeader(schema, tableName):
    lines_list = open(schema).read().splitlines()
    ## Note: tableName in schema file should be capital

    for lst in lines_list:
        if lst.startswith(tableName):
            col_header = []
            for col in lst.split("|"):
                if col:
                    col_header.append(col.partition(":")[0])
            return col_header[1:]

def loadTable(table, columnHeader):
    df = pd.read_csv(table, sep="|", index_col=False, names=columnHeader)
    return df


def getMeta(df_A, df_B, join_col):
    union_df = pd.concat([df_A, df_B], ignore_index=True, sort=True)
    print("{},{}".format(union_df[join_col].nunique(), int(len(union_df)/2)))

def print_usage():
    print("usage: ./metaGen.py <schema-file>.schema <tblA-name> <tblA>.tbl <tblB-name> <tblB>.tbl <joinCol-name>")
   

def main():
    if (len(sys.argv) != 7):
        print_usage()
        sys.exit(0)
    
    schema = sys.argv[1]
    tblA_name = (sys.argv[2]).upper()
    tblA = sys.argv[3]
    tblB_name = (sys.argv[4]).upper()
    tblB = sys.argv[5]
    joinCol_name = (sys.argv[6]).upper()
    
    return getMeta(loadTable(tblA, getColumnHeader(schema, tblA_name)),                    loadTable(tblB, getColumnHeader(schema, tblB_name)),                    joinCol_name)
    
if __name__ == "__main__":
    main()

