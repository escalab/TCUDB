#!/usr/bin/env python

"""
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import datetime
import os
import subprocess
import sys


sys.path.append(os.getcwd()+"/SQL2XML")
import sql2xml
sys.path.append(os.getcwd()+"/XML2CODE")
import code_gen

CURRENT_DIR = os.getcwd()
EXEC_DIR = 'bin'
TEMP_DIR = '.tmp'
WHERE_ATTR = []
denseMM_flag = False

def genXMLTree(queryFile, tmpFilePath):
    try:
        os.mkdir(TEMP_DIR)
    except:
        pass

    with open(queryFile) as inputFile:
        global WHERE_ATTR
        global denseMM_flag
        xmlStr, WHERE_ATTR, denseMM_flag = sql2xml.toXml(inputFile)
        with open(tmpFilePath, "w") as outputFile:
            outputFile.write(xmlStr)

def genGPUCode(schemaFile, tmpFilePath):
    #print 'TODO: call job generation program in ./bin/'
    os.chdir(CURRENT_DIR)
    if tmpFilePath is None:
        cmd = 'python XML2CODE/main.py ' + schemaFile
    else:
        cmd = 'python XML2CODE/main.py ' + schemaFile + ' ' + tmpFilePath 
    subprocess.check_call(cmd, shell=True)

def print_usage():
    print('usage 1: ./translate.py <schema-file>.schema')
    print('usage 2: ./translate.py <query-file>.sql <schema-file>.schema <tableA>.tbl <tableB>.tbl')

def main():
    # if (len(sys.argv) != 2 and len(sys.argv) != 3):
    if (len(sys.argv) != 2 and len(sys.argv) != 5):
        print_usage()
        sys.exit(0)

    # if len(sys.argv) == 3:
    if len(sys.argv) == 5: ## feed optimizer with metadata
        queryFile = sys.argv[1]
        schemaFile = sys.argv[2]
        # metadataFile = sys.argv[3]
        # distinctValues = int(sys.argv[3])
        tmpFile = str(datetime.datetime.now()).replace(' ', '_') + '.xml'
        tmpFilePath = './' + TEMP_DIR + '/' + tmpFile

        print('--------------------------------------------------------------------')
        print('Generating XML tree ...')
        genXMLTree(queryFile, tmpFilePath)

        print('Generating GPU Codes ...')
        
        global WHERE_ATTR ## from sql2xml
        enable_blockMM = False
        # print("where list", WHERE_ATTR)
        tableA_name = WHERE_ATTR[2]
        tableB_name = WHERE_ATTR[-3]
        
        tableAFile = sys.argv[3]
        tableBFile = sys.argv[4]

        # Maybe keep current format and let most cases go sparse, pattern match for dense
        output = subprocess.check_output(['python', 'metaGen.py', schemaFile, tableA_name, tableAFile, tableB_name, tableBFile, WHERE_ATTR[-1]])
        num_distinct_val, total_records = output.split(",")
        num_distinct_val, total_records = int(num_distinct_val), int(total_records)
        print("metaGen #distinct_value {} total_records {}".format(num_distinct_val, total_records))
        # print("denseMM flag {}".format(denseMM_flag))

        ## Note: num_records as a factor for MSplit, depending on device memory size
        ## You can adjust this threshold
        if (total_records >= pow(2, 26)):
            enable_blockMM = True
        
###     You may modify the threshold for your system. For Amphere, we set it to 4096
        sparsity = float(1/num_distinct_val)
        sparsityThreshold = float(1/4096)

        genGPUCode(schemaFile, tmpFilePath)

        if (denseMM_flag) and (enable_blockMM):
            print("MSplit!")
            os.system('cd src/cuda; make clean && make tcudbblock')
        elif (denseMM_flag) or (sparsity > sparsityThreshold):
            print("Dense!")
            os.system('cd src/cuda; make clean && make tcudbdense')
        # elif (sparsity > sparsityThreshold):
            # print "Dense!"
            # os.system('cd src/cuda; make clean && make tcudbdense')
        else:
            print("Sparse!")
            os.system('cd src/cuda; make clean && make tcudb')
            
            
    else:
        queryFile = None
        schemaFile = sys.argv[1]
        genGPUCode(schemaFile, None)

    print('Done')
    print('--------------------------------------------------------------------')
    #TODO: keep XML for debugging
    #subprocess.check_call(['rm', '-rf', './' + TEMP_DIR])


    
if __name__ == "__main__":
    main()

