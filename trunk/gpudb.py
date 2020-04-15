#! /usr/bin/python

import sys
import os
import shutil
import pickle
import readline
import re

sys.path.insert(0, "XML2CODE")
import ystree

COMMANDS = ["help","create", "delete", "list", "restore", "load", "translate", "execute", "exit"]

RE_SPACE = re.compile('.*\s+$', re.M)

class cmdCompleter(object):

    def _listdir(self, root):
        res = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                name += os.sep
            res.append(name)
        return res

    def _complete_path(self, path=None):
        if not path:
            return self._listdir('.')
        dirname, rest = os.path.split(path)
        tmp = dirname if dirname else '.'
        res = [os.path.join(dirname, p)
                for p in self._listdir(tmp) if p.startswith(rest)]
        if len(res) > 1 or not os.path.exists(path):
            return res

        if os.path.isdir(path):
            return [os.path.join(path, p) for p in self._listdir(path)]

        return [path + ' ']

    def complete_all(self, args):
        if not args:
            return self._complete_path('.')
        return self._complete_path(args[-1])

    def complete(self, text, state):

        buffer = readline.get_line_buffer()
        line = readline.get_line_buffer().split()

        if not line:
            return [c + ' ' for c in COMMANDS][state]
        if RE_SPACE.match(buffer):
            line.append('')
        cmd = line[0].strip()
        if cmd in COMMANDS:
            impl = getattr(self, 'complete_all')
            args = line[1:]
            if args:
                return (impl(args) + [None])[state]
            return [cmd + ' '][state]
        results = [c + ' ' for c in COMMANDS if c.startswith(cmd)] + [None]
        return results[state]


def dbHelp():
    print "Command:"
    print "\tcreate DBName: create the database"
    print "\tdelete DBName: delete the database"
    print "\tlist DBName: list the table infomation in the database"
    print "\trestore DBName: restore the metadata for a created Database"
    print "\tload TableName data: load data into the given table"
    print "\ttranslate SQL: translate SQL into CUDA file"
    print "\texecute SQL: translate and execute given SQL on GPU"
    print "\texit"

def dbCreate(dbName, schemaFile):

    ret = 0
    dbTop = "database"

    if not os.path.exists(dbTop):
        os.makedirs(dbTop)

    dbPath = dbTop + "/" + dbName

    if os.path.exists(dbPath):
        return -1

    os.makedirs(dbPath)

    cmd = 'python XML2CODE/main.py ' + schemaFile + ' &> /dev/null'
    ret = os.system(cmd)

    if ret !=0 :
        return -1

    cmd = 'make -C src/utility/ loader &> /dev/null'
    ret = os.system(cmd)

    if ret != 0:
        return -1

    cmd = 'mv src/utility/gpuDBLoader ' + dbPath
    ret = os.system(cmd)

    if ret != 0:
        return -1

    cmd = 'mv src/utility/.metadata ' + dbPath
    ret = os.system(cmd)

    if ret != 0:
        return -1

    return 0

def dbDelete(dbName):

    dbTop = "database"

    dbPath = dbTop + "/" + dbName
    if os.path.exists(dbPath):
        shutil.rmtree(dbPath)

def dbList(dbName):

    dbTop = "database"
    dbPath = dbTop + "/" + dbName

    if not os.path.exists(dbPath):
        return -1

    metaPath = dbPath + "/.metadata" 

    if not os.path.exists(metaPath):
        return -2

    metaFile = open(metaPath, 'rb')
    tableDict = pickle.load(metaFile)
    metaFile.close()

    for tn in tableDict.keys():
        print tn

    return 0

def dbRestore(dbName, schemaFile):

    dbTop = "database"
    dbPath = dbTop + "/" + dbName

    if not os.path.exists(dbPath):
        return -1

    cmd = 'python XML2CODE/main.py ' + schemaFile + ' &> /dev/null'
    ret = os.system(cmd)

    if ret !=0 :
        return -1

    cmd = 'make -C src/utility/ loader &> /dev/null'
    ret = os.system(cmd)

    if ret != 0:
        return -1

    cmd = 'mv src/utility/gpuDBLoader ' + dbPath
    ret = os.system(cmd)

    if ret != 0:
        return -1

    cmd = 'mv src/utility/.metadata ' + dbPath
    ret = os.system(cmd)

    if ret != 0:
        return -1

    return 0


if len(sys.argv) != 2:
    print "./gpudb.py schemaFile"
    exit(-1)

schemaFile = sys.argv[1]

while 1:
    ret = 0
    dbCreated = 0
    dbName = ""

    comp = cmdCompleter()
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(comp.complete)

    cmd = raw_input(">")
    cmdA = cmd.lstrip().rstrip().split()

    if len(cmdA) == 0:
        continue

    if cmdA[0].upper() == "HELP":
        dbHelp()

    elif cmdA[0].upper() == "?":
        dbHelp()

    elif cmdA[0].upper() == "EXIT":
        break

    elif cmdA[0].upper() == "CREATE":

        if dbCreated !=0:
            print "Already created database. Delete first."
            continue


        if len(cmdA) !=2:
            print "usage: create DBName"

        else:
            ret = dbCreate(cmdA[1].upper(), schemaFile)
            if ret == -1:
                print cmdA[1] + " already exists"
            else:
                dbCreated = 1
                dbName = cmdA[1].upper()
                print cmdA[1] + " has been successfully created."


    elif cmdA[0].upper() == "DELETE":
        if len(cmdA) != 2:
            print "usage: delete DBName"

        dbCreated = 0
        dbDelete(cmdA[1].upper())

        print cmdA[1] + " has been successfully deleted."

    elif cmdA[0].upper() == "LIST":
        if len(cmdA) != 2:
            print "usage: list DBName"
            continue

        ret = dbList(cmdA[1].upper())
        if ret == -1:
            print cmdA[1] + " doesn't exist"
        elif ret == -2:
            print cmdA[1] + " metaData doesn't exist"

    elif cmdA[0].upper() == "RESTORE":
        if len(cmdA) != 2:
            print "usage: restore DBName"
            continue

        ret = dbRestore(cmdA[1].upper(),schemaFile)

        if ret == -1:
            print "Failed to restore the metadata for " + cmdA[1]
            continue

    elif cmdA[0].upper() == "LOAD":
        pass

    else:
        print "Unknown command"

os.system("clear")

