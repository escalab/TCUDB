#!/usr/bin/env python

"""
   Copyright (c) 2013 The Ohio State University.

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

'''
The frontend of YSmart. Takes an SQL query file and generates an XML representation of it.
Created on May 6, 2013

@author: Meisam
'''

import antlr3
import antlr3.tree
from YSmartLexer import *
from YSmartParser import *


def toXml(sqlFile):
    '''
    parses the contents of the given file and returns a string that represents the AST in xml format 
    '''
    with sqlFile:
        # TODO Meisam: This is a hack to make the grammar case insensitive.
        query = sqlFile.read()
        stringStream = antlr3.StringStream(query.upper())
        lexer = YSmartLexer(stringStream)
        
        tokenStream = antlr3.CommonTokenStream(lexer)
        
        parser = YSmartParser(tokenStream)
        
        parseTree = parser.start_rule()
        
        return traverseTree(parseTree.tree, query)
    
def traverseTree(tree, query):
    '''
    traverses the given tree to create an XML string  
    '''
    isRoot = False

    xmlStr = ""
    if tree is None:
        return  xmlStr
    
    elif tree.token is None: # this is the root node of the SQL query
        xmlStr += '<query>\n'
        # TODO Meisam: change this to a method parameter?
        isRoot = True
    
    for child in tree.children:
        token = child.token
        if token is None:
            type = 0 # the "<invalid>" token
        else:
            type = token.getType()
        name = YSmartParser.tokenNames[type]
        line = child.getLine()
        position = child.getCharPositionInLine()
        childCount = len(child.children)
        
        xmlStr = xmlStr + ('<node tokentype="%d" tokenname="%s" line="%d" positioninline="%d" childcount="%d">\n' % (type, name, line, position, childCount))
        
        content = token2str(child, query)
        
        xmlStr += ('<content>%s</content>\n' % escapeXmlCharacters(content));
                
        xmlStr += traverseTree(child, query)
        xmlStr += '</node>'
    
    if isRoot:
        xmlStr += '</query>\n'
        
    return xmlStr

def token2str(token, query):
    if token.getType() in [DOUBLEQUOTED_STRING, QUOTED_STRING]: 
        lines = query.splitlines()
        start = token.charPositionInLine
        stop = start + len(token.text) # Meisam token.stopIndex does not work
        result = lines[token.line - 1][start:stop]
        return result
    elif token.getType() in [ID]:
        # if an keyword is used as ID (columns/table name), it should be quoted
        sql_id = str(token)
        if sql_id[0] in ["'", '"']:
            assert sql_id[-1] == sql_id[0]
            return sql_id[1:-1]
    
    return str(token)
    
#TODO Meisam: This should be escaped to XML characters
def escapeXmlCharacters(rawString):
    return rawString.replace(">", "?").replace("<", "?")

if __name__ == '__main__':
    pass
