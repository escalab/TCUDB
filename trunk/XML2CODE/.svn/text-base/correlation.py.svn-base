#! /usr/bin/python
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



import ystree
import sys
import copy

####
####__get_input_correlation__ parses the tree and put the node that scans the same node into a hash table.
####The key of the hash table is the tablenode's name
####

def __get_input_correlation__(tree,ret_dict):

    if isinstance(tree,ystree.TableNode):
        return 

    elif isinstance(tree,ystree.GroupByNode) or isinstance(tree,ystree.OrderByNode):
        if isinstance(tree.child,ystree.TableNode):
            tmp=[]
            if tree.child.table_name in ret_dict.keys():
                ret_dict[tree.child.table_name].append(tree)
            else:
                tmp.append(tree)
                ret_dict[tree.child.table_name] = tmp

        __get_input_correlation__(tree.child,ret_dict)

    elif isinstance(tree,ystree.TwoJoinNode):
        if isinstance(tree.left_child,ystree.TableNode): 
            tmp=[]
            if tree.left_child.table_name in ret_dict.keys():
                ret_dict[tree.left_child.table_name].append(tree)
            else:
                tmp.append(tree)
                ret_dict[tree.left_child.table_name] = tmp
 
        if isinstance(tree.right_child,ystree.TableNode):
            tmp=[]
            if tree.right_child.table_name in ret_dict.keys():
                ret_dict[tree.right_child.table_name].append(tree)
            else:
                tmp.append(tree)
                ret_dict[tree.right_child.table_name] = tmp

        __get_input_correlation__(tree.left_child,ret_dict)
        __get_input_correlation__(tree.right_child,ret_dict)

    elif isinstance(tree,ystree.SelectProjectNode):
        __get_input_correlation__(tree.child,ret_dict)


### decide if two nodes share the sanme partition key

def __node_pk_compare__(pk_list1,pk_list2):

    ret = True

    if pk_list1 is None or pk_list2 is None:
        return False

    if len(pk_list1) != len(pk_list2):
        return False

    for i in range(0,len(pk_list1)):
        tmp_bool = False
        for j in range(0,len(pk_list2)):
            if pk_list1[i].compare(pk_list2[j]):
                tmp_bool = True
        if tmp_bool is False:
            ret = False

    return ret


###
###__get_transmit_correlation__ checks if the node with the input correlation has transit correlation.
### The node with transit correlation will be placed into the hash table ret_dict, the key of which is table name
###

def __get_transit_correlation__(input_dict,ret_dict,pk_dict):

    for table_name in input_dict.keys():
    ## each element in the pk_list is an exp_list

        for i in range(0,len(input_dict[table_name])):
            nodex = input_dict[table_name][i]
            pk1 = nodex.get_pk()
            node_list=[]
            node_list.append(nodex)
            for j in range(i+1,len(input_dict[table_name])):
                nodey = input_dict[table_name][j]
                pk2 = nodey.get_pk() 
                for x in pk1:
                    for y in pk2:
                        if __node_pk_compare__(x,y):
                            if nodey not in node_list:
                                node_list.append(nodey)
                            if table_name in pk_dict.keys():
                                pk_dict[table_name].append(y)
                            else:
                                pk_list = []
                                pk_list.append(y)
                                pk_dict[table_name] = pk_list 
                                

            if len(node_list) > 1:
                if ret_dict.has_key(table_name) is False:
                    ret_dict[table_name] = node_list

                elif len(node_list) > len(ret_dict[table_name]):
                    ret_dict[table_name] = node_list
        

def input_transit_correlation(tree):


### step1: look for input correlation
### the key of node_dict is the table name and the value are the nodes(GroupByNode, 2JoinNode)
    node_dict = {}
    __get_input_correlation__(tree,node_dict)
    for x in node_dict.keys():
        if len(node_dict[x]) == 1:
            del node_dict[x]

### step2: look for transit correlation
    ret_dict = {}
    pk_dict = {}
    __get_transit_correlation__(node_dict,ret_dict,pk_dict)

    for table_name in ret_dict.keys():
        new_node = ystree.CompositeNode()
        new_node.pk_dict[table_name] = pk_dict[table_name]
        for node in ret_dict[table_name]:
            table_list = []
            if isinstance(node,ystree.TwoJoinNode):
                filter_name = ""
                if isinstance(node.left_child,ystree.TableNode):
                    tmp_name = ""
                    if node.left_child.table_alias != "":
                        tmp_name = node.left_child.table_alias
                    else:
                        tmp_name = node.left_child.table_name
                    table_list.append(tmp_name)
                    if node.left_child.table_name  not in pk_dict.keys():
                        filter_name = node.left_child.table_name

                if isinstance(node.right_child,ystree.TableNode):
                    tmp_name = ""
                    if node.right_child.table_alias !="":
                        tmp_name = node.right_child.table_alias
                    else:
                        tmp_name = node.right_child.table_name
                    table_list.append(tmp_name)
                    if node.right_child.table_name not in pk_dict.keys():
                        filter_name = node.right_child.table_name

                pk_list = node.get_pk()
                for x in pk_list:
                    for y in x:
                        if y.table_name == filter_name:
                            tmp=[]
                            tmp.append(x)
                            new_node.pk_dict[filter_name] = tmp
                            break

            else:
                table_list.append(table_name)

### set compositeNode's map output list
            for x in table_list:
                tn = x
                tn_alias = x

                if x in node.table_alias_dict.keys():
                    tn = node.table_alias_dict[x]

                tmp_exp_list = node.get_mapoutput(tn_alias)


                for y in tmp_exp_list:
                    new_exp = ystree.YRawColExp(tn,y.column_name)
                    new_exp.column_name = int(new_exp.column_name)
                    new_exp.column_type = y.column_type
                    if tn in new_node.mapoutput.keys():
                        if ystree.list_contain_exp(new_node.mapoutput[tn],new_exp) is False:
                            new_node.mapoutput[tn].append(new_exp)
                    else:
                        tmp=[]
                        tmp.append(new_exp)
                        new_node.mapoutput[tn] = tmp

### set mapfilter to reduce the data shuffled

                tmp_where = node.get_mapfilter(tn_alias)
                if tmp_where is not None:
                    col_list = []
                    ystree.__get_func_para__(tmp_where.where_condition_exp,col_list)
                    for exp in col_list:
                        if exp.table_name in node.table_alias_dict.keys():
                            exp.table_name = node.table_alias_dict[exp.table_name]

                    if tn in new_node.mapfilter.keys():
                        new_node.mapfilter[tn].append(tmp_where.where_condition_exp)
                    else:
                        tmp = []
                        tmp.append(tmp_where.where_condition_exp)
                        new_node.mapfilter[tn] = tmp
                else:
                    consExp = ystree.YConsExp(True,"BOOLEAN")
                    if tn in new_node.mapfilter.keys():
                        new_node.mapfilter[tn].append(consExp)
                    else:
                        tmp=[]
                        tmp.append(consExp)
                        new_node.mapfilter[tn] = tmp


###insert the composite node into the tree

            new_node.it_node_list.append(node)
            if node.parent not in new_node.it_node_list:
                node.parent.set_composite(new_node,node)

        for node in new_node.it_node_list:
            if isinstance(node,ystree.TwoJoinNode):
                if node.left_child in new_node.it_node_list:
                    continue
                if node.right_child in new_node.it_node_list:
                    continue

            gen_composite_child(new_node,node)

        for node in new_node.child_list:
### set pk_dict for each child node. The child node's parent must be a join node

            col_list = []
            join_exp = None
            if node.parent.join_explicit is True:
                join_exp = node.parent.join_condition.on_condition_exp
            else:
                join_exp = node.parent.join_condition.where_condition_exp
            if node == node.parent.left_child or node == node.parent.left_composite:
                __get_join_key__(join_exp,col_list,"LEFT")
            else:
                __get_join_key__(join_exp,col_list,"RIGHT")
            index = new_node.child_list.index(node)
            pk_list = []
            pk_list.append(col_list)
            new_node.pk_dict[index] = pk_list

def __get_join_key__(exp,col_list,table):

    if exp is None or not isinstance(exp,ystree.YFuncExp):
        return

    if len(exp.parameter_list ) ==2:
        para1 = exp.parameter_list[0]
        para2 = exp.parameter_list[1]
        tmp_bool = True
        if not isinstance(para1,ystree.YRawColExp):
            tmp_bool = False
        if not isinstance(para2,ystree.YRawColExp):
            tmp_bool = False

        if tmp_bool == True and para1.table_name != para2.table_name:
            if para1.table_name == table:
                col_list.append(para1)
            else:
                col_list.append(para2)

            return

    for x in exp.parameter_list:
        if isinstance(x,ystree.YFuncExp):
            __get_join_key__(x,col_list,table)


def gen_composite_child(new_node,child):

    if isinstance(child,ystree.GroupByNode):
        if child.composite is not None:
            new_node.child_list.append(child.composite)
        elif not isinstance(child.child,ystree.TableNode):
            new_node.child_list.append(child.child)

    elif isinstance(child,ystree.TwoJoinNode):
        if child.left_composite is not None:
            new_node.child_list.append(child.left_composite)
        elif not isinstance(child.left_child,ystree.TableNode):
            new_node.child_list.append(child.left_child)
        if child.right_composite is not None:
            new_node.child_list.append(child.right_composite)
        elif not isinstance(child.right_child,ystree.TableNode):
            new_node.child_list.append(child.right_child)


def jfc_child_pk(new_node,child,pk_list):
    for i in range(0,len(new_node.child_list)):
        tmp_node = new_node.child_list[i]
        tmp_list = []
        if isinstance(child,ystree.TwoJoinNode):
            if tmp_node == child.left_child or tmp_node == child.left_composite:
                for pk in child.get_pk()[0]:
                    for exp in child.left_child.select_list.tmp_exp_list:
                        tmp_exp = ystree.__trace_to_leaf__(child.left_child,exp,False)
                        if tmp_exp.compare(pk) is True:
                            pk.column_name = child.left_child.select_list.tmp_exp_list.index(exp)
                            tmp_list.append(pk)
                            break

            elif tmp_node == child.right_child or tmp_node == child.right_composite:
                for pk in child.get_pk()[1]:
                    for exp in child.right_child.select_list.tmp_exp_list:
                        tmp_exp = ystree.__trace_to_leaf__(child.right_child,exp,False)
                        if tmp_exp.compare(pk) is True:
                            pk.column_name = child.right_child.select_list.tmp_exp_list.index(exp)
                            tmp_list.append(pk)
                            break
        else:
            if isinstance(tmp_node,ystree.GroupByNode):
                for pk in pk_list[0]:
                    for exp in tmp_node.select_list.tmp_exp_list:
                        tmp_exp = ystree.__trace_to_leaf__(tmp_node,exp,False)
                        if tmp_exp.compare(pk) is True:
                            pk.column_name = child.right_child.select_list.tmp_exp_list.index(exp)
                            tmp_list.append(pk)
                            break
        new_pk_list = []
        new_pk_list.append(tmp_list)
        new_node.pk_dict[i] = new_pk_list


### for jfc select_list

def jfc_composite_select(new_node,child,pk_list):

    tn = None
    tn_alias = None
    if isinstance(child,ystree.TwoJoinNode):
        if isinstance(child.left_child,ystree.TableNode):
            if child.left_child.table_alias != "":
                tn = child.left_child.table_name
                tn_alias = child.left_child.table_alias
            else:
                tn = child.left_child.table_name
                tn_alias = tn

            tmp_exp_list = copy.deepcopy(child.get_mapoutput(tn_alias))
            for exp in tmp_exp_list:
                exp.table_name = tn
            new_node.mapoutput[tn] = tmp_exp_list
            new_node.pk_dict[tn] = pk_list

            tmp_where = child.get_mapfilter(tn_alias)
            if tmp_where is not None:
                col_list = []
                ystree.__get_func_para__(tmp_where.where_condition_exp,col_list)
                for col in col_list:
                    col.table_name = tn
                tmp = []
                tmp.append(tmp_where.where_condition_exp)
                new_node.mapfilter[tn] = tmp
        

        if isinstance(child.right_child,ystree.TableNode):
            if child.right_child.table_alias != "":
                tn = child.right_child.table_name
                tn_alias = child.right_child.table_alias
            else:
                tn = child.right_child.table_name
                tn_alias = tn 

            tmp_exp_list = copy.deepcopy(child.get_mapoutput(tn_alias))
            for exp in tmp_exp_list:
                exp.table_name = tn
            new_node.mapoutput[tn] = tmp_exp_list
            new_node.pk_dict[tn] = pk_list

            tmp_where = child.get_mapfilter(tn_alias)
            if tmp_where is not None:
                col_list = []
                ystree.__get_func_para__(tmp_where.where_condition_exp,col_list)
                for col in col_list:
                    col.table_name = tn
                tmp = []
                tmp.append(tmp_where.where_condition_exp)
                new_node.mapfilter[tn] = tmp

    else: ###### the child is a groupby node

        if isinstance(child.child,ystree.TableNode):
            tn = child.child.table_name
            if child.child.table_alias != "":
                tn_alias = child.child.table_alias
            else:
                tn_alias = tn
            tmp_exp_list = copy.deepcopy(child.get_mapoutput(tn_alias))

            for exp in tmp_exp_list:
                exp.table_name = tn
            new_node.mapoutput[tn] = tmp_exp_list
            new_node.pk_dict[tn] = pk_list

            tmp_where = child.get_mapfilter(tn_alias)
            if tmp_where is not None:
                col_list = []
                ystree.__get_func_para__(tmp_where.where_condition_exp,col_list)
                for col in col_list:
                    col.table_name = tn
                tmp = []
                tmp.append(tmp_where.where_condition_exp)
                new_node.mapfilter[tn] = tmp

def job_flow_correlation(tree):
    
    if isinstance(tree,ystree.TableNode):
        return

    elif isinstance(tree,ystree.GroupByNode):
        if isinstance(tree.child,ystree.TableNode):
            return

        if tree.composite is None:
            job_flow_correlation(tree.child)
        else:
            job_flow_correlation(tree.composite)
        
        pk1 = tree.get_pk()
        if tree.composite is None:
            pk2 = tree.child.get_pk()
        else:
            pk2 = tree.composite.get_pk()

        jfc = False
        pk_list = []

        for x in pk1:
            for y in pk2:
                if __node_pk_compare__(x,y):
                    jfc = True
                    pk_list.append(y)

        if jfc is True:
            if tree.composite is None:
                new_node = ystree.CompositeNode()
                if isinstance(tree.child,ystree.SelectProjectNode):
                    child = tree.child.child
                else:
                    child = tree.child

### generate the child list for the composite node
                gen_composite_child(new_node,child)

### generate the select list and partition key for the new CompositeNode 

                jfc_composite_select(new_node,child,pk_list)
                jfc_child_pk(new_node,child,pk_list)
                
                new_node.it_node_list.append(child)
                new_node.jfc_node_list.append(tree)

### insert the new node into the tree

                if tree.parent is not None:
                    tree.parent.set_composite(new_node,tree)
                else:
                    tree = tree.composite

            else:
                tree.composite.jfc_node_list.append(tree)
                if tree.parent is not None:
                    tree.parent.set_composite(tree.composite,tree)
                else:
                    tree = tree.composite


    elif isinstance(tree,ystree.TwoJoinNode):
        if tree.left_composite is None:
            job_flow_correlation(tree.left_child)
        else:
            job_flow_correlation(tree.left_composite)

        if tree.right_composite is None:
            job_flow_correlation(tree.right_child)
        else:
            job_flow_correlation(tree.right_composite)

        left_jfc = False
        right_jfc = False
        pk_left_list = []
        pk_right_list = []

        if tree.left_composite is None:

            if not isinstance(tree.left_child,ystree.TableNode):
                pk1 = tree.get_pk()
                pk2 = tree.left_child.get_pk()
                for x in pk1:
                    for y in pk2:
                        if __node_pk_compare__(x,y):
                            if x not in pk_left_list:
                                pk_left_list.append(x)
                            left_jfc = True

        else:
            pk1 = tree.get_pk()
            pk2 = tree.left_composite.get_pk()
            for x in pk1:
                for y in pk2:
                    if __node_pk_compare__(x,y):
                        if x not in pk_left_list:
                            pk_left_list.append(x)
                        left_jfc = True

        if tree.right_composite is None:

            if not isinstance(tree.right_child,ystree.TableNode):
                pk1 = tree.get_pk()
                pk2 = tree.right_child.get_pk()
                for x in pk1:
                    for y in pk2:
                        if __node_pk_compare__(x,y):
                            if x not in pk_right_list:
                                pk_right_list.append(x)
                            right_jfc = True

        else:
                pk1 = tree.get_pk()
                pk2 = tree.right_composite.get_pk()
                for x in pk1:
                    for y in pk2:
                        if __node_pk_compare__(x,y):
                            if x not in pk_right_list:
                                pk_right_list.append(x)
                            right_jfc = True

        if left_jfc == True and right_jfc == True:
            if tree.left_composite is not None and tree.right_composite is not None:
                tree.left_composite.jfc_node_list.append(tree)

                if tree.parent is not None:
                    tree.parent.set_composite(tree.left_composite,tree)
                else:
                    tree = tree.left_composite

            else:
                print >>sys.stderr,"Internal Error: join correlation"
                exit(29)


        elif left_jfc == True:
            if tree.left_composite is None:
                new_node = ystree.CompositeNode()
                child = None
                if isinstance(tree.left_child,ystree.SelectProjectNode):
                    child = tree.left_child.child
                else:
                    child = tree.left_child

### set the child list for the new node

                gen_composite_child(new_node,child)

                if tree.right_composite is not None:
                    new_node.child_list.append(tree.right_composite)
                elif not isinstance(tree.right_child,ystree.TableNode):
                    new_node.child_list.append(tree.right_child)

### set the select list for the new node
                jfc_composite_select(new_node,child,pk_left_list)

                new_node.it_node_list.append(child)
                new_node.jfc_node_list.append(tree)

                if tree.parent is not None:
                    tree.parent.set_composite(new_node,tree)
                else:
                    tree = new_node

            else: ### tree.left_composite is not None
                tree.left_composite.jfc_node_list.append(tree)
                if tree.right_composite is not None:
                    tree.left_composite.child_list.append(tree.right_composite)
                elif not isinstance(tree.right_child,ystree.TableNode):
                    tree.left_composite.child_list.append(tree.right_child)
                if tree.parent is not None:
                    tree.parent.set_composite(tree.left_composite,tree)
                else:
                    tree = tree.left_composite

        elif right_jfc == True:
            if tree.right_composite is None:
                new_node = ystree.CompositeNode()
                child = None
                if isinstance(tree.right_child,ystree.SelectProjectNode):
                    child = tree.right_child.child
                else:
                    child = tree.right_child

### set the child list for the new node
                gen_composite(new_node,child)
                if tree.left_composite is not None:
                    new_node.child_list.append(tree.left_composite)
                elif not isinstance(tree.left_child,ystree.TableNode):
                    new_node.child_list.append(tree.left_child)

### set the select list for the new node
                jfc_composite_select(new_node,child,pk_right_list)

                new_node.it_node_list.append(child)
                new_node.jfc_node_list.append(tree)

                if tree.parent is not None:
                    tree.parent.set_composite(new_node,tree)
                else:
                    tree = new_node

            else:
                tree.right_composite.jfc_node_list.append(tree)
                tree.right_composite.child_list.append(tree.left_child)
                if tree.parent is not None:
                    tree.parent.set_composite(tree.right_composite,tree)
                else:
                    tree = tree.right_composite
            

    elif isinstance(tree,ystree.SelectProjectNode):
        job_flow_correlation(tree.child)

    elif isinstance(tree,ystree.CompositeNode):
        if tree.dep == 0:
            tree.dep = 1
            for x in tree.child_list:
                job_flow_correlation(x)
            
            for x in tree.child_list:
                if isinstance(x,ystree.SelectProjectNode):
                    node = x.child
                else:
                    node = x

                if node in tree.it_node_list or node in tree.jfc_node_list:
                    x.parent.set_composite(None,x)
                    tree.child_list.remove(x)

            if len(tree.child_list)>0 and tree.dep>1:
                print >>sys.stderr,"Correlatioin:dependency! need to roll back"
                exit(29)

        else:
            tree.dep = tree.dep+1

    elif isinstance(tree,ystree.OrderByNode):
        job_flow_correlation(tree.child)

def remove_sp_node(tree):

    if isinstance(tree,ystree.TableNode):
        return

    elif isinstance(tree,ystree.GroupByNode) or isinstance(tree,ystree.OrderByNode):
        remove_sp_node(tree.child)
        if tree.composite is not None:
            remove_sp_node(tree.composite)

    elif isinstance(tree,ystree.TwoJoinNode):
        if tree.left_composite is not None:
            remove_sp_node(tree.left_composite)
        remove_sp_node(tree.left_child)
        if tree.right_composite is not None:
            remove_sp_node(tree.right_composite)
        remove_sp_node(tree.right_child)

    elif isinstance(tree,ystree.SelectProjectNode):
        remove_sp_node(tree.child)
        tree.child.parent = tree.parent
        if tree.parent is None:
            return
        if isinstance(tree.parent,ystree.TwoJoinNode):
            if tree == tree.parent.left_child:
                tree.parent.left_child = tree.child
            else:
                tree.parent.right_child = tree.child

        else:
            tree.parent.child = tree.child

    elif isinstance(tree,ystree.CompositeNode):
        for node in tree.child_list:
            remove_sp_node(node)

def adjust_composite_index(tree):
    if isinstance(tree,ystree.TableNode):
        return

    elif isinstance(tree,ystree.GroupByNode) or isinstance(tree,ystree.OrderByNode):

        if tree.composite is None:
            adjust_composite_index(tree.child)
        else:
            adjust_composite_index(tree.composite)

    elif isinstance(tree,ystree.TwoJoinNode):
        if tree.left_composite is not None:
            adjust_composite_index(tree.left_composite)
        else:
            adjust_composite_index(tree.left_child)

        if tree.right_composite is not None:
            adjust_composite_index(tree.right_composite)
        else:
            adjust_composite_index(tree.right_child)

    elif isinstance(tree,ystree.SelectProjectNode):
        adjust_composite_index(tree.child)

    elif isinstance(tree,ystree.CompositeNode):
        for tn in tree.mapoutput.keys():
            exp_list = tree.mapoutput[tn]
            for node in tree.it_node_list:
                if tn in node.table_list:
                    node.adjust_index(exp_list,tn)
                elif tn in node.table_alias_dict.values():
                    tn_alias = None
                    for tn_alias in node.table_alias_dict.keys():
                        if node.table_alias_dict[tn_alias] == tn:
                            break
                    if tn_alias is None:
                        print >>sys.stderr,"Internal Error:adjust_composite_index"
                        exit(29)
                    node.adjust_index(exp_list,tn_alias)

        for child in tree.child_list:
            adjust_composite_index(child)
    

def ysmart_correlation(tree):

    input_transit_correlation(tree)

    job_flow_correlation(tree)

    adjust_composite_index(tree)

    remove_sp_node(tree)

    return tree



