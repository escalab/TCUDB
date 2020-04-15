/*
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
*/

//*************************************************************************
// The file is used to generate an SQL parser via ANTLR.                 
//                                                                       
// The file is compatible with ANTLR 3.3 (not with 3.4).                 
//                                                                       
// Authors: Rubao Lee liru@cse.ohio-state.edu                            
//          Yuan Yuan yuanyu@cse.ohio-state.edu                          
//          Yin Huai  huai@cse.ohio-state.edu                            
// Time: 11-30-2011                                                      
//*************************************************************************



//**************************************************************************
//                                                                                                                                 
// The file is adapated from an existing grammar file for Oracle SQL 
// in the ANTLR grammer list (http://www.antlr.org/grammar/list).
// Its direct link is http://www.ibrezina.net/OracleSQL.tgz
// The original authors and comments are cited as follows.                
//                                                                        
//**************************************************************************
/*******************************************************************************

DESCRIPTION:
		Grammar for Oracle's SELECT statement for ANTLR v3, target C language
AUTHOR:
		Ivan.Brezina (ibre5041@ibrezina.net)

DATE:
		AUG 2010
BASED ON:
		PLSQL3.g Andrey Kharitonkin (thikone@gmail.com)
		PLSQLGrammar.g for ANTLR v2
		Qazi Firdous Ahmed (qazif_ahmed@infosys.com) 
		Krupa Benhur (krupa_bg@infosys.com)
		Manojaba Banerjee (manojaba_banerjee@infosys.com)
		Infosys Technologies Ltd., Bangalore, India
		Sept 18, 2002
		This grammar is for PL/SQL.
COMMENT:
		This grammar file is based on freely downloadable
		file PLSQL3.g. I extracted only those rules that
		are mandatory for SELECT statement - no PL/SQL support.
		Column list was partialy rewritten. These features were added:
		- support for analytic queries
		- list of reserved words and keywords
		- query factoring
		- model
		- perl style string notation 
ORIGINAL COMMENT:
		The grammar has been mostly re-written for ANTLR v3,
		using Oracle 10g Release 2 documentation and ANTLR book.
		New SQL and PL/SQL expression rules, SQL statments
		SELECT, INSERT, UPDATE, DELETE are fully supported.
		Generated parser can parse most of valid PL/SQL and 
		it was tested with over 10 Mb of test source code.
		Let me know if something cannot be parsed by this grammar.
KNOWN ISSUES:
		XQUERIES are unsupported. List of reserved words/keywords
		needs to be amended. PL/SQL support was removed.
		Some reserved words are treaded like keywords('JOIN', 'MODEL', 'PARTITION', ...)
*******************************************************************************/



grammar YSmart;

options {
	language=Python;
//	k=5;
	backtrack=true;
	memoize=true;
	output=AST;
}

tokens {
    T_RESERVED = 'reserved';
    T_ALIAS = 'alias';
    T_TABLE_NAME = 'table_name';
    
    T_WITH = 't_with';
    T_SELECT = 't_select';
    T_COLUMN_LIST = 't_column_list';
    T_SELECT_COLUMN = 't_select_column';
    T_FROM = 't_from';
    T_SELECTED_TABLE = 'selected_table';
    T_WHERE = 't_where';
    T_HIERARCHICAL = 't_hierarchical';
    T_GROUP_BY = 't_group_by';
    T_HAVING = 't_having';
    T_MODEL = 't_model';
    T_UNION = 't_union';
    T_ORDER_BY_CLAUSE='t_order_by';
    T_FOR_UPDATE_CLAUSE='t_for_update';

    T_COND_OR='t_cond_or';
    T_COND_AND='t_cond_and';
    T_COND_NOT='t_cond_not';
    T_COND_exists='t_cond_exists';
    T_COND_is='t_cond_is';
    T_COND_comparison='t_cond_comparison';
    T_COND_group_comparison='t_cond_group_comparison';
    T_COND_in='t_cond_in';
    T_COND_is_a_set='t_cond_is_a_set';
    T_COND_is_any='t_cond_is_any';
    T_COND_is_empty='t_cond_is_empty';
    T_COND_is_of_type='t_cond_is_of_type';
    T_COND_is_present='t_cond_is_present';
    T_COND_like='t_cond_like';
    T_COND_memeber='t_cond_memeber';
    T_COND_between='t_cond_between';
    T_COND_regexp_like='t_cond_regexp_like';
    T_COND_submultiset='t_cond_submultiset';
    T_COND_equals_path='t_cond_equals_path';
    T_COND_under_path='t_cond_under_path';
    T_COND_paren='t_cond_paren';
}

start_rule
	: (
            select_statement
      )            
      SEMI
      ;      

/* ================================================================================
   SELECT Statement
   ================================================================================ */
select_statement
	:
	sel=k_select select_list
	k_from	table_reference_list        
	( where_clause )?
	( group_by_clause )?
	( having_clause )?
	( order_by_clause )?
	->	^('t_select'
		$sel
		select_list
		^('t_from' k_from table_reference_list)
		where_clause?
       	group_by_clause?
       	having_clause?
		order_by_clause?
		)
	;

/* ================================================================================
   Query column list specs (ie. everything between "SELECT" ... "FROM"
   ================================================================================ */
select_list
	: displayed_column_part_first displayed_column_part_next*
		-> ^('t_column_list' displayed_column_part_first displayed_column_part_next*)
	;
displayed_column_part_first
	: displayed_column
    ;
displayed_column_part_next options { backtrack=false; }
	: c=COMMA displayed_column {$c.setChannel(30) }
    ;        
displayed_column
	: (
        asterisk1=ASTERISK
		| schema=sql_identifier DOT asterisk2=ASTERISK
		| sql_expression
		)   
		(alias|alias_name=sql_identifier)?
        -> ^('t_select_column' $asterisk1? $schema? DOT? $asterisk2? sql_expression? alias? $alias_name? )
    ;
sql_expression
	:	expr_add
	;
expr_add
	:	expr_mul ( ( PLUS | MINUS | DOUBLEVERTBAR ) expr_mul )*
	;
expr_mul
	:	expr_sign ( ( ASTERISK | DIVIDE ) expr_sign )*
	;
expr_sign // in fact this is not "sign" but unary operator
	:	( PLUS | MINUS | k_prior | k_connect_by_root )? expr_pow
	;
expr_pow
	:	expr_expr ( EXPONENT expr_expr )*
	;
expr_expr
	:	datetime_expression
	|	interval_expression        
	|	( expr_paren ) => expr_paren
	|	( cast_expression) => cast_expression
	|	( function_expression ) => function_expression
	|	( case_expression ) => case_expression
	|	( simple_expression ) => simple_expression
	|	( subquery ) => subquery
	;
expr_paren
	:	LPAREN nested_expression RPAREN
	;
nested_expression
	:	sql_expression
	;
function_expression
 	:	(function_name) LPAREN call_parameters RPAREN
	;

call_parameters
	: ASTERISK
	| call_parameter ( COMMA call_parameter )*
	;
call_parameter
	:	 ( parameter_name ARROW )? nested_expression
	;
parameter_name
	:	identifier
	;
case_expression
	:	k_case ( simple_case_expression | searched_case_expression ) ( else_case_expression )? k_end
	;
simple_case_expression
	:	nested_expression ( k_when nested_expression k_then nested_expression )+
	;
searched_case_expression
	:	( k_when sql_condition k_then nested_expression )+
	;
else_case_expression
	:	k_else nested_expression
	;

simple_expression
	:	boolean_literal
	|	k_sql ( FOUND_ATTR | NOTFOUND_ATTR | ISOPEN_ATTR | ROWCOUNT_ATTR | BULK_ROWCOUNT_ATTR )
	|	( column_spec ) => column_spec
	|	quoted_string
	|	NUMBER
	;        


subquery
	:	LPAREN select_statement RPAREN
	|	LPAREN subquery RPAREN
	;

datetime_expression
	:
        ( function_expression | cast_expression | simple_expression )
        k_at (k_local | k_time k_zone ( quoted_string | k_dbtimezone | k_sessiontimezone | sql_expression ));

interval_expression
	:
		LPAREN ( function_expression | cast_expression | simple_expression ) MINUS ( function_expression | cast_expression | simple_expression ) RPAREN
		(	k_day (LPAREN NUMBER RPAREN)? k_to k_second (LPAREN NUMBER RPAREN)?
		|	k_year (LPAREN NUMBER RPAREN)? k_to k_month (LPAREN NUMBER RPAREN)?
		)
	;
/* ================================================================================
   Special expressions
   ================================================================================ */
cast_expression
	:	k_cast LPAREN (sql_expression | k_multiset subquery) k_as (datatype|sql_identifier) RPAREN
	;	
datatype
	:	k_binary_integer 
	|	k_binary_float
	|	k_binary_double
	|	k_natural
	|	k_positive
	|	( k_number | k_numeric | k_decimal | k_dec ) ( LPAREN NUMBER ( COMMA NUMBER )? RPAREN )?
	|	k_long ( k_raw)? ( LPAREN NUMBER RPAREN )?
	|	k_raw ( LPAREN NUMBER RPAREN )?
	|	k_boolean
	|	k_date
	|	k_interval k_day ( LPAREN NUMBER RPAREN )? k_to k_second ( LPAREN NUMBER RPAREN )?
	|	k_interval k_year ( LPAREN NUMBER RPAREN )? k_to k_month
	|	( k_time | k_timestamp ) ( LPAREN NUMBER RPAREN )? ( k_with ( k_local )? k_time k_zone )?
	|	k_integer
	|	k_int
	|	k_smallint
	|	k_float ( LPAREN NUMBER RPAREN )?
	|	k_real
	|	k_double k_precision
	|	k_char      ( k_varying )? ( LPAREN NUMBER ( k_byte | k_char )? RPAREN )? ( k_character k_set ( identifier | column_spec CHARSET_ATTR ) )?
	|	k_varchar                  ( LPAREN NUMBER ( k_byte | k_char )? RPAREN )? ( k_character k_set ( identifier | column_spec CHARSET_ATTR ) )?
	|	k_varchar2                 ( LPAREN NUMBER ( k_byte | k_char )? RPAREN )? ( k_character k_set ( identifier | column_spec CHARSET_ATTR ) )?
	|	k_character ( k_varying )? ( LPAREN NUMBER RPAREN )?
	|	k_nchar     ( k_varying )? ( LPAREN NUMBER RPAREN )?
	|	k_nvarchar  ( LPAREN NUMBER RPAREN )?
	|	k_nvarchar2 ( LPAREN NUMBER RPAREN )?
	|	k_national  ( k_character | k_char ) ( k_varying )? ( LPAREN NUMBER RPAREN )?
	|	k_mlslabel
	|	k_pls_integer
	|	k_blob
	|	k_clob ( k_character k_set ( identifier | column_spec CHARSET_ATTR ) )?
	|	k_nclob
	|	k_bfile
	|	k_rowid 
	|	k_urowid ( LPAREN NUMBER RPAREN )?
	;

boolean_literal
	:	k_true | k_false
	;

t_alias
	: alias_name=sql_identifier 
	;

c_alias
	: (k_as? sql_identifier) 
	| k_as
	;

alias
	:	k_as sql_identifier?
	;

column_spec
	: sql_identifier DOT sql_identifier DOT sql_identifier
	| sql_identifier DOT sql_identifier
	| sql_identifier
	| pseudo_column
	;
//TODO more pseudocolumns here - especially those who are reserved words
pseudo_column
	: k_null
    | k_sysdate
	| k_rowid
	| k_rownum
	| k_level				// hierarchical query
	| k_connect_by_isleaf
	| k_connect_by_iscycle
	| k_versions_starttime	// flashback query
	| k_versions_starscn
	| k_versions_endtime
	| k_versions_endscn 
	| k_versions_xid 
	| k_versions_operation
	| k_column_value	// XMLTABLE query
	| k_object_id		// 
	| k_object_value	//
	| k_ora_rowscn		//
	| k_xmldata
	;

function_name
	: sql_identifier DOT sql_identifier DOT sql_identifier
	| sql_identifier DOT sql_identifier
	| sql_identifier
	;

identifier
	:	ID
	|	DOUBLEQUOTED_STRING 
   	;

sql_identifier
	:	identifier
    |	keyword
	|	k_rowid
	|	k_rownum
	;

/* ================================================================================
   Query tables specs (ie. everything between "FROM" ... "WHERE"
   ================================================================================ */
table_reference_list
	:	(
			(join_clause|(LPAREN join_clause RPAREN)|table_reference)
			(COMMA (join_clause|(LPAREN join_clause RPAREN)|table_reference))*
		)
	;            
table_reference
	:	query_table_expression 
	;
query_table_expression
	:	
		( schema_name DOT)? table_name t_alias?
	|	subquery t_alias
	|       LPAREN subquery RPAREN t_alias
	;
join_clause
	:	table_reference (inner_cross_join_clause|outer_join_clause)+
	;
inner_cross_join_clause
	:	k_inner? k_join table_reference ((k_on sql_condition)|(k_using LPAREN column_specs RPAREN))
    |	(k_cross | k_natural k_inner?) (k_join table_reference)
	;        
outer_join_clause
	:	(	outer_join_type k_join
		|	k_natural ( outer_join_type )? k_join
		)
		table_reference ( k_on sql_condition | k_using LPAREN column_specs RPAREN )?
	;
outer_join_type
 	:	( k_full | k_left | k_right ) ( k_outer )?
	;        

outer_join_sign
	:	LPAREN PLUS RPAREN
	;
table_name
	:	sql_identifier
	;
schema_name
	:	sql_identifier
	;
column_specs
	:	column_spec ( COMMA column_spec )*
	;

/* ================================================================================
   where clause
   ================================================================================ */
where_clause
	:	k_where sql_condition
        -> ^( 't_where' k_where sql_condition)
	;

/* ================================================================================
   having clause
   ================================================================================ */
having_clause
	:	k_having sql_condition
        -> ^( 't_having' k_having sql_condition)
	;




/* ================================================================================
   group by clause
   ================================================================================ */
group_by_clause
	:	k_group k_by group_by_exprs
	-> ^('t_group_by' k_group k_by group_by_exprs)
	;
group_by_exprs
	:	group_by_expr ( COMMA group_by_expr )*
	;
group_by_expr
	:	rollup_cube_clause
	|	grouping_sets_clause
	|	grouping_expression_list
	;
rollup_cube_clause
	:	( k_rollup | k_cube ) LPAREN grouping_expression_list RPAREN
	;
grouping_sets_clause
	:	k_grouping k_sets LPAREN grouping_expression_list RPAREN
	;
grouping_sets_exprs
	:	grouping_sets_expr ( COMMA grouping_sets_expr )*
	;
grouping_sets_expr
	:	rollup_cube_clause | grouping_expression_list
	;
sql_condition
	:	condition_or
	;
condition_or
	:	(condition_or_part_first condition_or_part_next+ -> ^(T_COND_OR ^(T_COND_OR condition_or_part_first) condition_or_part_next*))
	|	condition_or_part_first
	;
condition_or_part_first
	:	condition_and
    ;
condition_or_part_next
	:	k_or condition_and -> ^(T_COND_OR k_or condition_and)
    ;
condition_and
    :	(condition_and_part_first condition_and_part_next+ -> ^(T_COND_AND ^(T_COND_AND condition_and_part_first) condition_and_part_next*))
	|	condition_and_part_first 
	;
condition_and_part_first
	:	condition_not
    ;
condition_and_part_next
	:	k_and condition_not -> ^(T_COND_AND k_and condition_not)
    ;
condition_not
	:	(k_not condition_expr -> ^(T_COND_NOT k_not condition_expr))
	|	condition_expr
	;
condition_expr
	:	condition_exists
	|	condition_is
	|	condition_comparison
	|	condition_group_comparison
	|	condition_in
	|	condition_is_a_set
	|	condition_is_any
	|	condition_is_empty
	|	condition_is_of_type
	|	condition_is_present
	|	condition_like
	|	condition_memeber
	|	condition_between
	|	condition_regexp_like
	|	condition_submultiset
	|	condition_equals_path
	|	condition_under_path
	|	condition_paren
	;

condition_exists
	:	k_exists subquery
	;
condition_is
	:	sql_expression k_is ( k_not )? ( k_nan | k_infinite | k_null )
	;
condition_comparison
	:	LPAREN sql_expressions RPAREN ( outer_join_sign )? ( EQ | NOT_EQ ) subquery ( outer_join_sign )?
	|	( k_prior )? sql_expression ( outer_join_sign )? ( EQ | NOT_EQ | GTH | GEQ | LTH | LEQ ) ( k_prior )? ( sql_expression | LPAREN select_statement RPAREN ) ( outer_join_sign )?
	;
condition_group_comparison
	:	LPAREN sql_expressions RPAREN ( EQ | NOT_EQ ) ( k_any | k_some | k_all ) LPAREN ( grouping_expression_list | select_statement ) RPAREN
	|	sql_expression ( EQ | NOT_EQ | GTH | GEQ | LTH | LEQ ) ( k_any | k_some | k_all ) LPAREN ( sql_expressions | select_statement ) RPAREN
	;
condition_in
	:	LPAREN sql_expressions RPAREN ( k_not )? k_in LPAREN ( grouping_expression_list | select_statement ) RPAREN
	|	sql_expression ( k_not )? k_in LPAREN ( expression_list | select_statement ) RPAREN
	;
condition_is_a_set
	:	nested_table_column_name k_is ( k_not )? k_a k_set
	;
condition_is_any
	:	( column_name k_is )? k_any
	;
condition_is_empty
	:	nested_table_column_name k_is ( k_not )? k_empty
	;
condition_is_of_type
	:	sql_expression k_is (k_not)? k_of ( k_type )? LPAREN type_name RPAREN
	;
condition_is_of_type_names
	:	condition_is_of_type_name ( COMMA condition_is_of_type_name )*
	;
condition_is_of_type_name
	:	( k_only )? type_name
	;
condition_is_present
	:	cell_reference k_is k_present
	;
condition_like
	:	sql_expression ( k_not )? ( k_like | k_likec | k_like2 | k_like4 ) sql_expression ( k_escape sql_expression )?
	;
condition_memeber
	:	sql_expression ( k_not )? k_member ( k_of )? nested_table_column_name
	;
condition_between
	:	sql_expression ( k_not )? k_between sql_expression k_and sql_expression
	;
condition_regexp_like
	:	k_regexp_like LPAREN call_parameters RPAREN
	;
condition_submultiset
	:	nested_table_column_name ( k_not )? k_submultiset ( k_of )? nested_table_column_name
	;
condition_equals_path
	:	k_equals_path LPAREN column_name COMMA path_string ( COMMA correlation_integer )? RPAREN
	;
condition_under_path
	:	k_under_path LPAREN column_name ( COMMA levels )? COMMA path_string ( COMMA correlation_integer )? RPAREN
	;
levels
	:	integer
	;
correlation_integer
	:	integer
	;
path_string
	:	quoted_string
	;
type_name
	:	identifier ( DOT identifier )*
	;
integer
	:	NUMBER
	;
column_name
	:	sql_identifier
	;
nested_table
	:	sql_identifier
	;
nested_table_column_name
	:	( schema_name DOT )? (table_name DOT)? (nested_table DOT)? column_name
	;
sql_expressions
	:	sql_expression ( COMMA sql_expression )*
	;
grouping_expression_list
	:	expression_list ( COMMA expression_list )*
	;
expression_list
	:	LPAREN sql_expressions RPAREN
	|	sql_expressions
	;
cell_reference
	:	sql_identifier
	;

condition_paren
	:	LPAREN sql_condition RPAREN
	;

	
/* ================================================================================
   ORDER BY clause
   ================================================================================ */
order_by_clause
	:	k_order k_siblings ? k_by order_by_clause_part_first order_by_clause_part_next*
	-> ^('t_order_by' k_order k_siblings ? k_by order_by_clause_part_first order_by_clause_part_next*)
	;
order_by_clause_part_first
	:	sql_expression k_asc ? k_desc ? (k_nulls k_first)? (k_nulls k_last)?
	;        
order_by_clause_part_next
	:	COMMA sql_expression k_asc ? k_desc ? (k_nulls k_first)? (k_nulls k_last)?
	;        


/* ================================================================================
   Oracle reserved words
   cannot by used for name database objects such as columns, tables, or indexes.
   ================================================================================ */
k_access : r='ACCESS' { $r.setType(T_RESERVED) }  ;
k_add : r='ADD' { $r.setType(T_RESERVED) }  ;
k_all : r='ALL' { $r.setType(T_RESERVED) }  ;
k_alter : r='ALTER' { $r.setType(T_RESERVED) }  ;
k_and : r='AND' { $r.setType(T_RESERVED) }  ;
k_any : r='ANY' { $r.setType(T_RESERVED) }  ;
k_arraylen : r='ARRAYLEN' { $r.setType(T_RESERVED) }  ;
k_as : r='AS' { $r.setType(T_RESERVED) }  ;
k_asc : r='ASC' { $r.setType(T_RESERVED) }  ;
k_audit : r='AUDIT' { $r.setType(T_RESERVED) }  ;
k_between : r='BETWEEN' { $r.setType(T_RESERVED) }  ;
k_by : r='BY' { $r.setType(T_RESERVED) }  ;
k_case : r='CASE' { $r.setType(T_RESERVED) }  ; //PL/SQL
k_char : r='CHAR' { $r.setType(T_RESERVED) }  ;
k_check : r='CHECK' { $r.setType(T_RESERVED) }  ;
k_cluster : r='CLUSTER' { $r.setType(T_RESERVED) }  ;
k_column : r='COLUMN' { $r.setType(T_RESERVED) }  ;
k_comment : r='COMMENT' { $r.setType(T_RESERVED) }  ;
k_compress : r='COMPRESS' { $r.setType(T_RESERVED) }  ;
k_connect : r='CONNECT' { $r.setType(T_RESERVED) }  ;
k_create : r='CREATE' { $r.setType(T_RESERVED) }  ;
k_current : r='CURRENT' { $r.setType(T_RESERVED) }  ;
k_date : r='DATE' { $r.setType(T_RESERVED) }  ;
k_decimal : r='DECIMAL' { $r.setType(T_RESERVED) }  ;
k_default : r='DEFAULT' { $r.setType(T_RESERVED) }  ;
k_delete : r='DELETE' { $r.setType(T_RESERVED) }  ;
k_desc : r='DESC' { $r.setType(T_RESERVED) }  ;
k_distinct : r='DISTINCT' { $r.setType(T_RESERVED) }  ;
k_drop : r='DROP' { $r.setType(T_RESERVED) }  ;
k_else : r='ELSE' { $r.setType(T_RESERVED) }  ;
k_exclusive : r='EXCLUSIVE' { $r.setType(T_RESERVED) }  ;
k_exists : r='EXISTS' { $r.setType(T_RESERVED) }  ;
k_false : r='FALSE' { $r.setType(T_RESERVED) }  ; //PL/SQL
k_file : r='FILE' { $r.setType(T_RESERVED) }  ;
k_float : r='FLOAT' { $r.setType(T_RESERVED) }  ;
k_for : r='FOR' { $r.setType(T_RESERVED) }  ;
k_from : r='FROM' { $r.setType(T_RESERVED) }  ;
k_grant : r='GRANT' { $r.setType(T_RESERVED) }  ;
k_group : r='GROUP' { $r.setType(T_RESERVED) }  ;
k_having : r='HAVING' { $r.setType(T_RESERVED) }  ;
k_identified : r='IDENTIFIED' { $r.setType(T_RESERVED) }  ;
k_immediate : r='IMMEDIATE' { $r.setType(T_RESERVED) }  ;
k_in : r='IN' { $r.setType(T_RESERVED) }  ;
k_increment : r='INCREMENT' { $r.setType(T_RESERVED) }  ;
k_index : r='INDEX' { $r.setType(T_RESERVED) }  ;
k_initial : r='INITIAL' { $r.setType(T_RESERVED) }  ;
k_insert : r='INSERT' { $r.setType(T_RESERVED) }  ;
k_integer : r='INTEGER' { $r.setType(T_RESERVED) }  ;
k_intersect : r='INTERSECT' { $r.setType(T_RESERVED) }  ;
k_into : r='INTO' { $r.setType(T_RESERVED) }  ;
k_is : r='IS' { $r.setType(T_RESERVED) }  ;
k_level : r='LEVEL' { $r.setType(T_RESERVED) }  ;
k_like : r='LIKE' { $r.setType(T_RESERVED) }  ;
k_like2 : r='LIKE2' { $r.setType(T_RESERVED) }  ;
k_like4 : r='LIKE4' { $r.setType(T_RESERVED) }  ;
k_likec : r='LIKEC' { $r.setType(T_RESERVED) }  ;
k_lock : r='LOCK' { $r.setType(T_RESERVED) }  ;
k_long : r='LONG' { $r.setType(T_RESERVED) }  ;
k_maxextents : r='MAXEXTENTS' { $r.setType(T_RESERVED) }  ;
k_minus : r='MINUS' { $r.setType(T_RESERVED) }  ;
k_mode : r='MODE' { $r.setType(T_RESERVED) }  ;
k_modify : r='MODIFY' { $r.setType(T_RESERVED) }  ;
k_noaudit : r='NOAUDIT' { $r.setType(T_RESERVED) }  ;
k_nocompress : r='NOCOMPRESS' { $r.setType(T_RESERVED) }  ;
k_not : r='NOT' { $r.setType(T_RESERVED) }  ;
k_notfound : r='NOTFOUND' { $r.setType(T_RESERVED) }  ;
k_nowait : r='NOWAIT' { $r.setType(T_RESERVED) }  ;
k_null : r='NULL' { $r.setType(T_RESERVED) }  ;
k_number : r='NUMBER' { $r.setType(T_RESERVED) }  ;
k_of : r='OF' { $r.setType(T_RESERVED) }  ;
k_offline : r='OFFLINE' { $r.setType(T_RESERVED) }  ;
k_on : r='ON' { $r.setType(T_RESERVED) }  ;
k_online : r='ONLINE' { $r.setType(T_RESERVED) }  ;
k_option : r='OPTION' { $r.setType(T_RESERVED) }  ;
k_or : r='OR' { $r.setType(T_RESERVED) }  ;
k_order : r='ORDER' { $r.setType(T_RESERVED) }  ;
k_pctfree : r='PCTFREE' { $r.setType(T_RESERVED) }  ;
k_prior : r='PRIOR' { $r.setType(T_RESERVED) }  ;
k_privileges : r='PRIVILEGES' { $r.setType(T_RESERVED) }  ;
k_public : r='PUBLIC' { $r.setType(T_RESERVED) }  ;
k_raw : r='RAW' { $r.setType(T_RESERVED) }  ;
k_rename : r='RENAME' { $r.setType(T_RESERVED) }  ;
k_resource : r='RESOURCE' { $r.setType(T_RESERVED) }  ;
k_revoke : r='REVOKE' { $r.setType(T_RESERVED) }  ;
k_row : r='ROW' { $r.setType(T_RESERVED) }  ;
k_rowid : r='ROWID' { $r.setType(T_RESERVED) }  ;
k_rowlabel : r='ROWLABEL' { $r.setType(T_RESERVED) }  ;
k_rownum : r='ROWNUM' { $r.setType(T_RESERVED) }  ;
k_rows : r='ROWS' { $r.setType(T_RESERVED) }  ;
k_select : r='SELECT' { $r.setType(T_RESERVED) }  ;
k_session : r='SESSION' { $r.setType(T_RESERVED) }  ;
k_set : r='SET' { $r.setType(T_RESERVED) }  ;
k_share : r='SHARE' { $r.setType(T_RESERVED) }  ;
k_size : r='SIZE' { $r.setType(T_RESERVED) }  ;
k_smallint : r='SMALLINT' { $r.setType(T_RESERVED) }  ;
k_sqlbuf : r='SQLBUF' { $r.setType(T_RESERVED) }  ;
k_start : r='START' { $r.setType(T_RESERVED) }  ;
k_successful : r='SUCCESSFUL' { $r.setType(T_RESERVED) }  ;
k_synonym : r='SYNONYM' { $r.setType(T_RESERVED) }  ;
k_sysdate : r='SYSDATE' { $r.setType(T_RESERVED) }  ;
k_table : r='TABLE' { $r.setType(T_RESERVED) }  ;
k_then : r='THEN' { $r.setType(T_RESERVED) }  ;
k_to : r='TO' { $r.setType(T_RESERVED) }  ;
k_trigger  : r='TRIGGER' { $r.setType(T_RESERVED) }  ;
k_true  : r='TRUE' { $r.setType(T_RESERVED) }  ; // PL/SQL
k_uid : r='UID' { $r.setType(T_RESERVED) }  ;
k_union : r='UNION' { $r.setType(T_RESERVED) }  ;
k_unique : r='UNIQUE' { $r.setType(T_RESERVED) }  ;
k_update : r='UPDATE' { $r.setType(T_RESERVED) }  ;
k_user  : r='USER' { $r.setType(T_RESERVED) }  ;
k_validate : r='VALIDATE' { $r.setType(T_RESERVED) }  ;
k_values : r='VALUES' { $r.setType(T_RESERVED) }  ;
k_varchar : r='VARCHAR' { $r.setType(T_RESERVED) }  ;
k_varchar2 : r='VARCHAR2' { $r.setType(T_RESERVED) }  ;
k_view : r='VIEW' { $r.setType(T_RESERVED) }  ;
k_whenever : r='WHENEVER' { $r.setType(T_RESERVED) }  ;
k_where : r='WHERE' { $r.setType(T_RESERVED) }  ;
k_with : r='WITH' { $r.setType(T_RESERVED) }  ;

reserved_word options { backtrack=false; }
	: r=( 'ACCESS'	| 'ADD'	| 'ALL'	| 'ALTER'	| 'AND'	| 'ANY'	| 'ARRAYLEN'	| 'AS'	| 'ASC'	| 'AUDIT'
	| 'BETWEEN'	| 'BY'
	| 'CASE'
	| 'CHAR'	| 'CHECK'	| 'CLUSTER'	| 'COLUMN'	| 'COMMENT'	| 'COMPRESS'	| 'CONNECT'	| 'CREATE'	| 'CURRENT'	
	| 'DATE'	| 'DECIMAL'	| 'DEFAULT'	| 'DELETE'	| 'DESC'	| 'DISTINCT'	| 'DROP'	
	| 'ELSE'	| 'EXCLUSIVE'	| 'EXISTS'	
	| 'FILE'	| 'FLOAT'	| 'FOR'	| 'FROM'	
	| 'GRANT'	| 'GROUP'	
	| 'HAVING'	
	| 'IDENTIFIED'	| 'IMMEDIATE'	| 'IN'	| 'INCREMENT'	| 'INDEX'	| 'INITIAL'	| 'INSERT'	| 'INTEGER'	
	| 'INTERSECT'	| 'INTO'	| 'IS'	
	| 'LEVEL'	| 'LIKE'	| 'LOCK'	| 'LONG'	
	| 'MAXEXTENTS'	| 'MINUS'	| 'MODE'	| 'MODIFY'	
	| 'NOAUDIT'	| 'NOCOMPRESS'	| 'NOT'	| 'NOTFOUND'	| 'NOWAIT'	| 'NULL'	| 'NUMBER'	
	| 'OF'	| 'OFFLINE'	| 'ON'	| 'ONLINE'	| 'OPTION'	| 'OR'	| 'ORDER'	
	| 'PCTFREE'	| 'PRIOR'	| 'PRIVILEGES'	| 'PUBLIC'	
	| 'RAW'	| 'RENAME'	| 'RESOURCE'	| 'REVOKE'	| 'ROW'	| 'ROWID'	| 'ROWLABEL'	| 'ROWNUM'	| 'ROWS'	
	| 'SELECT'	| 'SESSION'	| 'SET'	| 'SHARE'	| 'SIZE'	| 'SMALLINT'	| 'SQLBUF'	
	| 'START'	| 'SUCCESSFUL'	| 'SYNONYM'	| 'SYSDATE'	
	| 'TABLE'	| 'THEN'	| 'TO'	| 'TRIGGER'	
	| 'UID'	| 'UNION'	| 'UNIQUE'	| 'UPDATE'	| 'USER'	
	| 'VALIDATE'	| 'VALUES'	| 'VARCHAR'	| 'VARCHAR2'	| 'VIEW'	
	| 'WHENEVER'	| 'WHERE'	| 'WITH'
	) //{ $r.setType(T_RESERVED) }
	  //{ $type = T_RESERVED; }
	// -> ^(T_RESERVED[$r])
	;

/* ================================================================================
   Oracle keywords
   can by used for name database objects such as columns, tables, or indexes.
   ================================================================================ */
k_a  : r='A' { $r.setType(T_RESERVED) }  ;
k_at  : r='AT' { $r.setType(T_RESERVED) }  ;
k_admin : r='ADMIN' { $r.setType(T_RESERVED) }  ;
k_after : r='AFTER' { $r.setType(T_RESERVED) }  ;
k_allocate : r='ALLOCATE' { $r.setType(T_RESERVED) }  ;
k_analyze : r='ANALYZE' { $r.setType(T_RESERVED) }  ;
k_archive : r='ARCHIVE' { $r.setType(T_RESERVED) }  ;
k_archivelog : r='ARCHIVELOG' { $r.setType(T_RESERVED) }  ;
k_authorization : r='AUTHORIZATION' { $r.setType(T_RESERVED) }  ;
k_avg	 : r='AVG' { $r.setType(T_RESERVED) }  ;
k_backup : r='BACKUP' { $r.setType(T_RESERVED) }  ;
k_become : r='BECOME' { $r.setType(T_RESERVED) }  ;
k_before : r='BEFORE' { $r.setType(T_RESERVED) }  ;
k_begin : r='BEGIN' { $r.setType(T_RESERVED) }  ;
k_block : r='BLOCK' { $r.setType(T_RESERVED) }  ;
k_body	 : r='BODY' { $r.setType(T_RESERVED) }  ;
k_cache : r='CACHE' { $r.setType(T_RESERVED) }  ;
k_cancel : r='CANCEL' { $r.setType(T_RESERVED) }  ;
k_cascade : r='CASCADE' { $r.setType(T_RESERVED) }  ;
k_change : r='CHANGE' { $r.setType(T_RESERVED) }  ;
k_character : r='CHARACTER' { $r.setType(T_RESERVED) }  ;
k_checkpoint : r='CHECKPOINT' { $r.setType(T_RESERVED) }  ;
k_close	 : r='CLOSE' { $r.setType(T_RESERVED) }  ;
k_cobol : r='COBOL' { $r.setType(T_RESERVED) }  ;
k_commit : r='COMMIT' { $r.setType(T_RESERVED) }  ;
k_compile : r='COMPILE' { $r.setType(T_RESERVED) }  ;
k_constraint : r='CONSTRAINT' { $r.setType(T_RESERVED) }  ;
k_constraints : r='CONSTRAINTS' { $r.setType(T_RESERVED) }  ;
k_contents : r='CONTENTS' { $r.setType(T_RESERVED) }  ;
k_continue	 : r='CONTINUE' { $r.setType(T_RESERVED) }  ;
k_controlfile : r='CONTROLFILE' { $r.setType(T_RESERVED) }  ;
k_count : r='COUNT' { $r.setType(T_RESERVED) }  ;
k_cursor : r='CURSOR' { $r.setType(T_RESERVED) }  ;
k_cycle	 : r='CYCLE' { $r.setType(T_RESERVED) }  ;
k_database : r='DATABASE' { $r.setType(T_RESERVED) }  ;
k_datafile : r='DATAFILE' { $r.setType(T_RESERVED) }  ;
k_day : r='DAY' { $r.setType(T_RESERVED) }  ;
k_dba : r='DBA' { $r.setType(T_RESERVED) }  ;
k_dbtimezone : r='DBTIMEZONE' { $r.setType(T_RESERVED) }  ;
k_dec : r='DEC' { $r.setType(T_RESERVED) }  ;
k_declare : r='DECLARE' { $r.setType(T_RESERVED) }  ;
k_disable : r='DISABLE' { $r.setType(T_RESERVED) }  ;
k_dismount : r='DISMOUNT' { $r.setType(T_RESERVED) }  ;
k_double : r='DOUBLE' { $r.setType(T_RESERVED) }  ;
k_dump	 : r='DUMP' { $r.setType(T_RESERVED) }  ;
k_each : r='EACH' { $r.setType(T_RESERVED) }  ;
k_enable : r='ENABLE' { $r.setType(T_RESERVED) }  ;
k_end : r='END' { $r.setType(T_RESERVED) }  ;
k_escape : r='ESCAPE' { $r.setType(T_RESERVED) }  ;
k_events : r='EVENTS' { $r.setType(T_RESERVED) }  ;
k_except : r='EXCEPT' { $r.setType(T_RESERVED) }  ;
k_exceptions : r='EXCEPTIONS' { $r.setType(T_RESERVED) }  ;
k_exec : r='EXEC' { $r.setType(T_RESERVED) }  ;
k_execute	 : r='EXECUTE' { $r.setType(T_RESERVED) }  ;
k_explain : r='EXPLAIN' { $r.setType(T_RESERVED) }  ;
k_extent : r='EXTENT' { $r.setType(T_RESERVED) }  ;
k_externally	 : r='EXTERNALLY' { $r.setType(T_RESERVED) }  ;
k_fetch : r='FETCH' { $r.setType(T_RESERVED) }  ;
k_flush : r='FLUSH' { $r.setType(T_RESERVED) }  ;
k_force : r='FORCE' { $r.setType(T_RESERVED) }  ;
k_foreign : r='FOREIGN' { $r.setType(T_RESERVED) }  ;
k_fortran : r='FORTRAN' { $r.setType(T_RESERVED) }  ;
k_found : r='FOUND' { $r.setType(T_RESERVED) }  ;
k_freelist : r='FREELIST' { $r.setType(T_RESERVED) }  ;
k_freelists : r='FREELISTS' { $r.setType(T_RESERVED) }  ;
k_function	 : r='FUNCTION' { $r.setType(T_RESERVED) }  ;
k_go : r='GO' { $r.setType(T_RESERVED) }  ;
k_goto : r='GOTO' { $r.setType(T_RESERVED) }  ;
k_groups : r='GROUPS' { $r.setType(T_RESERVED) }  ;
k_including : r='INCLUDING' { $r.setType(T_RESERVED) }  ;
k_indicator : r='INDICATOR' { $r.setType(T_RESERVED) }  ;
k_initrans : r='INITRANS' { $r.setType(T_RESERVED) }  ;
k_instance : r='INSTANCE' { $r.setType(T_RESERVED) }  ;
k_int	 : r='INT' { $r.setType(T_RESERVED) }  ;
k_key	 : r='KEY' { $r.setType(T_RESERVED) }  ;
k_language : r='LANGUAGE' { $r.setType(T_RESERVED) }  ;
k_layer : r='LAYER' { $r.setType(T_RESERVED) }  ;
k_link : r='LINK' { $r.setType(T_RESERVED) }  ;
k_lists : r='LISTS' { $r.setType(T_RESERVED) }  ;
k_logfile	 : r='LOGFILE' { $r.setType(T_RESERVED) }  ;
k_local	 : r='LOCAL' { $r.setType(T_RESERVED) }  ;
k_locked	 : r='LOCKED' { $r.setType(T_RESERVED) }  ;
k_manage : r='MANAGE' { $r.setType(T_RESERVED) }  ;
k_manual : r='MANUAL' { $r.setType(T_RESERVED) }  ;
k_max : r='MAX' { $r.setType(T_RESERVED) }  ;
k_maxdatafiles : r='MAXDATAFILES' { $r.setType(T_RESERVED) }  ;
k_maxinstances : r='MAXINSTANCES' { $r.setType(T_RESERVED) }  ;
k_maxlogfiles : r='MAXLOGFILES' { $r.setType(T_RESERVED) }  ;
k_maxloghistory	 : r='MAXLOGHISTORY' { $r.setType(T_RESERVED) }  ;
k_maxlogmembers : r='MAXLOGMEMBERS' { $r.setType(T_RESERVED) }  ;
k_maxtrans : r='MAXTRANS' { $r.setType(T_RESERVED) }  ;
k_maxvalue : r='MAXVALUE' { $r.setType(T_RESERVED) }  ;
k_min : r='MIN' { $r.setType(T_RESERVED) }  ;
k_minextents : r='MINEXTENTS' { $r.setType(T_RESERVED) }  ;
k_minvalue : r='MINVALUE' { $r.setType(T_RESERVED) }  ;
k_module : r='MODULE' { $r.setType(T_RESERVED) }  ;
k_month	 : r='MONTH' { $r.setType(T_RESERVED) }  ;
k_mount	 : r='MOUNT' { $r.setType(T_RESERVED) }  ;
k_new : r='NEW' { $r.setType(T_RESERVED) }  ;
k_next : r='NEXT' { $r.setType(T_RESERVED) }  ;
k_noarchivelog : r='NOARCHIVELOG' { $r.setType(T_RESERVED) }  ;
k_nocache : r='NOCACHE' { $r.setType(T_RESERVED) }  ;
k_nocycle : r='NOCYCLE' { $r.setType(T_RESERVED) }  ;
k_nomaxvalue : r='NOMAXVALUE' { $r.setType(T_RESERVED) }  ;
k_nominvalue : r='NOMINVALUE' { $r.setType(T_RESERVED) }  ;
k_none	 : r='NONE' { $r.setType(T_RESERVED) }  ;
k_noorder : r='NOORDER' { $r.setType(T_RESERVED) }  ;
k_noresetlogs : r='NORESETLOGS' { $r.setType(T_RESERVED) }  ;
k_normal : r='NORMAL' { $r.setType(T_RESERVED) }  ;
k_nosort : r='NOSORT' { $r.setType(T_RESERVED) }  ;
k_numeric	 : r='NUMERIC' { $r.setType(T_RESERVED) }  ;
k_off : r='OFF' { $r.setType(T_RESERVED) }  ;
k_old : r='OLD' { $r.setType(T_RESERVED) }  ;
k_only : r='ONLY' { $r.setType(T_RESERVED) }  ;
k_open : r='OPEN' { $r.setType(T_RESERVED) }  ;
k_optimal : r='OPTIMAL' { $r.setType(T_RESERVED) }  ;
k_own	 : r='OWN' { $r.setType(T_RESERVED) }  ;
k_package : r='PACKAGE' { $r.setType(T_RESERVED) }  ;
k_parallel : r='PARALLEL' { $r.setType(T_RESERVED) }  ;
k_pctincrease : r='PCTINCREASE' { $r.setType(T_RESERVED) }  ;
k_pctused : r='PCTUSED' { $r.setType(T_RESERVED) }  ;
k_plan : r='PLAN' { $r.setType(T_RESERVED) }  ;
k_pli : r='PLI' { $r.setType(T_RESERVED) }  ;
k_precision : r='PRECISION' { $r.setType(T_RESERVED) }  ;
k_primary : r='PRIMARY' { $r.setType(T_RESERVED) }  ;
k_private : r='PRIVATE' { $r.setType(T_RESERVED) }  ;
k_procedure : r='PROCEDURE' { $r.setType(T_RESERVED) }  ;
k_profile	 : r='PROFILE' { $r.setType(T_RESERVED) }  ;
k_quota	 : r='QUOTA' { $r.setType(T_RESERVED) }  ;
k_read : r='READ' { $r.setType(T_RESERVED) }  ;
k_real : r='REAL' { $r.setType(T_RESERVED) }  ;
k_recover : r='RECOVER' { $r.setType(T_RESERVED) }  ;
k_references : r='REFERENCES' { $r.setType(T_RESERVED) }  ;
k_referencing : r='REFERENCING' { $r.setType(T_RESERVED) }  ;
k_resetlogs : r='RESETLOGS' { $r.setType(T_RESERVED) }  ;
k_restricted : r='RESTRICTED' { $r.setType(T_RESERVED) }  ;
k_reuse	 : r='REUSE' { $r.setType(T_RESERVED) }  ;
k_role : r='ROLE' { $r.setType(T_RESERVED) }  ;
k_roles : r='ROLES' { $r.setType(T_RESERVED) }  ;
k_rollback	 : r='ROLLBACK' { $r.setType(T_RESERVED) }  ;
k_savepoint : r='SAVEPOINT' { $r.setType(T_RESERVED) }  ;
k_schema : r='SCHEMA' { $r.setType(T_RESERVED) }  ;
k_scn : r='SCN' { $r.setType(T_RESERVED) }  ;
k_second : r='SECOND' { $r.setType(T_RESERVED) }  ;
k_section : r='SECTION' { $r.setType(T_RESERVED) }  ;
k_segment : r='SEGMENT' { $r.setType(T_RESERVED) }  ;
k_sequence : r='SEQUENCE' { $r.setType(T_RESERVED) }  ;
k_sessiontimezone : r='SESSIONTIMEZONE' { $r.setType(T_RESERVED) }  ;
k_shared : r='SHARED' { $r.setType(T_RESERVED) }  ;
k_snapshot	 : r='SNAPSHOT' { $r.setType(T_RESERVED) }  ;
k_skip : r='SKIP' { $r.setType(T_RESERVED) }  ;
k_some : r='SOME' { $r.setType(T_RESERVED) }  ;
k_sort : r='SORT' { $r.setType(T_RESERVED) }  ;
k_sql : r='SQL' { $r.setType(T_RESERVED) }  ;
k_sqlcode : r='SQLCODE' { $r.setType(T_RESERVED) }  ;
k_sqlerror : r='SQLERROR' { $r.setType(T_RESERVED) }  ;
k_sqlstate : r='SQLSTATE' { $r.setType(T_RESERVED) }  ;
k_statement_ID : r='STATEMENT' { $r.setType(T_RESERVED) }  ;
k_statistics : r='STATISTICS' { $r.setType(T_RESERVED) }  ;
k_stop : r='STOP' { $r.setType(T_RESERVED) }  ;
k_storage : r='STORAGE' { $r.setType(T_RESERVED) }  ;
k_sum : r='SUM' { $r.setType(T_RESERVED) }  ;
k_switch : r='SWITCH' { $r.setType(T_RESERVED) }  ;
k_system	 : r='SYSTEM' { $r.setType(T_RESERVED) }  ;
k_tables : r='TABLES' { $r.setType(T_RESERVED) }  ;
k_tablespace : r='TABLESPACE' { $r.setType(T_RESERVED) }  ;
k_temporary : r='TEMPORARY' { $r.setType(T_RESERVED) }  ;
k_thread : r='THREAD' { $r.setType(T_RESERVED) }  ;
k_time : r='TIME' { $r.setType(T_RESERVED) }  ;
k_tracing : r='TRACING' { $r.setType(T_RESERVED) }  ;
k_transaction : r='TRANSACTION' { $r.setType(T_RESERVED) }  ;
k_triggers	 : r='TRIGGERS' { $r.setType(T_RESERVED) }  ;
k_truncate	 : r='TRUNCATE' { $r.setType(T_RESERVED) }  ;
k_under : r='UNDER' { $r.setType(T_RESERVED) }  ;
k_unlimited : r='UNLIMITED' { $r.setType(T_RESERVED) }  ;
k_until : r='UNTIL' { $r.setType(T_RESERVED) }  ;
k_use : r='USE' { $r.setType(T_RESERVED) }  ;
k_using	 : r='USING' { $r.setType(T_RESERVED) }  ;
k_wait : r='WAIT' { $r.setType(T_RESERVED) }  ;
k_when : r='WHEN' { $r.setType(T_RESERVED) }  ;
k_work : r='WORK' { $r.setType(T_RESERVED) }  ;
k_write	 : r='WRITE' { $r.setType(T_RESERVED) }  ;
k_year	 : r='YEAR' { $r.setType(T_RESERVED) }  ;
k_zone	 : r='ZONE' { $r.setType(T_RESERVED) }  ;

k_automatic : r='AUTOMATIC' { $r.setType(T_RESERVED) }  ;
k_bfile : r='BFILE' { $r.setType(T_RESERVED) }  ;
k_binary_double : r='BINARY_DOUBLE' { $r.setType(T_RESERVED) }  ;
k_binary_float : r='BINARY_FLOAT' { $r.setType(T_RESERVED) }  ;
k_binary_integer : r='BINARY_INTEGER' { $r.setType(T_RESERVED) }  ;
k_blob : r='BLOB' { $r.setType(T_RESERVED) }  ;
k_boolean : r='BOOLEAN' { $r.setType(T_RESERVED) }  ;
k_byte : r='BYTE' { $r.setType(T_RESERVED) }  ;
k_cast : r='CAST' { $r.setType(T_RESERVED) }  ;
k_clob : r='CLOB' { $r.setType(T_RESERVED) }  ;
k_cluster_set : r='CLUSTER_SET' { $r.setType(T_RESERVED) }  ;
k_column_value : r='COLUMN_VALUE' { $r.setType(T_RESERVED) }  ;
k_connect_by_iscycle : r='CONNECT_BY_ISCYCLE' { $r.setType(T_RESERVED) }  ;
k_connect_by_isleaf : r='CONNECT_BY_ISLEAF' { $r.setType(T_RESERVED) }  ;
k_connect_by_root : r='CONNECT_BY_ROOT' { $r.setType(T_RESERVED) }  ;
k_corr : r='CORR' { $r.setType(T_RESERVED) }  ;
k_covar_pop : r='COVAR_POP' { $r.setType(T_RESERVED) }  ;
k_covar_samp : r='COVAR_SAMP' { $r.setType(T_RESERVED) }  ;
k_cross : r='CROSS' { $r.setType(T_RESERVED) }  ;
k_cube : r='CUBE' { $r.setType(T_RESERVED) }  ;
k_cume_dist : r='CUME_DIST' { $r.setType(T_RESERVED) }  ;
k_decrement : r='DECREMENT' { $r.setType(T_RESERVED) }  ;
k_dense_rank : r='DENSE_RANK' { $r.setType(T_RESERVED) }  ;
k_dimension : r='DIMENSION' { $r.setType(T_RESERVED) }  ;
k_empty : r='EMPTY' { $r.setType(T_RESERVED) }  ;
k_equals_path : r='EQUALS_PATH' { $r.setType(T_RESERVED) }  ;
k_first_value : r='FIRST_VALUE' { $r.setType(T_RESERVED) }  ;
k_full : r='FULL' { $r.setType(T_RESERVED) }  ;
k_grouping : r='GROUPING' { $r.setType(T_RESERVED) }  ;
k_ignore : r='IGNORE' { $r.setType(T_RESERVED) }  ;
k_infinite : r='INFINITE' { $r.setType(T_RESERVED) }  ;
k_inner : r='INNER' { $r.setType(T_RESERVED) }  ;
k_interval : r='INTERVAL' { $r.setType(T_RESERVED) }  ;
k_iterate : r='ITERATE' { $r.setType(T_RESERVED) }  ;
k_join : r='JOIN' { $r.setType(T_RESERVED) }  ;
k_keep : r='KEEP' { $r.setType(T_RESERVED) }  ;
k_lag : r='LAG' { $r.setType(T_RESERVED) }  ;
k_last : r='LAST' { $r.setType(T_RESERVED) }  ;
k_last_value : r='LAST_VALUE' { $r.setType(T_RESERVED) }  ;
k_lead : r='LEAD' { $r.setType(T_RESERVED) }  ;
k_left : r='LEFT' { $r.setType(T_RESERVED) }  ;
k_main : r='MAIN' { $r.setType(T_RESERVED) }  ;
k_measures : r='MEASURES' { $r.setType(T_RESERVED) }  ;
k_member : r='MEMBER' { $r.setType(T_RESERVED) }  ;
k_mlslabel : r='MLSLABEL' { $r.setType(T_RESERVED) }  ;
k_model : r='MODEL' { $r.setType(T_RESERVED) }  ;
k_multiset : r='MULTISET' { $r.setType(T_RESERVED) }  ;
k_nan : r='NAN' { $r.setType(T_RESERVED) }  ;
k_national : r='NATIONAL' { $r.setType(T_RESERVED) }  ;
k_natural : r='NATURAL' { $r.setType(T_RESERVED) }  ;
k_nav : r='NAV' { $r.setType(T_RESERVED) }  ;
k_nchar : r='NCHAR' { $r.setType(T_RESERVED) }  ;
k_nclob : r='NCLOB' { $r.setType(T_RESERVED) }  ;
k_ntile : r='NTILE' { $r.setType(T_RESERVED) }  ;
k_nulls : r='NULLS' { $r.setType(T_RESERVED) }  ;
k_nvarchar : r='NVARCHAR' { $r.setType(T_RESERVED) }  ;
k_nvarchar2 : r='NVARCHAR2' { $r.setType(T_RESERVED) }  ;
k_object_id : r='OBJECT_ID' { $r.setType(T_RESERVED) }  ;
k_object_value : r='OBJECT_VALUE' { $r.setType(T_RESERVED) }  ;
k_ora_rowscn : r='ORA_ROWSCN' { $r.setType(T_RESERVED) }  ;
k_outer : r='OUTER' { $r.setType(T_RESERVED) }  ;
k_over : r='OVER' { $r.setType(T_RESERVED) }  ;
k_partition : r='PARTITION' { $r.setType(T_RESERVED) }  ;
k_percentile_cont : r='PERCENTILE_CONT' { $r.setType(T_RESERVED) }  ;
k_percentile_disc : r='PERCENTILE_DISC' { $r.setType(T_RESERVED) }  ;
k_percent_rank : r='PERCENT_RANK' { $r.setType(T_RESERVED) }  ;
k_pivot : r='PIVOT' { $r.setType(T_RESERVED) }  ;
k_pls_integer : r='PLS_INTEGER' { $r.setType(T_RESERVED) }  ;
k_positive : r='POSITIVE' { $r.setType(T_RESERVED) }  ;
k_present : r='PRESENT' { $r.setType(T_RESERVED) }  ;
k_rank : r='RANK' { $r.setType(T_RESERVED) }  ;
k_ratio_to_report : r='RATIO_TO_REPORT' { $r.setType(T_RESERVED) }  ;
k_reference : r='REFERENCE' { $r.setType(T_RESERVED) }  ;
k_regexp_like : r='REGEXP_LIKE' { $r.setType(T_RESERVED) }  ;
k_regr_avgx : r='REGR_AVGX' { $r.setType(T_RESERVED) }  ;
k_regr_avgy : r='REGR_AVGY' { $r.setType(T_RESERVED) }  ;
k_regr_count : r='REGR_COUNT' { $r.setType(T_RESERVED) }  ;
k_regr_intercept : r='REGR_INTERCEPT' { $r.setType(T_RESERVED) }  ;
k_regr_r2 : r='REGR_R2' { $r.setType(T_RESERVED) }  ;
k_regr_slope : r='REGR_SLOPE' { $r.setType(T_RESERVED) }  ;
k_regr_sxx : r='REGR_SXX' { $r.setType(T_RESERVED) }  ;
k_regr_sxy : r='REGR_SXY' { $r.setType(T_RESERVED) }  ;
k_regr_syy : r='REGR_SYY' { $r.setType(T_RESERVED) }  ;
k_right : r='RIGHT' { $r.setType(T_RESERVED) }  ;
k_rollup : r='ROLLUP' { $r.setType(T_RESERVED) }  ;
k_row_number : r='ROW_NUMBER' { $r.setType(T_RESERVED) }  ;
k_rules : r='RULES' { $r.setType(T_RESERVED) }  ;
k_sample : r='SAMPLE' { $r.setType(T_RESERVED) }  ;
k_search : r='SEARCH' { $r.setType(T_RESERVED) }  ;
k_sequential : r='SEQUENTIAL' { $r.setType(T_RESERVED) }  ;
k_sets : r='SETS' { $r.setType(T_RESERVED) }  ;
k_single : r='SINGLE' { $r.setType(T_RESERVED) }  ;
k_stddev : r='STDDEV' { $r.setType(T_RESERVED) }  ;
k_stddev_pop : r='STDDEV_POP' { $r.setType(T_RESERVED) }  ;
k_stddev_samp : r='STDDEV_SAMP' { $r.setType(T_RESERVED) }  ;
k_submultiset : r='SUBMULTISET' { $r.setType(T_RESERVED) }  ;
k_subpartition : r='SUBPARTITION' { $r.setType(T_RESERVED) }  ;
k_the : r='THE' { $r.setType(T_RESERVED) }  ;
k_timestamp : r='TIMESTAMP' { $r.setType(T_RESERVED) }  ;
k_type : r='TYPE' { $r.setType(T_RESERVED) }  ;
k_unbounded : r='UNBOUNDED' { $r.setType(T_RESERVED) }  ;
k_under_path : r='UNDER_PATH' { $r.setType(T_RESERVED) }  ;
k_updated : r='UPDATED' { $r.setType(T_RESERVED) }  ;
k_upsert : r='UPSERT' { $r.setType(T_RESERVED) }  ;
k_urowid : r='UROWID' { $r.setType(T_RESERVED) }  ;
k_variance : r='VARIANCE' { $r.setType(T_RESERVED) }  ;
k_varying : r='VARYING' { $r.setType(T_RESERVED) }  ;
k_var_pop : r='VAR_POP' { $r.setType(T_RESERVED) }  ;
k_var_samp : r='VAR_SAMP' { $r.setType(T_RESERVED) }  ;
k_versions_endscn : r='VERSIONS_ENDSCN' { $r.setType(T_RESERVED) }  ;
k_versions_endtime : r='VERSIONS_ENDTIME' { $r.setType(T_RESERVED) }  ;
k_versions_operation : r='VERSIONS_OPERATION' { $r.setType(T_RESERVED) }  ;
k_versions_starscn : r='VERSIONS_STARSCN' { $r.setType(T_RESERVED) }  ;
k_versions_starttime : r='VERSIONS_STARTTIME' { $r.setType(T_RESERVED) }  ;
k_versions_xid : r='VERSIONS_XID' { $r.setType(T_RESERVED) }  ;
k_xml : r='XML' { $r.setType(T_RESERVED) }  ;
k_xmldata : r='XMLDATA' { $r.setType(T_RESERVED) }  ;

k_errors : r='ERRORS' { $r.setType(T_RESERVED) }  ;
k_first : r='FIRST' { $r.setType(T_RESERVED) }  ;
k_limit : r='LIMIT' { $r.setType(T_RESERVED) }  ;
k_log : r='LOG' { $r.setType(T_RESERVED) }  ;
k_reject : r='REJECT' { $r.setType(T_RESERVED) }  ;
k_return : r='RETURN' { $r.setType(T_RESERVED) }  ;
k_returning : r='RETURNING' { $r.setType(T_RESERVED) }  ;

k_merge : r='MERGE' { $r.setType(T_RESERVED) }  ;
k_matched : r='MATCHED' { $r.setType(T_RESERVED) }  ;

k_following : r='FOLLOWING' { $r.setType(T_RESERVED) }  ;
k_range : r='RANGE' { $r.setType(T_RESERVED) }  ;
k_siblings : r='SIBLINGS' { $r.setType(T_RESERVED) }  ;
k_unpivot : r='UNPIVOT' { $r.setType(T_RESERVED) }  ;

k_value : r= 'VALUE' { $r.setType(T_RESERVED) }  ;

k_breadth : r='BREADTH' { $r.setType(T_RESERVED) }  ;
k_depth : r='DEPTH' { $r.setType(T_RESERVED) }  ;
k_exclude : r='EXCLUDE' { $r.setType(T_RESERVED) }  ;
k_include : r='INCLUDE' { $r.setType(T_RESERVED) }  ;
k_mivalue : r='MIVALUE' { $r.setType(T_RESERVED) }  ;
k_preceding : r='PRECEDING' { $r.setType(T_RESERVED) }  ;
k_respect : r='RESPECT' { $r.setType(T_RESERVED) }  ;
k_seed : r='SEED' { $r.setType(T_RESERVED) }  ;
k_versions : r='VERSIONS' { $r.setType(T_RESERVED) }  ;

keyword
	: 'A' // note: this one is not listed in the docs but is a part of "IS A SET" condition clause
	| 'ADMIN'	| 'AFTER'	| 'ALLOCATE'	| 'ANALYZE'	| 'ARCHIVE'	| 'ARCHIVELOG'	| 'AT'	| 'AUTHORIZATION'	| 'AVG'	
	| 'BACKUP'	| 'BECOME'	| 'BEFORE'	| 'BEGIN'	| 'BLOCK'	| 'BODY'	| 'BREADTH'
	| 'CACHE'	| 'CANCEL'	| 'CASCADE'	| 'CHANGE'	| 'CHARACTER'	| 'CHECKPOINT'	| 'CLOSE'	
	| 'COBOL'	| 'COMMIT'	| 'COMPILE'	| 'CONSTRAINT'	| 'CONSTRAINTS'	| 'CONTENTS'	| 'CONTINUE'	
	| 'CONTROLFILE'	| 'COUNT'	| 'CURSOR'	| 'CYCLE'	
	| 'DATABASE'	| 'DATAFILE'	| 'DAY'	| 'DBA'	| 'DBTIMEZONE'	| 'DEC'	| 'DECLARE'	| 'DISABLE'	| 'DISMOUNT'	| 'DOUBLE'	| 'DUMP'
    | 'DEPTH'
	| 'EACH'	| 'ENABLE'	| 'END'	| 'ESCAPE'	| 'EVENTS'	| 'ERRORS'	| 'EXCEPT'	| 'EXCEPTIONS'	| 'EXEC'	| 'EXECUTE'
    | 'EXCLUDE'
	| 'EXPLAIN'	| 'EXTENT'	| 'EXTERNALLY'	
	| 'FETCH'	| 'FIRST'	| 'FLUSH'	| 'FORCE'	| 'FOREIGN'	| 'FORTRAN'	| 'FOUND'
	| 'FOLLOWING'    | 'FREELIST'
	| 'FREELISTS'	| 'FUNCTION'	
	| 'GO'	| 'GOTO'	| 'GROUPS'
	| 'INCLUDE'	| 'INCLUDING'	| 'INDICATOR'	| 'INITRANS'	| 'INSTANCE'	| 'INT'	
	| 'KEY'	
	| 'LANGUAGE'	| 'LAYER'	| 'LIMIT'	| 'LINK'	| 'LISTS'	| 'LOCAL'	| 'LOCKED'	| 'LOG'	| 'LOGFILE'
	| 'MANAGE'	| 'MANUAL'	| 'MATCHED'	| 'MAX'	| 'MAXDATAFILES'	| 'MAXINSTANCES'	| 'MAXLOGFILES'	| 'MAXLOGHISTORY'	
	| 'MAXLOGMEMBERS'	| 'MAXTRANS'	| 'MAXVALUE'	| 'MERGE'	| 'MIN'	| 'MINEXTENTS'	| 'MINVALUE'	| 'MODULE'	| 'MONTH'	| 'MOUNT'
	| 'NEW'	| 'NEXT'	| 'NOARCHIVELOG'	| 'NOCACHE'	| 'NOCYCLE'	| 'NOMAXVALUE'	| 'NOMINVALUE'	| 'NONE' 
	| 'NOORDER'	| 'NORESETLOGS'	| 'NORMAL'	| 'NOSORT'	| 'NUMERIC'	
	| 'OFF'	| 'OLD'	| 'ONLY'	| 'OPEN'	| 'OPTIMAL'	| 'OWN'	
	| 'PACKAGE'	| 'PARALLEL'	| 'PCTINCREASE'	| 'PCTUSED'	| 'PLAN'	| 'PLI'	| 'PRECISION'	| 'PRIMARY'	
	| 'PRIVATE'	| 'PRECEDING'	| 'PROCEDURE'	| 'PROFILE'	
	| 'QUOTA'	
	| 'READ'	| 'REAL'	| 'RECOVER'	| 'REFERENCES'	| 'REFERENCING'	| 'REJECT'	| 'RETURN'	| 'RETURNING'
	| 'RESETLOGS'	| 'RESTRICTED'	| 'REUSE'	| 'ROLE'	| 'ROLES'	| 'ROLLBACK'
    | 'RESPECT'
	| 'SAVEPOINT'	| 'SECOND'	| 'SESSIONTIMEZONE'	| 'SCHEMA'	| 'SCN'	| 'SECTION'	| 'SEGMENT'	| 'SEQUENCE'
	| 'SHARED'	| 'SKIP'	| 'SNAPSHOT'	| 'SOME'	| 'SORT'	| 'SQL'	| 'SQLCODE'	| 'SQLERROR'	| 'SQLSTATE'
	| 'STATEMENT_ID'	| 'STATISTICS'	| 'STOP'	| 'STORAGE'	| 'SUM'	| 'SWITCH'	| 'SYSTEM'
    | 'SEED'
	| 'TABLES'	| 'TABLESPACE'	| 'TEMPORARY'	| 'THREAD'	| 'TIME'	| 'TRACING'	| 'TRANSACTION'	| 'TRIGGERS'	
	| 'TRUNCATE'	
	| 'UNDER'	| 'UNLIMITED'	| 'UNTIL'	| 'USE'
    | 'VALUE'	| 'VERSIONS'
	| 'WHEN'	| 'WORK'	| 'WRITE'
   	| 'YEAR'
	| 'ZONE'
    | 'AUTOMATIC'	| 'BFILE'	| 'BINARY_DOUBLE'
    | 'BINARY_FLOAT'	| 'BINARY_INTEGER'	| 'BLOB'	| 'BOOLEAN'
    | 'BYTE'	| 'CAST'	| 'CLOB'	| 'CLUSTER_SET'
    | 'COLUMN_VALUE'
    | 'CONNECT_BY_ISCYCLE'	| 'CONNECT_BY_ISLEAF'	| 'CONNECT_BY_ROOT'
    | 'CORR'	| 'COVAR_POP'	| 'COVAR_SAMP'	| 'CROSS'
    | 'CUBE'	| 'CUME_DIST'	| 'DECREMENT'	| 'DENSE_RANK'
    | 'DIMENSION'	| 'EMPTY'	|    'EQUALS_PATH'	| 'FIRST_VALUE'
    | 'FULL'	| 'GROUPING'	| 'IGNORE'	| 'INFINITE'
    | 'INNER'
    | 'INTERVAL'	| 'ITERATE'
    | 'KEEP'	| 'LAG'	| 'LAST'	| 'LAST_VALUE'
    | 'LEAD'	| 'LEFT'	| 'MAIN'	| 'MEASURES'
    | 'MEMBER'	| 'MLSLABEL'
    | 'NAN'	| 'NATIONAL'	| 'NATURAL'	| 'NAV'
    | 'NCHAR'	| 'NCLOB'	| 'NTILE'	| 'NULLS'
    | 'NVARCHAR'	| 'NVARCHAR2'	| 'OBJECT_ID'	|    'OBJECT_VALUE'
    | 'ORA_ROWSCN'
    | 'OVER'
    | 'PIVOT'
    | 'PLS_INTEGER'	| 'POSITIVE'	| 'PRESENT'
    | 'RANGE'    | 'RANK'
    | 'RATIO_TO_REPORT'	| 'REFERENCE'	| 'REGEXP_LIKE'	| 'REGR_AVGX'
    | 'REGR_AVGY'	| 'REGR_COUNT'	| 'REGR_INTERCEPT'	| 'REGR_R2'
    | 'REGR_SLOPE'	| 'REGR_SXX'	| 'REGR_SXY'	| 'REGR_SYY'
    | 'RIGHT'	| 'ROLLUP'	| 'ROW_NUMBER'	| 'RULES'
    | 'SAMPLE'	| 'SEARCH'	| 'SEQUENTIAL'	| 'SETS'
    | 'SIBLINGS'
    | 'SINGLE'	| 'STDDEV'	| 'STDDEV_POP'	| 'STDDEV_SAMP'
    | 'SUBMULTISET'	| 'SUBPARTITION'	| 'THE'	| 'TIMESTAMP'
    | 'TYPE'	| 'UNBOUNDED'	| 'UNDER_PATH'	| 'UNPIVOT'
    | 'UPDATED'
    | 'UPSERT'	| 'UROWID'	| 'VARIANCE'	| 'VARYING'
    | 'VAR_POP'	| 'VAR_SAMP'
    | 'VERSIONS_XID'
    | 'XML'        
    | 'XMLDATA'        
	;

quoted_string
	:	QUOTED_STRING
	;

QUOTED_STRING
	:	( 'n'|'N' )? '\'' ( '\'\'' | ~('\'') )* '\''
	;


ID /*options { testLiterals=true; }*/
    :	'A' .. 'Z' ( 'A' .. 'Z' | '0' .. '9' | '_' | '$' | '#' )*
    |	DOUBLEQUOTED_STRING
    ;
SEMI
	:	';'
	;
COLON
	:	':'
	;
DOUBLEDOT
	:	POINT POINT
	;
DOT
	:	POINT
	;
fragment
POINT
	:	'.'
	;
COMMA 
	:	','
	;
EXPONENT
	:	'**'
	;
ASTERISK
	:	'*'
	;
AT_SIGN
	:	'@'
	;
RPAREN
	:	')'
	;
LPAREN
	:	'('
	;
RBRACK
	:	']'
	;
LBRACK
	:	'['
	;
PLUS
	:	'+'
	;
MINUS
	:	'-'
	;
DIVIDE
	:	'/'
	;
EQ
	:	'='
	;
PERCENTAGE
	:	'%'
	;
LLABEL
	:	'<<'
	;
RLABEL
	:	'>>'
	;
ASSIGN
	:	':='
	;
ARROW
	:	'=>'
	;
VERTBAR
	:	'|'
	;
DOUBLEVERTBAR
	:	'||'
	;
NOT_EQ
	:	'<>' | '!=' | '^='
	;
LTH
	:	'<'
	;
LEQ
	:	'<='
	;
GTH
	:	'>'
	;
GEQ
	:	'>='
	;
NUMBER
	:	//( PLUS | MINUS )?
		(	( NUM POINT NUM ) => NUM POINT NUM
		|	POINT NUM
		|	NUM
		)
		( 'E' ( PLUS | MINUS )? NUM )?
    ;
fragment
NUM
	: '0' .. '9' ( '0' .. '9' )*
	;
QUOTE
	:	'\''
	;
fragment
DOUBLEQUOTED_STRING
	:	'"' ( ~('"') )* '"'
	;
WS	:	(' '|'\r'|'\t'|'\n') {$channel=HIDDEN;}
	;
SL_COMMENT
	:	'--' ~('\n'|'\r')* '\r'? '\n' {$channel=HIDDEN;}
	;
ML_COMMENT
	:	'/*' ( options {greedy=false;} : . )* '*/' {$channel=HIDDEN;}
	;
TYPE_ATTR
	:	'%TYPE'
	;
ROWTYPE_ATTR
	:	'%ROWTYPE'
	;
NOTFOUND_ATTR
	:	'%NOTFOUND'
	;
FOUND_ATTR
	:	'%FOUND'
	;
ISOPEN_ATTR
	:	'%ISOPEN'
	;
ROWCOUNT_ATTR
	:	'%ROWCOUNT'
	;
BULK_ROWCOUNT_ATTR
	:	'%BULK_ROWCOUNT'
	;
CHARSET_ATTR
	:	'%CHARSET'
	;

