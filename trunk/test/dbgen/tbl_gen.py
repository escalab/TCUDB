import numpy as np
from scipy.sparse import coo_matrix
import argparse

parser = argparse.ArgumentParser(description='Generate two NxN tables for TCUDB.')
parser.add_argument('--dim', default='16', type=int, help='dimension of the table')
parser.add_argument('--tbl1', default='large_mat1.tbl', help='first output table')
parser.add_argument('--tbl2', default='large_mat2.tbl', help='second output table')

args = parser.parse_args()
global MAT1, MAT2

# create two 2D representation of the matrix
rand_max = pow(2, 14) # threshold of max value
MAT1 = np.random.randint(0, rand_max, (args.dim, args.dim))
MAT2 = np.random.randint(0, rand_max, (args.dim, args.dim))

# randomly replace 3 elements in each row to zero
def replace2Zero(mat, how_many_to_zero=3):
    for row in mat:
        indices = np.random.choice(np.arange(row.size), replace=False, size=int(how_many_to_zero))
        row[indices] = 0

# generate text output for the verification
def write_reference(mat, num, sparse_mat):
    filename = "output_mat"+str(num)+".txt"
    with open(filename, 'w') as f:
        f.write("Input matrix:\n")
        np.savetxt(f, mat, fmt='% 5d', delimiter=' ', newline='\n')
        f.write("Sparse matrix:\n")
        for row, col, val in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            f.write("{0}|{1}|{2}|\n".format(row, col, val))
        f.close()

# generate tbl for TCUDB
def generate_table(tbl_name, sparse_mat):
    with open(tbl_name, 'w') as f:
        for row, col, val in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            f.write("{0}|{1}|{2}|\n".format(row, col, val))
        f.close()

replace2Zero(MAT1)
replace2Zero(MAT2)

'''
print("Matrix 1:")
print(MAT1)
print("\n")
print("Matrix 2:")
print(MAT2)
'''

# convert to sparse matrix representation 
S1, S2 = coo_matrix(MAT1), coo_matrix(MAT2)
'''
print("Sparse Matrix 1:")
print(S1)
print("\n")
print("Sparse Matrix 2:")
print(S2)
'''

# convert back to 2-D representation of the matrix
B1, B2 = S1.todense(), S2.todense()
'''
print("Dense Matrix 1:")
print(B1)
print("\n")
print("Dense Matrix 2:")
print(B2)
'''

'''
# generate text output for the verification
def write_reference(mat, num, sparse_mat):
    filename = "output_mat"+str(num)+".txt"
    with open(filename, 'w') as f:
        f.write("Input matrix:\n")
        np.savetxt(f, mat, fmt='% 5d', delimiter=' ', newline='\n')
        f.write("Sparse matrix:\n")
        for row, col, val in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            f.write("{0}|{1}|{2}|\n".format(row, col, val))
        f.close()

# generate tbl for TCUDB
def generate_table(tbl_name, sparse_mat):
    with open(tbl_name, 'w') as f:
        for row, col, val in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            f.write("{0}|{1}|{2}|\n".format(row, col, val))
        f.close()
'''

write_reference(MAT1, 1, S1)
write_reference(MAT2, 2, S2)

generate_table(args.tbl1, S1)
generate_table(args.tbl2, S2)

print("Finished.")
