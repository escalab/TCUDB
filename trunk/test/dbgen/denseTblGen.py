import argparse
import random

parser = argparse.ArgumentParser(description='Generate two NxN tables to form dense matrices for msplitm.')
parser.add_argument('--dim', default='16', type=int, help='dimension of the table')
parser.add_argument('--tbl1', default='A', help='first output table')
parser.add_argument('--tbl2', default='B', help='second output table')

args = parser.parse_args()

# generate NxN table inputs
def generate_table(tbl_name, dimension):
    tbl_name = tbl_name+"_"+str(dimension)+"x"+str(dimension)+".tbl"
    with open(tbl_name, 'w') as f:
        
        for row in range(dimension):
            for col in range(dimension):
                f.write("{0}|{1}|{2}|\n".format(row, col, random.getrandbits(8)))
        f.close()

generate_table(args.tbl1, args.dim)
generate_table(args.tbl2, args.dim)

print("Finished.")
