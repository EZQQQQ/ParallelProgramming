import random
import os

row_num = 65
col_num = 64

maximum_limit = 100.0

output_directory = "/nfsmnt/123100001/CSC4005-2023Fall/project2/matrices"
output_file = os.path.join(output_directory, "matrix.txt")

with open(output_file, "w") as matrix_file:
    matrix_file.write(f"{row_num} {col_num}\n")
    for i in range(row_num):
        for j in range(col_num):
            random_number = int(random.uniform(0.0, maximum_limit))
            if j < col_num - 1:
                matrix_file.write(f"{random_number} ")
            else:
                matrix_file.write(f"{random_number}\n")
