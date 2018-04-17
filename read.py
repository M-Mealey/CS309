import numpy as np
import copy

call_nums=np.genfromtxt("data_small.csv", dtype=long, delimiter=",",skip_header=1, usecols=0)
call_types = np.genfromtxt("data_small.csv", dtype=str, delimiter = ",", skip_header=1, usecols=3)

print call_nums
print call_types
