import numpy as np
import pandas as pd
import copy

# name of file being read
data_name = "data_small.csv"
# list of all "bad" indices, indices for data we can't use
bad_ind = [] 


# read call numbers
call_nums=np.genfromtxt(data_name, dtype=long, delimiter=",",skip_header=1, usecols=0)
# gets index of duplicate call numbers
# this just keeps the first one and deletes the rest, we should
# think about how to determine which to keep
print call_nums
ret0,ret1 = np.unique(call_nums,return_index=True)
unique = np.isin(range(0,20),ret1)
for i in range(0,unique.size):
    if not unique[i]:
        bad_ind.append(i)
print call_nums


# read call types (as strings)
call_types_str = np.genfromtxt(data_name, dtype=str, delimiter = ",", skip_header=1, usecols=3)
# convert call types to a numerical list
# 0 - Medical Incident
# 1 - Alarm
# 2 - Structure Fire
# 3 - Traffic Collision
# 4 - Other (will be deleted later)
call_types = []
for i in range(0,call_types_str.size):
    st = call_types_str[i]
    if st=="Medical Incident":
        call_types.append(0)
    elif st=="Alarms":
        call_types.append(1)
    elif st=="Structure Fire":
        call_types.append(2)
    elif st=="Traffic Collision":
        call_types.append(3)
    else:
        call_types.append(4)
        bad_ind.append(i)
call_types = np.array(call_types)


# combine it all into one big array!
data = np.column_stack((call_nums,call_types))
data = np.core.records.fromarrays(data.transpose(),
                                    names="callNum, callType",
                                    formats = "uint32, uint8")

# remove the "bad" indices
bad_ind = np.unique(bad_ind)
data = np.delete(data,bad_ind)


#this will sort data by call number, not sure if this is needed
# sort = np.sort(data,axis=0,order=['callNum'])
# print sort


# write cleaned data to a csv file (so that we only have to clean once)
# csv out file will be filename_clean.csv
data_out_name = data_name[:-4]+"_clean.csv"
w_data = pd.DataFrame(data)
w_data.to_csv(data_out_name,header=None, index=None)







