import numpy as np
import copy


# list of all "bad" indices, indices for data we can't use
bad_ind = [] 


call_nums=np.genfromtxt("data_small.csv", dtype=long, delimiter=",",skip_header=1, usecols=0)
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



call_types_str = np.genfromtxt("data_small.csv", dtype=str, delimiter = ",", skip_header=1, usecols=3)

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
#    elif st=="Traffic Collision":
#        call_types.append(3)
    else:
        call_types.append(4)
        bad_ind.append(i)
call_types = np.array(call_types)

data = np.column_stack((call_nums,call_types))
print data

data = np.core.records.fromarrays(data.transpose(),
                                    names="callNum, callType",
                                    formats = "uint32, uint8")

bad_ind = np.unique(bad_ind)
data = np.delete(data,bad_ind)

print data

#this will sort data by call number, not sure if this is needed
# sort = np.sort(data,axis=0,order=['callNum'])
# print sort
