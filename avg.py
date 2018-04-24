import numpy as np
import pandas as pd
import config

# name of file being read
data_name = "toAverage.csv"
# list of all "bad" indices, indices for data we can't use
bad_ind = [] 

# read call numbers
call_nums=np.genfromtxt(data_name, dtype=long, delimiter=",",skip_header=1, usecols=0)
resp_time=np.genfromtxt(data_name, dtype=float, delimiter=",",skip_header=1, usecols=1)




# combine it all into one big array!
data = np.column_stack((call_nums,resp_time))
data = np.core.records.fromarrays(data.transpose(),
                                    names="callNum, respTime",
                                    formats = "uint32, float64")





for i in range(0,data.shape[0]):
    r_times = []
    if np.isnan(data[i][1]):
        r_times.append(0)
    else: 
        r_times.append(data[i][1])
    call = data[i][0]
    if data[i-1][0]==call:
        data[i][1]=data[i-1][1]
        continue
    while i+1<data.shape[0] and data[i+len(r_times)][0]==call:
        if np.isnan(data[i+len(r_times)][1]):
            r_times.append(0)
        else:
            r_times.append(data[i+len(r_times)][1])
    for j in range(0,len(r_times)):
        if r_times[j]<0:
            r_times[j]=0
    avg_res = 0;
    if np.sum(r_times)>0:
        while np.isin(0,r_times):
            r_times.remove(0)
        avg_res=np.average(r_times)
    data[i][1] = avg_res


# write cleaned data to a csv file (so that we only have to clean once)
# csv out file will be filename_clean.csv
data_out_name = data_name[:-4]+"_clean.csv"
w_data = pd.DataFrame(data)
w_data.to_csv(data_out_name, index=None)

print "done! :)"







