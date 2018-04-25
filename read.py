import numpy as np
import pandas as pd
import config

# name of file being read
data_name = "data.csv"
# list of all "bad" indices, indices for data we can't use
bad_ind = [] 

def tsToNum(s):
    hour = s[11:13]
    minute = s[14:16]
    sec = s[17:19]
    time = float(hour)+float(minute)*(1.0/60)+float(sec)*(1.0/3600)
    return 60.0*time

# read call numbers
call_nums=np.genfromtxt(data_name, dtype=long, delimiter=",",skip_header=1, usecols=0)
# gets index of duplicate call numbers
# this just keeps the first one and deletes the rest, we should
# think about how to determine which to keep
#ret0,ret1 = np.unique(call_nums,return_index=True)
#unique = np.isin(range(0,20),ret1)
#for i in range(0,unique.size):
#    if not unique[i]:
#        bad_ind.append(i)
#print call_nums


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


# read latitude
lat=np.genfromtxt(data_name, dtype=float, delimiter=",",skip_header=1, usecols=35)
lat_min = config.params['lat_min']
lat_max = config.params['lat_max']
lat_range = lat_max-lat_min
# lat_sc = scaled latitude, over range (0,1)
lat_sc = np.subtract(lat,lat_min)
lat_sc = np.true_divide(lat_sc,lat_range)
for i in range(0,lat_sc.size):
    if np.isnan(lat_sc[i]):
        bad_ind.append(i)


# read longitude
lon=np.genfromtxt(data_name, dtype=float, delimiter=",",skip_header=1, usecols=36)
lon_min = config.params['lon_min']
lon_max = config.params['lon_max']
lon_range = lon_max-lon_min
# lon scaled
lon_sc = np.subtract(lon,lon_min)
lon_sc = np.true_divide(lon_sc,lon_range)
for i in range(0,lon_sc.size):
    if np.isnan(lon_sc[i]):
        bad_ind.append(i)


# read call timestamp
call_in_ts = np.genfromtxt(data_name, dtype=str, delimiter = ",", skip_header=1, usecols=6)
time_nums = []
for i in range(0,call_in_ts.size):
    time_nums.append(tsToNum(call_in_ts[i]))
call_in = np.array(time_nums)

# get call duration
call_enter_ts = np.genfromtxt(data_name, dtype=str, delimiter = ",", skip_header=1, usecols=7)
time_nums=[]
for i in range(0,call_enter_ts.size):
    time_nums.append(tsToNum(call_enter_ts[i]))
call_done = np.array(time_nums)
call_dur = np.subtract(call_done,call_in)


# dispatch times
disp_ts = np.genfromtxt(data_name, dtype=str, delimiter = ",", skip_header=1, usecols=8)
d_time = []
for i in range(0,disp_ts.size):
    if(disp_ts[i]==''):
        d_time.append(0)
       # bad_ind.append(i)
    else:d_time.append(tsToNum(disp_ts[i]))
d_time = np.array(d_time)
# on scene times
os_ts = np.genfromtxt(data_name, dtype=str, delimiter = ",", skip_header=1, usecols=10)
os_time = []
for i in range(0,os_ts.size):
    if(os_ts[i]==''):
        os_time.append(0)
        #bad_ind.append(i)
    else: os_time.append(tsToNum(os_ts[i]))
os_time = np.array(os_time)
resp_time = np.subtract(os_time,d_time)


# combine it all into one big array!
data = np.column_stack((call_nums,call_types,lat_sc,lon_sc,call_in,call_dur,resp_time))
data = np.core.records.fromarrays(data.transpose(),
                                    names="callNum, callType, latitude, longitude, callTime, callDur, respTime",
                                    formats = "uint32, uint8, float64, float64, float64, float64, float64")

# remove the "bad" indices
bad_ind = np.unique(bad_ind)
data = np.delete(data,bad_ind)


#this will sort data by call number, not sure if this is needed
data = np.sort(data,axis=0,order=['callNum'])

dup_ind=[]
for i in range(0,data.shape[0]):
    r_times = []
    r_times.append(data[i][6])
    call = data[i][0]
    while i+1<data.shape[0] and data[i+len(r_times)][0]==call:
        dup_ind.append(i+len(r_times))
        r_times.append(data[i+len(r_times)][6])
    for j in range(0,len(r_times)):
        if r_times[j]<0:
            r_times[j]=0
    avg_res = 0;
    if np.sum(r_times)>0:
        while np.isin(0,r_times):
            r_times.remove(0)
        avg_res=np.average(r_times)
    data[i][6] = avg_res

dup_ind = np.unique(dup_ind)
data = np.delete(data,dup_ind)

# write cleaned data to a csv file (so that we only have to clean once)
# csv out file will be filename_clean.csv
data_out_name = data_name[:-4]+"_clean.csv"
w_data = pd.DataFrame(data)
w_data.to_csv(data_out_name, index=None)

print "done! :)"







