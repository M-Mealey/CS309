import numpy as np
from ann import ANN

fileName = "testing.csv"


weights=np.genfromtxt("topind.csv", dtype=float, delimiter=",",skip_header=0, usecols=0)
weights = weights.tolist()

lat=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=0)
lon=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=1)
time=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=2)
call_dur=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=3)
call_type=np.genfromtxt(fileName, dtype=int, delimiter=",",skip_header=1, usecols=4)
response_time=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=5)


inputs = np.column_stack((lat,lon,time,call_dur))
outputs = np.column_stack((call_type,response_time))



ann = ANN(weights)
ann_out_ctype = []
ann_out_rtime = []
for i in range(0,inputs.shape[0]):
    out=ann.evaluate(inputs[i])
    ann_out_ctype.append(out[0])
    ann_out_rtime.append(out[1])
ann_out_ctype = np.around(ann_out_ctype)

correct = 0
for i in range(len(ann_out_ctype)):
##    print ann_out_ctype[i], "   ", outputs[i][0]
    if (ann_out_ctype[i]==outputs[i][0]):
        correct = correct+1

ctype_correct = (float(correct)/len(ann_out_ctype))*100
print "correct: ",correct
print "percent: ", ctype_correct

within_one = 0;
response_err = []
for i in range(len(ann_out_rtime)):
##    print ann_out_rtime[i], "    ",outputs[i][1]
    response_err.append(ann_out_rtime[i] - outputs[i][1])
    if( np.abs(ann_out_rtime[i]-outputs[i][1]) < 1):
        within_one = within_one+1;

print within_one, " predictions were within one minute"
print float(within_one)/len(ann_out_rtime) *100, "percent of predictions were within one minute"
print "mean response prediction error: ",np.mean(response_err)
print "response prediction error st dev: ", np.std(response_err)




