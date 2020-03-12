# Plotting the plots and loss curves
# Refer to the Notebook 'AutoEncodersAtlasDataCompression2.ipynb' if any doubts
# Plotting the loss curve
qw=b['Epoch']
qe=b['train_loss']
qr=b['val_loss']
plt.plot(qw,qe)
plt.plot(qw,qr)
plt.legend(['train','validation'],loc='upper right')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Reconstruction Loss Curve')
plt.savefig('reconstruction_loss.png',dpi=500,bbox_inches='tight')

# Evaluation of the model architecture
model.eval()
# printing the input and output values for comparison
print('Comparing input and output:')
for ii in np.arange(90, 105):
    data = valid_ds.tensors[0][ii]
    pred = model(data)
    print('Inp:', data)
    print('Out:', pred)
    print(' ')
# Unnormalization 
aa=[]
ab=[]
for each in valid_ds.tensors[0]:
    aa.append((each*train_std)+train_mean)
    c=(model(each)).detach().numpy()
    ab.append((c*train_std)+train_mean)
# List are being appended for being used in plotting
ac=[]
ad=[]
ae=[]
af=[]
ag=[]
ah=[]
ai=[]
aj=[]
i=0
while(i<len(aa)):
    ac.append(aa[i][0])
    ad.append(ab[i][0])
    ae.append(aa[i][1])
    af.append(ab[i][1])
    ag.append(aa[i][2])
    ah.append(ab[i][2])
    ai.append(aa[i][3])
    aj.append(ab[i][3])
    i+=1
# Standards used for graphs
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from statistics import mean
import numpy as np
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    return m
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
a="R2".translate(SUP)
c="0C".translate(SUP)
title_font = {'fontname':'Arial', 'size':'16', 'color':'black'}
# Regression plot for parameter- m
fig, ax = plt.subplots()
data1=ac
data2=ad
xs = np.array(data1, dtype=np.float64)
ys = np.array(data2, dtype=np.float64)
m = best_fit_slope(xs,ys)
b = mean(ys) - m*mean(xs)
import matplotlib.pyplot as plt
ax.scatter(data1, data2)
yt=int(max(data2))
ax.plot([0, yt], [0, yt], 'k--', lw=4)
y=m*xs+b
ax.plot(xs,y,color='red',label='y='+str(m)+'x'+'+'+str(b))
ax.set_xlabel('Measured values')
ax.set_ylabel('Estimated values')
ax.text(10,100000,'Parameter-m', **title_font)
ax.text(10,80000,'y='+str(round(m,4))+'x'+'+'+str(round(b,4)))
ax.text(10,60000,str(a)+'='+str(round(r2_score(data1,data2),3)))
ax.text(10,70000,'MAE='+str(round(mean_absolute_error(data1,data2),3)))
plt.savefig('RegressionPlotm.png',dpi=500,bbox_inches='tight')
plt.show()
# Regression plot for parameter - pt
fig, ax = plt.subplots()
data1=ae
data2=af
xs = np.array(data1, dtype=np.float64)
ys = np.array(data2, dtype=np.float64)
m = best_fit_slope(xs,ys)
b = mean(ys) - m*mean(xs)
import matplotlib.pyplot as plt
ax.scatter(data1, data2)
yt=int(max(data2))
ax.plot([0, yt], [0, yt], 'k--', lw=4)
y=m*xs+b
ax.plot(xs,y,color='red',label='y='+str(m)+'x'+'+'+str(b))
ax.set_xlabel('Measured values')
ax.set_ylabel('Estimated values')
ax.text(10,700000,'Parameter-pt', **title_font)
ax.text(10,600000,'y='+str(round(m,4))+'x'+'+'+str(round(b,4)))
ax.text(10,500000,str(a)+'='+str(round(r2_score(data1,data2),3)))
ax.text(10,400000,'MAE='+str(round(mean_absolute_error(data1,data2),3)))
plt.savefig('RegressionPlotpt.png',dpi=500,bbox_inches='tight')
plt.show()
# Regression plot for parameter - phi
fig, ax = plt.subplots()
data1=ag
data2=ah
xs = np.array(data1, dtype=np.float64)
ys = np.array(data2, dtype=np.float64)
m = best_fit_slope(xs,ys)
b = mean(ys) - m*mean(xs)
import matplotlib.pyplot as plt
ax.scatter(data1, data2)
yt=int(max(data2))
yu=(min(data2))
ax.plot([yu, yt], [yu, yt], 'k--', lw=4)
y=m*xs+b
ax.plot(xs,y,color='red',label='y='+str(m)+'x'+'+'+str(b))
ax.set_xlabel('Measured values')
ax.set_ylabel('Estimated values')
ax.text(-3,3,'Parameter-phi', **title_font)
ax.text(-3,2.5,'y='+str(round(m,4))+'x'+'+'+str(round(b,4)))
ax.text(-3,2,str(a)+'='+str(round(r2_score(data1,data2),3)))
ax.text(-3,1.5,'MAE='+str(round(mean_absolute_error(data1,data2),3)))
plt.savefig('Regressionplotphi.png',dpi=500)
plt.show()
# Regression plot for parameter - eta
fig, ax = plt.subplots()
data1=ai
data2=aj
xs = np.array(data1, dtype=np.float64)
ys = np.array(data2, dtype=np.float64)
m = best_fit_slope(xs,ys)
b = mean(ys) - m*mean(xs)
import matplotlib.pyplot as plt
ax.scatter(data1, data2)
yt=int(max(data2))
yu=(min(data2))
ax.plot([yu, yt], [yu, yt], 'k--', lw=4)
y=m*xs+b
ax.plot(xs,y,color='red',label='y='+str(m)+'x'+'+'+str(b))
ax.set_xlabel('Measured values')
ax.set_ylabel('Estimated values')
ax.text(-4.5,3.5,'Parameter-eta', **title_font)
ax.text(-4.5,2.8,'y='+str(round(m,4))+'x'+'+'+str(round(b,4)))
ax.text(-4.5,2.1,str(a)+'='+str(round(r2_score(data1,data2),3)))
ax.text(-4.5,1.4,'MAE='+str(round(mean_absolute_error(data1,data2),3)))
plt.savefig('Regressionploteta.png',dpi=500,bbox_inches='tight')
plt.show()

# Lists being appended for relative difference value plots
ma=[]
mb=[]
md=[]
me=[]
mc=[]
po=0
while(po<len(ac)):
    ma.append((ad[po]-ac[po])/ac[po])
    mc.append((af[po]-ae[po])/ae[po])
    md.append((ah[po]-ag[po])/ag[po])
    me.append((aj[po]-ai[po])/ai[po])
    mb.append(po)
    po+=1

# Relative difference value plot for parameter- m    
plt.plot(mb,ma)
plt.xlabel('ith value')
plt.ylabel('Relative difference between estimated and original value')
plt.title('Plot for parameter - m')
plt.savefig('Relativediferrenceplotm.png',dpi=500,bbox_inches='tight')

#Histogram for parameter - m
plt.hist(ma)
plt.ylabel('Frequency')
plt.xlabel('Relative difference value')
plt.title('Histogram for Relative Difference values in m')
plt.savefig('Relativedifferencevalueshistogramm.png',dpi=500,bbox_inches='tight')

#Relative difference value plot for parameter - pt
plt.plot(mb,mc)
plt.xlabel('ith value')
plt.ylabel('Relative difference between estimated and original value')
plt.title('Plot for parameter - pt')
plt.savefig('Relativedifferencevaluespt.png',dpi=500,bbox_inches='tight')

#Histogram for parameter- pt
plt.hist(mc)
plt.ylabel('Frequency')
plt.xlabel('Relative difference value')
plt.title('Histogram for Relative Difference values in pt')
plt.savefig('Relativedifferencevaluespthist.png',dpi=500,bbox_inches='tight')

#Relative difference value plot for parameter - phi
plt.plot(mb,md)
plt.xlabel('ith value')
plt.ylabel('Relative difference between estimated and original value')
plt.title('Plot for parameter - phi')
plt.savefig('Relativedifferencevaluesphi.png',dpi=500,bbox_inches='tight')

#Histogram for phi
plt.hist(md)
plt.ylabel('Frequency')
plt.xlabel('Relative difference value')
plt.title('Histogram for Relative Difference values in phi')
plt.savefig('Relativedifferencevaluesphihist.png',dpi=500,bbox_inches='tight')

#Relative difference value plot in eta
plt.plot(mb,me)
plt.xlabel('ith value')
plt.ylabel('Relative difference between estimated and original value')
plt.title('Plot for parameter - eta')
plt.savefig('Relativedifferencevalueseta.png',dpi=500,bbox_inches='tight')

#Histogram for eta
plt.hist(me)
plt.ylabel('Frequency')
plt.xlabel('Relative difference value')
plt.title('Histogram for Relative Difference values in eta')
plt.savefig('Relativedifferencevaluesetahist.png',dpi=500,bbox_inches='tight')

