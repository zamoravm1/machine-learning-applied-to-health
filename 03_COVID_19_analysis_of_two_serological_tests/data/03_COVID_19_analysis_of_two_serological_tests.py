import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def findROC(x,y):# 
    """ findROC(x,y) generates data to plot the ROC curve.
    x and y are two 1D vectors each with length N
    x[k] is the scalar value measured in the test
    y[k] is either 0 (healthy person) or 1 (ill person)
    The output data is a 2D array N rows and three columns
    data[:,0] is the set of thresholds
    data[:,1] is the corresponding false alarm
    data[:,2] is the corresponding sensitivity"""
    
    if x.min()>0:# add a couple of zeros, in order to have the zero threshold
        x=np.insert(x,0,0)# add a zero as the first element of xs
        y=np.insert(y,0,0)# also add a zero in y
    
    ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
    ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
    x0=x[ii0]# test values for healthy patients
    x1=x[ii1]# test values for ill patients
    xs=np.sort(x)# sort test values: they represent all the possible  thresholds
    # if x> thresh -> test is positive
    # if x <= thresh -> test is negative
    # number of cases for which x0> thresh represent false positives
    # number of cases for which x0<= thresh represent true negatives
    # number of cases for which x1> thresh represent true positives
    # number of cases for which x1<= thresh represent false negatives
    # sensitivity = P(x>thresh|the patient is ill)=
    #             = P(x>thresh, the patient is ill)/P(the patient is ill)
    #             = number of pos1itives in x1/number of positives in y
    # false alarm = P(x>thresh|the patient is healthy)
    #             = number of positives in x0/number of negatives in y
    Np=ii1.size# number of positive cases
    Nn=ii0.size# number of negative cases
    data=np.zeros((Np+Nn,7),dtype=float)
    i=0
    ROCarea=0
    pD=0.02
    pH=1-pD
    min=100
    pos=100
    for thresh in xs:
        n1=np.sum(x1>thresh)#true positives
        sens=n1/Np
        n2=np.sum(x0>thresh)#false positives
        #false negatives
        falsealarm=n2/Nn
        #Probability healthy people with negative test
        PHTn=((1-falsealarm)*pH)/((1-sens)*pD+(1-falsealarm)*pH)
        #Probability ill people with negative test
        PDTn=1-PHTn
        #Probability healthy people with positive test
        PDTp=(sens*pD)/(sens*pD+falsealarm*pH)
        #Probability ill people with positive test
        PHTp=1-PDTp
        data[i,0]=thresh
        data[i,1]=falsealarm
        data[i,2]=sens
        data[i,3]=PHTn
        data[i,4]=PDTn
        data[i,5]=PDTp
        data[i,6]=PHTp
        #print(abs(sens-(1-falsealarm)))
        if (abs(sens-(1-falsealarm))<min) and ((1-falsealarm)!=0) and (sens!=0):
            min=abs(sens-(1-falsealarm))
            pos=i
            #print(pos)
        if i>0:
            ROCarea=ROCarea+sens*(data[i-1,1]-data[i,1])
        i=i+1
    return data,ROCarea,pos
#%% Clean data
plt.close('all')
xx=pd.read_csv("covid_serological_results.csv")
swab=xx.COVID_swab_res.values# results from swab: 0= no illness, 1 = unclear, 2=illness
Test1=xx.IgG_Test1_titre.values
Test2=xx.IgG_Test2_titre.values
ii=np.argwhere(swab==1).flatten() 
# argwhere -  find the indices of array elements
# flatten a matrix to one dimension
swab=np.delete(swab,ii)
swab=swab//2 
# to pass vector in terms of zero and one, int division 0= no illness, 1= illness

Test1=np.delete(Test1,ii)
Test2=np.delete(Test2,ii)

#%% Remove outliers from Test1.
Test_rs= Test1.reshape(-1,1)
clustering = DBSCAN(eps=0.4, min_samples=3).fit(Test_rs)
clustering.labels_
ii_test=np.argwhere(clustering.labels_== -1).flatten()
Test1_with_dbs=np.delete(Test_rs,ii_test)

plt.figure()
plt.plot(Test_rs,'r')
plt.plot(Test1_with_dbs,'b')


swab_test1=np.delete(swab,ii_test)


#%% Hist Test1
ii0_test1=np.argwhere(swab_test1==0)
ii1_test1=np.argwhere(swab_test1==1)
plt.figure()
plt.hist(Test1_with_dbs[ii0_test1],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
plt.hist(Test1_with_dbs[ii1_test1],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
plt.grid()
plt.legend()
plt.title('Test1- Histogram')

#%% Hist Test2
ii0_test2=np.argwhere(swab==0)
ii1_test2=np.argwhere(swab==1)
plt.figure()
plt.hist(Test2[ii0_test2],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
plt.hist(Test2[ii1_test2],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
plt.grid()
plt.legend()
plt.title('Test2- Histogram')




#%% ROC Test2
data_Test2,ROCarea_Test2,pos=findROC(Test2,swab)
thresh_Test2= data_Test2[:,1]

print('thresh_ref_Test2: ',data_Test2[pos,0])
print('sen_ref_Test2: ',data_Test2[pos,2])
print('spe_ref_Test2: ',1-data_Test2[pos,1])
print('pDTn_Test2: ',data_Test2[pos,4])
print('pDTp_Test2: ',data_Test2[pos,5])

plt.figure()
plt.plot(data_Test2[:,1],data_Test2[:,2],'-',label='Test2')
plt.xlabel('FA')
plt.ylabel('Sens')
plt.grid()
plt.legend()
plt.title('Test2 - ROC')
plt.savefig('./ROC2.png')

plt.figure()
plt.plot(data_Test2[:,0],data_Test2[:,1],'.',label='False alarm')
plt.plot(data_Test2[:,0],data_Test2[:,2],'.',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2')
plt.grid()


plt.figure()
plt.plot(data_Test2[:,0],1-data_Test2[:,1],'-',label='Specificity')
plt.plot(data_Test2[:,0],data_Test2[:,2],'-',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2 - Specificity VS Sensitivity')
plt.grid()
plt.savefig('./tresh2.png')

#%% P(D|Tp) and P(D|Tn) versus threshold for Test 2

plt.figure()
plt.plot(data_Test2[:,0],data_Test2[:,5],'-',label='P(D|Tp)')
plt.plot(data_Test2[:,0],data_Test2[:,4],'-',label='P(D|Tn)')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2- P(D|Tp) vs P(D|Tn)')
plt.grid()
plt.savefig('./test2_pdtp_pdtn.png')

#%% P(H|Tp) and P(H|Tn) versus threshold for Test 2

plt.figure()
plt.plot(data_Test2[:,0],data_Test2[:,6],'-',label='P(H|Tp)')
plt.plot(data_Test2[:,0],data_Test2[:,3],'-',label='P(H|Tn)')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2- P(H|Tp) vs P(H|Tn)')
plt.grid()
plt.savefig('./test2_pHtp_pHtn.png')

#%% ROC Test1

data_Test1,ROCarea_Test1,pos=findROC(Test1_with_dbs,swab_test1)
thresh_Test1= data_Test1[:,1]

print('thresh_ref_Test1: ',data_Test1[pos,0])
print('sen_ref_Test1: ',data_Test1[pos,2])
print('spe_ref_Test1: ',1-data_Test1[pos,1])
print('pDTn_Test1: ',data_Test1[pos,4])
print('pDTp_Test1: ',data_Test1[pos,5])
plt.figure()
plt.plot(data_Test1[:,1],data_Test1[:,2],'-',label='Test1')
plt.xlabel('FA')
plt.ylabel('Sens')
plt.grid()
plt.legend()
plt.title('Test1 - ROC')
plt.savefig('./ROC1.png')

plt.figure()
plt.plot(data_Test1[:,0],data_Test1[:,1],'.',label='False alarm')
plt.plot(data_Test1[:,0],data_Test1[:,2],'.',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test1')
plt.grid()


plt.figure()
plt.plot(data_Test1[:,0],1-data_Test1[:,1],'-',label='Specificity')
plt.plot(data_Test1[:,0],data_Test1[:,2],'-',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test1 - Specificity VS Sensitivity')
plt.grid()
plt.savefig('./tresh1.png')

#%% P(D|Tp) and P(D|Tn) versus threshold for Test 1

plt.figure()
plt.plot(data_Test1[:,0],data_Test1[:,5],'-',label='P(D|Tp)')
plt.plot(data_Test1[:,0],data_Test1[:,4],'-',label='P(D|Tn)')
plt.legend()
plt.xlabel('threshold')
plt.title('Test1- P(D|Tp) vs P(D|Tn)')
plt.grid()
plt.savefig('./test1_pdtp_pdtn.png')
#%% P(H|Tp) and P(H|Tn) versus threshold for Test 1

plt.figure()
plt.plot(data_Test1[:,0],data_Test1[:,6],'-',label='P(H|Tp)')
plt.plot(data_Test1[:,0],data_Test1[:,3],'-',label='P(H|Tn)')
plt.legend()
plt.xlabel('threshold')
plt.title('Test1 - P(H|Tp) vs P(H|Tn)')
plt.grid()
plt.savefig('./test1_pHtp_pHtn.png')