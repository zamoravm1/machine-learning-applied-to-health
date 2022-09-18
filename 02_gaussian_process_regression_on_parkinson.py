# -*- coding: utf-8 -*-
"""
PEP 8 -- Style Guide for Python Code
https://www.python.org/dev/peps/pep-0008/

@author: visintin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def GPR(X_train,y_train,X_val,r2,s2):
    """ Estimates the output y_val given the input X_val, using the training data 
    and  hyperparameters r2 and s2"""
    Nva=X_val.shape[0]
    yhat_val=np.zeros((Nva,))
    sigmahat_val=np.zeros((Nva,))
    for k in range(Nva):
        x=X_val[k,:]# k-th point in the validation dataset
        A=X_train-np.ones((Ntr,1))*x
        dist2=np.sum(A**2,axis=1)
        ii=np.argsort(dist2)
        ii=ii[0:N-1];
        refX=X_train[ii,:]
        Z=np.vstack((refX,x))
        sc=np.dot(Z,Z.T)# dot products
        e=np.diagonal(sc).reshape(N,1)# square norms
        D=e+e.T-2*sc# matrix with the square distances 
        R_N=np.exp(-D/2/r2)+s2*np.identity(N)#covariance matrix
        R_Nm1=R_N[0:N-1,0:N-1]#(N-1)x(N-1) submatrix 
        K=R_N[0:N-1,N-1]# (N-1)x1 column
        d=R_N[N-1,N-1]# scalar value
        C=np.linalg.inv(R_Nm1)
        refY=y_train[ii]
        mu=K.T@C@refY# estimation of y_val for X_val[k,:]
        sigma2=d-K.T@C@K
        sigmahat_val[k]=np.sqrt(sigma2)
        yhat_val[k]=mu        
    return yhat_val,sigmahat_val


plt.close('all')
xx=pd.read_csv("./data/parkinsons_updrs.csv") # read the dataset
z=xx.describe().T # gives the statistical description of the content of each column
#xx.info()
# features=list(xx.columns)
features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
#%% scatter plots
todrop=['subject#', 'sex', 'test_time',  
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']
x1=xx.copy(deep=True)
X=x1.drop(todrop,axis=1)
#%% Generate the shuffled dataframe
np.random.seed(301368)
Xsh = X.sample(frac=1).reset_index(drop=True)
[Np,Nc]=Xsh.shape
F=Nc-1
#%% Generate training, validation and testing matrices
Ntr=int(Np*0.5)  # number of training points
Nva=int(Np*0.25) # number of validation points
Nte=Np-Ntr-Nva   # number of testing points
X_tr=Xsh[0:Ntr] # training dataset
# find mean and standard deviations for the features in the training dataset
mm=X_tr.mean()
ss=X_tr.std()
my=mm['total_UPDRS']# get mean for the regressand
sy=ss['total_UPDRS']# get std for the regressand
# normalize data
Xsh_norm=(Xsh-mm)/ss
ysh_norm=Xsh_norm['total_UPDRS']
Xsh_norm=Xsh_norm.drop('total_UPDRS',axis=1)
Xsh_norm=Xsh_norm.values
ysh_norm=ysh_norm.values
# get the training, validation, test normalized data
X_train_norm=Xsh_norm[0:Ntr]
X_val_norm=Xsh_norm[Ntr:Ntr+Nva]
X_test_norm=Xsh_norm[Ntr+Nva:]
y_train_norm=ysh_norm[0:Ntr]
y_val_norm=ysh_norm[Ntr:Ntr+Nva]
y_test_norm=ysh_norm[Ntr+Nva:]
y_train=y_train_norm*sy+my
y_val=y_val_norm*sy+my
y_test=y_test_norm*sy+my
#%% Optimizing hyperparameters r2 and s2

N=10
r2= np.linspace(1,10,100)
s2= np.array([0.0001,0.0002,0.0005,0.001,0.002])
labels= ['s2=0.0001','s2=0.0002','s2=0.0005','s2=0.001','s2=0.002']
MSE_val=np.zeros((len(r2),))

MSE_min=500 #inicializing optimize value for mse min

for v in range(len(s2)):
    
    for i in range(len(r2)):
        yhat_val_norm,sigmahat_val=GPR(X_train_norm,y_train_norm,X_val_norm,r2[i],s2[v])
        yhat_val=yhat_val_norm*sy+my
        err_val=y_val-yhat_val
        MSE_val[i]=round(np.mean((err_val)**2),3)
        
    
    if MSE_min> min(MSE_val):
            r2_opt=min(r2)
            s2_opt=s2[v]
            MSE_min=min(MSE_val)
            print(r2_opt,s2_opt,min(MSE_val))
    print(v)
    print(MSE_val)
    plt.plot(r2,MSE_val,label=labels[v])
    plt.legend()
    plt.title('Optimization of hyperparameters r2 and s2')
    plt.grid()
    plt.xlabel('r2')
    plt.ylabel('MSE_val')
    #plt.savefig('optimization.png')
   
plt.show()   

#print(r2_opt,s2_opt)

#%% Apply Gaussian Process Regression

yhat_train_norm,sigmahat_train=GPR(X_train_norm,y_train_norm,X_train_norm,r2_opt,s2_opt)
yhat_train=yhat_train_norm*sy+my
yhat_test_norm,sigmahat_test=GPR(X_train_norm,y_train_norm,X_test_norm,r2_opt,s2_opt)
yhat_test=yhat_test_norm*sy+my
yhat_val_norm,sigmahat_val=GPR(X_train_norm,y_train_norm,X_val_norm,r2_opt,s2_opt)
yhat_val=yhat_val_norm*sy+my
err_train=y_train-yhat_train
err_test=y_test-yhat_test
err_val=y_val-yhat_val
 

#%% plots
plt.figure()
plt.plot(y_test,yhat_test,'.b')
plt.plot(y_test,y_test,'r')
plt.grid()
plt.xlabel('y')
plt.ylabel('yhat')
plt.title('Gaussian Process Regression')
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5
#plt.savefig('GP_regression.png')

plt.figure()
plt.errorbar(y_test,yhat_test,yerr=3*sigmahat_test*sy,fmt='o',ms=2)
plt.plot(y_test,y_test,'r')
plt.grid()
plt.xlabel('y')
plt.ylabel('yhat')
plt.title('Gaussian Process Regression - with errorbars')
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5
#plt.savefig('GP_regression_errorbars.png')

e=[err_train,err_val,err_test]
plt.figure()
plt.hist(e,bins=50,density=True,range=[-8,17], histtype='bar',label=['Train.','Val.','Test'])
plt.xlabel('error')
plt.ylabel('P(error in bin)')
plt.legend()
plt.grid()
plt.title('Error histogram')
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5
#plt.savefig('GP_error_hist.png')

print('MSE train',round(np.mean((err_train)**2),3))
print('MSE test',round(np.mean((err_test)**2),3))
print('MSE valid',round(np.mean((err_val)**2),3))
print('Mean error train',round(np.mean(err_train),4))
print('Mean error test',round(np.mean(err_test),4))
print('Mean error valid',round(np.mean(err_val),4))
print('St dev error train',round(np.std(err_train),3))
print('St dev error test',round(np.std(err_test),3))
print('St dev error valid',round(np.std(err_val),3))
print('R^2 train',round(1-np.mean((err_train)**2)/np.mean(y_train**2),4))
print('R^2 test',round(1-np.mean((err_test)**2)/np.mean(y_test**2),4))
print('R^2 val',round(1-np.mean((err_val)**2)/np.mean(y_val**2),4))

#%% Normalize the three subsets

X_tr_norm=X_train_norm
X_te_norm=X_test_norm
y_tr_norm=y_train_norm
y_te_norm=y_test_norm

#%% Linear Least Squares
w_hat=np.linalg.inv(X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)
y_hat_te_norm=X_te_norm@w_hat
#y_hat_te=sy*h_hat_te_norm+my
#MSE=np.mean((y_hat_te-y_te)**2)
MSE_norm=np.mean((y_hat_te_norm-y_te_norm)**2)
MSE=sy**2*MSE_norm
#%% plots LLS

# plot the error histogram
E_tr=(y_tr_norm-X_tr_norm@w_hat)*sy# training
E_te=(y_te_norm-X_te_norm@w_hat)*sy# test
e=[E_tr,E_te]
plt.figure(figsize=(6,4))
plt.hist(e,bins=50,density=True, histtype='bar',
label=['training','test'])
plt.xlabel(r'$e=y-\^y$')
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title('LLS-Error histograms')
plt.tight_layout()
plt.savefig('./LLS-hist.png')
plt.show()
# plot the regression line
y_hat_te=(X_te_norm@w_hat)*sy+my
y_te=y_te_norm*sy+my
plt.figure(figsize=(6,4))
plt.plot(y_te,y_hat_te,'.')
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
plt.xlabel(r'$y$')
plt.ylabel(r'$\^y$')
plt.grid()
plt.title('LLS-test')
plt.tight_layout()
#plt.savefig('./LLS-yhat_vs_y.png')
plt.show()
#%% statistics of the errors
E_tr_mu=E_tr.mean()
E_tr_sig=E_tr.std()
E_tr_MSE=np.mean(E_tr**2)
y_tr=y_tr_norm*sy+my
R2_tr=1-(np.mean(E_tr_sig**2)/np.std(y_tr)**2)
E_te_mu=E_te.mean()
E_te_sig=E_te.std()
E_te_MSE=np.mean(E_te**2)
y_te=y_te_norm*sy+my
R2_te=1-(np.mean(E_te_sig**2)/np.std(y_te)**2)
rows=['Training','test']
cols=['mean','std','MSE','R^2']
p=np.array([[E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr],
            [E_te_mu,E_te_sig,E_te_MSE,R2_te]])
results=pd.DataFrame(p,columns=cols,index=rows)
print(results)
