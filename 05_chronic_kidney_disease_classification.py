import pandas as pd
import numpy as np
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt

# define the feature names:
feat_names=['age','bp','sg','al','su','rbc','pc',
'pcc','ba','bgr','bu','sc','sod','pot','hemo',
'pcv','wbcc','rbcc','htn','dm','cad','appet','pe',
'ane','classk']
ff=np.array(feat_names)
feat_cat=np.array(['num','num','cat','cat','cat','cat','cat','cat','cat',
         'num','num','num','num','num','num','num','num','num',
         'cat','cat','cat','cat','cat','cat','cat'])
# import the dataframe:
#xx=pd.read_csv("./data/chronic_kidney_disease.arff",sep=',',
#               skiprows=29,names=feat_names, 
#               header=None,na_values=['?','\t?'],
#               warn_bad_lines=True)
xx=pd.read_csv("./data/chronic_kidney_disease_v2.arff",sep=',',
    skiprows=29,names=feat_names, 
    header=None,na_values=['?','\t?'],)
Np,Nf=xx.shape
#%% change categorical data into numbers:
key_list=["normal","abnormal","present","notpresent","yes",
"no","poor","good","ckd","notckd","ckd\t","\tno"," yes","\tyes"]
key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
xx=xx.replace(key_list,key_val)
print(xx.nunique())# show the cardinality of each feature in the dataset; in particular classk should have only two possible values

#%% manage the missing data through regression
print(xx.info())
x=xx.copy()
# drop rows with less than 19=Nf-6 recorded features:
x=x.dropna(thresh=19)
x.reset_index(drop=True, inplace=True)# necessary to have index without "jumps"
n=x.isnull().sum(axis=1)# check the number of missing values in each row
print('max number of missing values in the reduced dataset: ',n.max())
print('number of points in the reduced dataset: ',len(n))
# take the rows with exctly Nf=25 useful features; this is going to be the training dataset
# for regression
Xtrain=x.dropna(thresh=25)
Xtrain.reset_index(drop=True, inplace=True)# reset the index of the dataframe
# get the possible values (i.e. alphabet) for the categorical features
alphabets=[]
for k in range(len(feat_cat)):
    if feat_cat[k]=='cat':
        val=Xtrain.iloc[:,k]
        val=val.unique()
        alphabets.append(val)
    else:
        alphabets.append('num')

#%% run regression tree on all the missing data
#normalize the training dataset
mm=Xtrain.mean(axis=0)
ss=Xtrain.std(axis=0)
Xtrain_norm=(Xtrain-mm)/ss
# get the data subset that contains missing values 
Xtest=x.drop(x[x.isnull().sum(axis=1)==0].index)
Xtest.reset_index(drop=True, inplace=True)# reset the index of the dataframe
Xtest_norm=(Xtest-mm)/ss # nomralization
Np,Nf=Xtest_norm.shape
regr=tree.DecisionTreeRegressor() # instantiate the regressor
for kk in range(Np):
    xrow=Xtest_norm.iloc[kk]#k-th row
    mask=xrow.isna()# columns with nan in row kk
    Data_tr_norm=Xtrain_norm.loc[:,~mask]# remove the columns from the training dataset
    y_tr_norm=Xtrain_norm.loc[:,mask]# columns to be regressed
    regr=regr.fit(Data_tr_norm,y_tr_norm)
    Data_te_norm=Xtest_norm.loc[kk,~mask].values.reshape(1,-1) # row vector
    ytest_norm=regr.predict(Data_te_norm)
    Xtest_norm.iloc[kk][mask]=ytest_norm # substitute nan with regressed values
Xtest_new=Xtest_norm*ss+mm # denormalize
# substitute regressed numerical values with the closest element in the alphabet
index=np.argwhere(feat_cat=='cat').flatten()
for k in index:
    val=alphabets[k].flatten() # possible values for the feature
    c=Xtest_new.iloc[:,k].values # values in the column
    c=c.reshape(-1,1)# column vector
    val=val.reshape(1,-1) # row vector
    d=(val-c)**2 # matrix with all the distances w.r.t. the alphabet values
    ii=d.argmin(axis=1) # find the index of the closest alphabet value
    Xtest_new.iloc[:,k]=val[0,ii]
print(Xtest_new.nunique())
print(Xtest_new.describe().T)
#
X_new= pd.concat([Xtrain, Xtest_new], ignore_index=True, sort=False)
##------------------ Decision tree -------------------
## first decision tree, using Xtrain for training and Xtest_new for test
target_names = ['notckd','ckd']
labels = Xtrain.loc[:,'classk']
data = Xtrain.drop('classk', axis=1)
clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=4)
clfXtrain = clfXtrain.fit(data,labels)
test_pred = clfXtrain.predict(Xtest_new.drop('classk', axis=1))
from sklearn.metrics import accuracy_score
print('accuracy =', accuracy_score(Xtest_new.loc[:,'classk'],test_pred))
from sklearn.metrics import confusion_matrix
print('Confusion matrix')
print(confusion_matrix(Xtest_new.loc[:,'classk'],test_pred))
tn, fp, fn, tp =confusion_matrix(Xtest_new.loc[:,'classk'],test_pred).ravel()
spec_org=tn / (tn+fp)
sens_org=tp / (tp+fn)
print('Specificity original: ',spec_org)
print('Sensibility original: ',sens_org)
#%% export to graghviz to draw a grahp
# dot_data = tree.export_graphviz(clfXtrain, out_file=None,feature_names=feat_names[:24], class_names=target_names, filled=True, rounded=True, special_characters=True) 
# graph = graphviz.Source(dot_data) 
# graph.render("Tree_Xtrain") 

'''

#black and white option
tree.plot_tree(clfXtrain)
#text option
text_representation = tree.export_text(clfXtrain)
print(text_representation)
#option with colors
fig = plt.figure(figsize=(50,50))
tree.plot_tree(clfXtrain,
                    feature_names=feat_names[:24],
                    class_names=target_names,
                    filled=True, rounded=True)

'''
#%% second decision tree, using X_new  
#%% Generate the shuffled data
seed=301364
st_seed=4
np.random.seed(seed) # set the seed for random shuffling
Np,Nc=X_new.shape
indexsh=np.arange(Np)
np.random.shuffle(indexsh)
Xsh=X_new.copy(deep=True)
Xsh=Xsh.set_axis(indexsh,axis=0,inplace=False)
Xsh=Xsh.sort_index(axis=0)

#%% Generate training, validation and test matrices
Ntr=158 # number of training points
Nte=Np-Ntr   # number of test points
X_tr=Xsh[0:Ntr]
X_te=Xsh[Ntr:]
 #Desicion tree
target_names = ['notckd','ckd']
labels = X_tr.loc[:,'classk']
data = X_tr.drop('classk', axis=1)
clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=st_seed)
clfXtrain = clfXtrain.fit(data,labels)
test_pred = clfXtrain.predict(X_te.drop('classk', axis=1))
from sklearn.metrics import accuracy_score
print('accuracy =', accuracy_score(X_te.loc[:,'classk'],test_pred))
from sklearn.metrics import confusion_matrix
print('Confusion matrix')
print(confusion_matrix(X_te.loc[:,'classk'],test_pred))
tn, fp, fn, tp =confusion_matrix(X_te.loc[:,'classk'],test_pred).ravel()
spec_org=tn / (tn+fp)
sens_org=tp / (tp+fn)
print ('state seed',st_seed)
print('seed: ',seed)
print('Specificity item: ',spec_org)
print('Sensibility item: ',sens_org)

#black and white option
tree.plot_tree(clfXtrain)
#text option
text_representation = tree.export_text(clfXtrain)
print(text_representation)
#option with colors
fig = plt.figure(figsize=(50,50))
tree.plot_tree(clfXtrain,
                    feature_names=feat_names[:24],
                    class_names=target_names,
                    filled=True, rounded=True)


#random forest
target_names = ['notckd','ckd']
labels = X_tr.loc[:,'classk']
data = X_tr.drop('classk', axis=1)
from sklearn.ensemble import RandomForestClassifier

clf= RandomForestClassifier(criterion='entropy',random_state=4)
clf= clf.fit(data,labels)

test_pred = clf.predict(X_te.drop('classk', axis=1))
from sklearn.metrics import accuracy_score
print('accuracy =', accuracy_score(X_te.loc[:,'classk'],test_pred))
from sklearn.metrics import confusion_matrix
print('Confusion matrix')
print(confusion_matrix(X_te.loc[:,'classk'],test_pred))
tn, fp, fn, tp =confusion_matrix(X_te.loc[:,'classk'],test_pred).ravel()
spec_rf=tn / (tn+fp)
sens_rf=tp / (tp+fn)
print ('state seed',st_seed)
print('seed: ',seed)
print('Specificity forest: ',spec_org)
print('Sensibility forest: ',sens_org)
'''
#black and white option
fn=labels
cn=target_names
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(clf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')
#AQUÍ TERMINO
'''