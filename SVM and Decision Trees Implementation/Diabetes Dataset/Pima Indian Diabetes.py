#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

get_ipython().run_line_magic('matplotlib', 'inline')


# # Load and Review Data

# In[2]:


data = pd.read_csv(r'C:\Users\ANCHAL\Documents\utd_coursework\machine_learning\assignments\assignment 2\diabetes.csv')


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.isnull().values.any()


# In[7]:


columns = list(data)[0:-1] # Excluding Outcome column which has only 
data[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2)); 
# Histogram of first 8 columns


# # Identify Correlation in data

# In[8]:


def plot_corr(data, size=11):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr(data)


# # Calculate diabetes ratio of True/False from outcome variable

# In[9]:


n_true = len(data.loc[data['Outcome'] == True])
n_false = len(data.loc[data['Outcome'] == False])
print("Number of true cases: {0} ({1:2.2f}%)".format(n_true, (n_true / (n_true + n_false)) * 100 ))
print("Number of false cases: {0} ({1:2.2f}%)".format(n_false, (n_false / (n_true + n_false)) * 100))
count_Class = pd.value_counts(data["Outcome"], sort= True)
count_Class.plot(kind= 'bar')
plt.show()


# # import librraries for algorithms

# In[10]:


from sklearn import linear_model as lm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve,auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as rfc 


# # Spliting the data 

# In[11]:


X = data.drop('Outcome',axis=1) # Features
y = data['Outcome'] # Target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=52)


# # replace zeroes with serial median

# In[12]:


X


# In[13]:


from sklearn.preprocessing import Imputer

rep_0 = Imputer(missing_values=0, strategy="median", axis=0)

X_train = rep_0.fit_transform(X_train)
X_test = rep_0.fit_transform(X_test)


# # model selection and model fitting 

# # SVM

# # kernel = rbf 

# In[14]:


from sklearn.svm import SVC
model =SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

svm_rbf_roc_auc = roc_auc_score(y_test, model.predict(X_test))
print ("Logistic AUC = %2.2f" % svm_rbf_roc_auc)


# In[15]:


model.fit(X_train,y_train)


# In[16]:


rb_fpr,rb_tpr,_=roc_curve(y_test,predictions)
#calculate AUC
svm_rbf_roc_auc=auc(rb_fpr,rb_tpr)
print('AUC: %0.2f' % svm_rbf_roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(rb_fpr,rb_tpr,label='ROC curve(area= %2.f)' %svm_rbf_roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[17]:


from sklearn.model_selection import learning_curve, GridSearchCV
param_grid ={'C': [0.01,0.1,1,10,0.001],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[18]:


grid = GridSearchCV(SVC(probability=True),param_grid, verbose=5,refit=True)


# In[19]:


grid.fit(X_train,y_train)


# In[20]:


grid.best_params_


# In[21]:


grid.best_estimator_


# In[22]:


grid1 = SVC(verbose=5, C=1,gamma=0.0001, kernel='rbf')
grid1.fit(X_train,y_train)
grid_predictions =grid1.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))

svm_rbf_roc_auc1 = roc_auc_score(y_test, grid.predict(X_test))
print ("SVM RBF AUC = %2.2f" % svm_rbf_roc_auc1)
acc_rbf = accuracy_score(y_test, grid_predictions)
print('Accuracy Score = %f'% acc_rbf)


# In[23]:


rb_fpr1,rb_tpr1,_=roc_curve(y_test,grid_predictions)
#calculate AUC
svm_rbf_roc_auc1=auc(rb_fpr1,rb_tpr1)
print('AUC: %0.2f' % svm_rbf_roc_auc1)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(rb_fpr1,rb_tpr1,label='ROC curve(area= %2.f)' %svm_rbf_roc_auc1)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# # kernel = linear

# In[24]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',probability=True)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy",svclassifier.score(X_test,y_test)*100)
acc_linear = accuracy_score(y_test, y_pred)
print('Accuracy Score of linear = %f'% acc_linear)


# In[25]:


svm_linear_fpr,svm_linear_tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
svm_linear_roc_auc=auc(svm_linear_fpr,svm_linear_tpr)
print('AUC: %0.2f' %svm_linear_roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(svm_linear_fpr,svm_linear_tpr,label='ROC curve(area= %2.f)' %svm_linear_roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[26]:


param_grid1 ={'C': [0.01,0.1,1,10,0.001],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['linear']}


# In[27]:


grid1 = GridSearchCV(SVC(),param_grid1, verbose=5,refit=True)
grid1.fit(X_train,y_train)


# In[28]:


grid1.best_params_
grid1.best_estimator_


# In[29]:


grid1.best_params_


# In[30]:


grid_predictions1 =grid1.predict(X_test)
print(confusion_matrix(y_test, grid_predictions1))
print(classification_report(y_test, grid_predictions1))


# # kernel = sigmoid 

# In[31]:


from sklearn.svm import SVC
svclassifier1 = SVC(kernel='sigmoid',probability=True)
svclassifier1.fit(X_train, y_train)
y_pred1 = svclassifier1.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
print("Accuracy",svclassifier1.score(X_test,y_test)*100)
acc_sigmoid = accuracy_score(y_test, y_pred1)
print('Accuracy Score = %f'% acc_sigmoid)


# In[32]:


sig_fpr,sig_tpr,_=roc_curve(y_test,y_pred1)
#calculate AUC
sig_roc_auc=auc(sig_fpr,sig_tpr)
print('AUC: %0.2f' % sig_roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(sig_fpr,sig_tpr,label='ROC curve(area= %2.f)' %sig_roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# # Comparision between kernels 

# In[33]:


z = pd.DataFrame(data = [acc_linear,acc_rbf,acc_sigmoid],index = ['Linear','RBF','Sigmoid'],columns=[ "Accuracy"])
z.plot(kind = 'bar',title="Accuracy vs Algorithm for GPU kernel performance")
plt.show()


# # Decision Tree

# In[34]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
# use the model to make predictions with the test data
dtree_pred = dtree.predict(X_test)
count_misclassified = (y_test != dtree_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy_tree = accuracy_score(y_test, dtree_pred)

print(confusion_matrix(y_test, dtree_pred))
print(classification_report(y_test, dtree_pred))
print("Accuracy",dtree.score(X_test,y_test)*100)


# In[35]:


dtree.fit(X_train, y_train)


# In[36]:


dtree_fpr,dtree_tpr,_=roc_curve(y_test,dtree_pred)
#calculate AUC
dtree_roc_auc=auc(dtree_fpr,dtree_tpr)
print('AUC: %0.2f' % dtree_roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(dtree_fpr,dtree_tpr,label='ROC curve(area= %2.f)' %dtree_roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[37]:


# using GridSearchCV to find out best values of the hyper parameters
parameters={'max_depth': range(3,11,2), 'criterion': ["gini", "entropy"],
            'splitter': ["best"], 'max_leaf_nodes': range(5,15,2)}
clf = GridSearchCV(dtree, parameters, cv=10)
clf= clf.fit(X_train,y_train)
print(clf.best_estimator_)
y_pred_grid = clf.predict(X_test)
count_misclassified = (y_test != y_pred_grid).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy_tree_grid = accuracy_score(y_test, y_pred_grid)
print('Accuracy: {:.2f}'.format(accuracy_tree_grid))

print(confusion_matrix(y_test, y_pred_grid))
print(classification_report(y_test, y_pred_grid))


# In[38]:


dtree_fpr,dtree_tpr,_=roc_curve(y_test,y_pred_grid)
#calculate AUC
dtree_roc_auc=auc(dtree_fpr,dtree_tpr)
print('AUC: %0.2f' % dtree_roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(dtree_fpr,dtree_tpr,label='ROC curve(area= %2.f)' %dtree_roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# # XGBOOST CLassifier

# In[39]:


from xgboost import XGBClassifier


# In[40]:



gbc=XGBClassifier()
gbc.fit(X_train, y_train)
print ("\n\n ---GBC---")
gbc_roc_auc = roc_auc_score(y_test, gbc.predict(X_test))
print ("GBC AUC = %2.2f" % gbc_roc_auc)
print(classification_report(y_test, gbc.predict(X_test)))
print(confusion_matrix(y_test, gbc.predict(X_test)))
print ("GBC accuracy is %2.2f" % accuracy_score(y_test, gbc.predict(X_test) ))


# In[41]:


param_grid2 ={'learning_rate': [0.01,0.1,1,10,0.001],'max_features':[4,5,6,7,8],'n_estimators':[200,300,400,500,600]}


# In[42]:


gbc_grid = GridSearchCV(estimator = XGBClassifier(max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_grid2, scoring='accuracy',n_jobs=4,iid=False, cv=5)


# In[43]:


gbc_grid.fit(X_train,y_train)


# In[44]:


gbc_grid.best_params_


# In[45]:


print ("\n\n ---GBC---")
gbc_roc_auc = roc_auc_score(y_test, gbc_grid.predict(X_test))
print ("GBC AUC = %2.2f" % gbc_roc_auc)
print(classification_report(y_test, gbc_grid.predict(X_test)))
print(confusion_matrix(y_test, gbc_grid.predict(X_test)))
print ("GBC accuracy is %2.2f" % accuracy_score(y_test,gbc_grid.predict(X_test) ))


# # Gradient boosting classifier

# In[46]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier as abc


# In[47]:



gbc1=GradientBoostingClassifier()
gbc1.fit(X_train, y_train)
print ("\n\n ---GBC---")
gbc_roc_auc1 = roc_auc_score(y_test, gbc1.predict(X_test))
print ("GBC AUC = %2.2f" % gbc_roc_auc1)
print(classification_report(y_test, gbc1.predict(X_test)))
print(confusion_matrix(y_test, gbc1.predict(X_test)))
print ("GBC accuracy is %2.2f" % accuracy_score(y_test, gbc1.predict(X_test) ))


# In[48]:


param_grid3 ={'learning_rate': [0.01,0.1,1,10,0.001],'max_features':[4,5,6,7,8],'n_estimators':[200,300,400,500,600]}


# In[49]:


gbc_grid1 = GridSearchCV(estimator = GradientBoostingClassifier(max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_grid3, scoring='accuracy',n_jobs=4,iid=False, cv=5)


# In[50]:


gbc_grid1.fit(X_train,y_train)


# In[51]:


gbc_grid1.best_params_


# In[52]:


print ("\n\n ---GBC---")
gbc_roc_auc1 = roc_auc_score(y_test, gbc_grid1.predict(X_test))
print ("GBC AUC = %2.2f" % gbc_roc_auc1)
print(classification_report(y_test, gbc_grid1.predict(X_test)))
print(confusion_matrix(y_test, gbc_grid1.predict(X_test)))
print ("GBC accuracy is %2.2f" % accuracy_score(y_test,gbc_grid1.predict(X_test) ))


# # ROC Comparision
# 

# In[54]:


from sklearn.metrics import roc_curve
rb_fpr1, rb_tpr1, rb_thresholds1 = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
svm_linear_fpr, svm_linear_tpr, svm_linear_thresholds = roc_curve(y_test, svclassifier.predict_proba(X_test)[:,1])
#sig_fpr, sig_tpr, sig_thresholds = roc_curve(y_test, y_pred1.predict(X_test)[:,1])
dtree_fpr, dtree_tpr, dtree_thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:,1])

gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc_grid1.predict_proba(X_test)[:,1])
plt.figure(figsize=(10, 10))

# Plot Logistic Regression ROC
plt.plot(rb_fpr1, rb_tpr1, label = 'SVM kernel is RBF  (area = %0.2f)' % svm_rbf_roc_auc1)
plt.plot(svm_linear_fpr, svm_linear_tpr, label = 'SVM kernel is linear  (area = %0.2f)' % svm_linear_roc_auc)
plt.plot(sig_fpr, sig_tpr, label = 'SVM kernel is sigmoid  (area = %0.2f)' % sig_roc_auc)
plt.plot(dtree_fpr, dtree_tpr, label = 'Decision Tree   (area = %0.2f)' % dtree_roc_auc)

plt.plot(gbc_fpr, gbc_tpr, label = 'Gradient Booster   (area = %0.2f)' % gbc_roc_auc1)

plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


# # Cross Validation

# In[55]:


models = []

models.append(('SVC Sigmoid', SVC(kernel='sigmoid')))
models.append(('SVC linear', svclassifier))

models.append(('SVC rbf', grid1))
models.append(('Decision Tree', clf))
models.append(('Gradient booster', gbc_grid1))


# In[56]:


seed = 7
results = []
results1 = []
names = []
names1 = []
X = X_test
Y = y_test


# # MSE for all models 

# In[57]:


for name, model in models:
    kfold = model_selection.KFold(
        n_splits=6, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s (%s): %f (%f)" % (
        "MSE",name, cv_results.mean(), cv_results.std())
    print(msg)


# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 7))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # Accuracy for all models

# In[59]:


for name1, model1 in models:
    kfold1 = model_selection.KFold(
        n_splits=6, random_state=seed)
    cv_results1 = model_selection.cross_val_score(
        model1, X, Y, cv=kfold1, scoring='accuracy')
    results1.append(cv_results1)
    names1.append(name1)
    msg1 = "%s (%s): %f" % (
        "Accuracy",name1, cv_results1.mean())
    print(msg1)


# In[60]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
fig1 = plt.figure(figsize=(10, 7))
fig1.suptitle('Algorithm Comparison')
ax1 = fig1.add_subplot(111)
plt.boxplot(results1)
ax1.set_xticklabels(names1)
plt.show()


# In[61]:


z = pd.DataFrame(data = [0.753, 0.7727, 0.649, 0.80, 0.79],
                 index = ['Linear','RBF','Sigmoidal','Decision Tree','Gradient Boost' ],columns=[ "Accuracy"])
z.plot(kind = 'bar',title="Accuracy vs Algorithms for Diabetes dataset")
plt.show()

