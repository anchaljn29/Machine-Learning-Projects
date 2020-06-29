#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve,auc, roc_auc_score


# In[2]:


def confusionmatrix_plot(test, pred):
    conf_matrix = confusion_matrix(test, pred)
    fig = plt.plot(figsize=(8,5))
    sns.heatmap(conf_matrix, annot=True,cmap='Blues',fmt='g')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual value')
    plt.show()


# In[3]:


def roc_curve_plot(test, pred, name):
    fpr, tpr, _ = roc_curve(test, pred)
    # calculate AUC
    roc_auc = round(auc(fpr, tpr), 2)
    print('AUC: %0.2f' % roc_auc)
    # plot of ROC curve for a specified class
    plt.figure()
    plt.plot(fpr, tpr, label='{} ROC = {}'.format(name, roc_auc), marker='.')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


# In[4]:


def learning_curve(estimator, title, X_dt, y_dt, ylim, cv, n_jobs, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_dt, y_dt, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


# # data set info

# In[5]:


df = pd.read_csv(r"C:\Users\ANCHAL\Documents\utd_coursework\machine_learning\assignments\assignment 2\sgemm_product.csv")

df['AvgRun'] = round((df['Run1 (ms)']+df['Run2 (ms)']+df['Run3 (ms)']+df['Run4 (ms)'])/4, 2)
df.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], inplace=True, axis=1)
# converting numerical Y to binary form based on median
df['y_class'] = np.where(df['AvgRun']>df['AvgRun'].median(),1,0)
df.drop(['AvgRun'], inplace=True, axis=1)
# visualize target variable
count_Class = pd.value_counts(df["y_class"], sort= True)
count_Class.plot(kind= 'bar')
plt.show()

def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr(df)
# splitting data to 70-30 ratio.
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df['y_class'], test_size=0.3)


# # kernel= linear

# In[ ]:


#SVM implementation
cls_linear = svm.SVC(kernel='linear')
cls_linear.fit(x_train, y_train)
predict_linear = cls_linear.predict(x_test)
cm_linear = confusion_matrix(y_test, predict_linear)
print(cm_linear)
acc_linear = accuracy_score(y_test, predict_linear)
print('Accuracy Score = %f'% acc_linear)
print(metrics.classification_report(y_test, predict_linear))
confusionmatrix_plot(y_test, predict_linear)


# # kernel = poly

# In[ ]:


cls_poly = svm.SVC(kernel='poly')
cls_poly.fit(x_train, y_train)
predict_poly = cls_poly.predict(x_test)
cm_poly = confusion_matrix(y_test, predict_poly)
print(cm_poly)
acc_poly = accuracy_score(y_test, predict_poly)
print('Accuracy Score = %f'% acc_poly)
print(metrics.classification_report(y_test, predict_poly))
confusionmatrix_plot(y_test, predict_poly)


# # rbf kernel 

# In[ ]:


cls_rbf = svm.SVC(kernel='rbf')
cls_rbf.fit(x_train, y_train)
predict_rbf = cls_rbf.predict(x_test)
cm_rbf = confusion_matrix(y_test, predict_rbf)
print(cm_rbf)
acc_rbf = accuracy_score(y_test, predict_rbf)
print('Accuracy Score = %f'% acc_rbf)
print(metrics.classification_report(y_test, predict_rbf))
confusionmatrix_plot(y_test, predict_rbf)


# # comparision of kernels 

# In[ ]:


z = pd.DataFrame(data = [acc_linear,acc_poly,acc_rbf],index = ['Linear','Polynomial','RBF'],columns=[ "Accuracy"])
z.plot(kind = 'bar',title="Accuracy vs Algorithm for GPU kernel performance")
plt.show()


# # Decision Tree

# In[ ]:


dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
# use the model to make predictions with the test data
y_pred_tree = dtree.predict(X_test)
count_misclassified = (y_test != y_pred_tree).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print('Accuracy: {:.2f}'.format(accuracy_tree))
confusionmatrix_plot(y_test, y_pred_tree)


# In[ ]:


# using GridSearchCV to find out best values of the hyper parameters
parameters={'min_samples_split' : range(10,500,50),'max_depth': range(3,11,2), 'criterion': ["gini", "entropy"],
            'splitter': ["best", "random"], 'max_leaf_nodes': range(5,15,2)}
clf = GridSearchCV(dtree, parameters, cv=10)
clf.fit(x_train,y_train)
print(clf.best_estimator_)
y_pred_grid = clf.predict(x_test)
count_misclassified = (y_test != y_pred_grid).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy_tree_grid = accuracy_score(y_test, y_pred_grid)
print('Accuracy: {:.2f}'.format(accuracy_tree_grid))


# In[ ]:


# probability prediction of 1's for ROC curve
y_prob = dtree.predict_proba(x_test)[:,1]
plot_roc_curve(y_test, y_prob, 'Decision Tree')

title = r"Learning Curve for Decision tree on train dataset"
estimator = dtree
learning_curve(estimator, title, x_train, y_train, None, 10, None)
#boosting

abc = AdaBoostClassifier(n_estimators=400,learning_rate=1)
model_ada = abc.fit(x_train, y_train)

y_pred_boosting = model_ada.predict(x_test)

print("ADABOOST Accuracy:", metrics.accuracy_score(y_test,y_pred_boosting))


# # XGBoost

# In[ ]:


# # XGBoost
model_gb = XGBClassifier()
model_gb.fit(x_train, y_train)
# make predictions for test data
y_pred_gb = model_gb.predict(x_test)
predictions1 = [round(value) for value in y_pred_gb]
accuracy1 = accuracy_score(y_test, predictions1)
print("XGBoost Accuracy: ", accuracy1)
y_prob = model_gb.predict_proba(x_test)[:,1]
plot_roc_curve(y_test, y_prob, 'XG Boost')


# In[ ]:


# pruning with boosting
parameters = {'objective':['binary:logistic'],
              'learning_rate': [0.05, 0.1],
              'max_depth': [6,7,8],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5, 50, 100],
              'missing':[-999],
              'seed': [1337]}


# In[ ]:


clf = GridSearchCV(model_gb, parameters,
                   cv=5,
                   scoring='roc_auc',
                   verbose=2, refit=True)
clf.fit(x_train,y_train)
print(clf.best_estimator_)
y_pred_grid = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_grid)
print('Accuracy XGBoost Grid: {:.2f}'.format(accuracy))

y_prob = clf.predict_proba(x_test)[:,1]
roc_curve_plot(y_test, y_prob, 'XG Boost after GridSearch CV')


# In[ ]:


title = r"Learning Curve for XG Boost on train dataset"
estimator = clf
learning_curve(estimator, title, x_train, y_train, None, 5, None)

