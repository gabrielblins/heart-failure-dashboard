import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTEN
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from rmvnoise import heart_df_no_outlier, columns_continuous
import joblib as job
warnings.filterwarnings('ignore')

"""## Training and Validation
---

### Choosing our features and setting our target
"""

# sem tempo, com tempo, sem tempo scale, com tempo scale, sem tempo discreto, com tempo discreto
X = heart_df_no_outlier.drop(['DEATH_EVENT','time'], axis=1)
X_t = heart_df_no_outlier.drop('DEATH_EVENT', axis=1)
y = heart_df_no_outlier['DEATH_EVENT']

"""### Splitting our data into training and test"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=5, stratify=y)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t,y,test_size=.3,random_state=5,stratify=y)

"""### Scaling and Discretizing the data"""

columns_continuous_not = columns_continuous.copy()
columns_continuous_not.pop()

discretizer = EqualFrequencyDiscretiser(variables=columns_continuous_not, q=8)
discretizer_t = EqualFrequencyDiscretiser(variables=columns_continuous, q=8)

#scaler = StandardScaler()
scaler = MinMaxScaler()

#Discrete without Time
discrete = discretizer.fit_transform(X_train)
X_train_d = pd.DataFrame()
for column in X_train.columns.to_list():
    if column in discrete.columns.to_list():
        X_train_d[column] = discrete[column]
    else:
        X_train_d[column] = X_train[column]
discrete_test = discretizer.transform(X_test)
X_test_d = pd.DataFrame()
for column in X_test.columns.to_list():
    if column in discrete_test.columns.to_list():
        X_test_d[column] = discrete_test[column]
    else:
        X_test_d[column] = X_test[column]

# Discrete with Time
discrete_t = discretizer_t.fit_transform(X_train_t)
X_train_dt = pd.DataFrame()
for column in X_train_t.columns.to_list():
    if column in discrete_t.columns.to_list():
        X_train_dt[column] = discrete_t[column]
    else:
        X_train_dt[column] = X_train_t[column]
discrete_test_t = discretizer_t.transform(X_test_t)
X_test_dt = pd.DataFrame()
for column in X_test_t.columns.to_list():
    if column in discrete_test_t.columns.to_list():
        X_test_dt[column] = discrete_test_t[column]
    else:
        X_test_dt[column] = X_test_t[column]
# Scaled without Time
X_train_s = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns.to_list())
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns.to_list())

#Scaled with Time
X_train_st = pd.DataFrame(scaler.fit_transform(X_train_t), columns=X_train_t.columns.to_list())
X_test_st = pd.DataFrame(scaler.transform(X_test_t), columns=X_test_t.columns.to_list())

"""### Applying SMOTE to handle with imbalanced data"""

smote = SMOTENC(categorical_features= [1, 4, 8],random_state=5)
smotec = SMOTEN(random_state=5)

X_train_res, y_train_res = smote.fit_resample(X_train,y_train)
X_train_res_t, y_train_res_t = smote.fit_resample(X_train_t,y_train_t)

X_train_d_res, y_train_d = smotec.fit_resample(X_train_d,y_train)
X_train_dt_res, y_train_dt = smotec.fit_resample(X_train_dt,y_train_t)

X_train_s_res, y_train_s = smote.fit_resample(X_train_s,y_train)
X_train_st_res, y_train_st = smote.fit_resample(X_train_st,y_train_t)

"""### Choosing the best Tree based model

#### Using Decision Tree to make the initial classification
* The Classification Report show some important metrics to evaluate the tree classifier
* The plot show how the tree split to fit the training data
* The scores can be improved if we use Random Forests instead of one Decision Tree
"""

clf_tree = DecisionTreeClassifier(random_state=5, max_depth=5)
clf_tree.fit(X_train_res,y_train_res)
report_tree = classification_report(y_test, clf_tree.predict(X_test), output_dict=True)
tree_fig = plt.figure(figsize=(30,20))
ax = tree_fig.add_subplot(1,1,1)
plot_tree(clf_tree, feature_names=X_train_res.columns.to_list(), class_names=['Recovered', 'Died'], fontsize=13, ax=ax)
plt.savefig('tree.png')
#plt.show()

"""##### Confusion Matrix for decision tree"""

figtreecm = plt.figure(figsize=(8,6))
ax = figtreecm.add_subplot(1,1,1)
sns.heatmap(confusion_matrix(y_test, clf_tree.predict(X_test)),cmap='viridis', annot=True, ax=ax)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.show()

"""##### ROC curve for Decision Tree"""
figroc = plt.figure()
ax = figroc.add_subplot(1,1,1)
fpr, tpr, thresholds = roc_curve(y_test,clf_tree.predict(X_test))
roc_auc = auc(fpr,tpr)
roctree = RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
#roctreea = RocCurveDisplay.from_estimator(clf_tree, X_test, y_test, ax=ax)
ax.plot([0,1],[0,1], '--')
roctree.plot(ax=ax)
plt.title('ROC Curve with AUC score for Decision Tree ')
plt.xlim([0,1.03])
plt.ylim([0,1.03])
#plt.show()

"""##### Feature importances for the decision tree classifier"""

features = X_train_res.columns.to_list()
features_val = clf_tree.feature_importances_

features_importances = {}

for key,value in zip(features,features_val):
    features_importances[key] = value

sort_features_tree = dict(sorted(features_importances.items(), key=lambda item: item[1], reverse=True))

# print('Feature Importances for Decision Tree:\n')
# for key,value in zip(sort_features_tree.keys(), sort_features_tree.values()):
    # print(f'{key}:',value)

"""#### Using Random Forest to improve the score
* As expected, the scores were better
"""

clf_rf = RandomForestClassifier(random_state=5)
clf_rf.fit(X_train_res,y_train_res)
report_rf = classification_report(y_test,clf_rf.predict(X_test), output_dict=True)

"""##### Confusion matrix for Random Forest"""

figrfcm = plt.figure(figsize=(8,6))
ax = figrfcm.add_subplot(1,1,1)
rf_cm = sns.heatmap(confusion_matrix(y_test, clf_rf.predict(X_test)),cmap='viridis', annot=True, ax=ax)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

"""##### ROC curve for Random Forest"""
figrocrf = plt.figure()
ax = figrocrf.add_subplot(1,1,1)
fpr, tpr, thresholds = roc_curve(y_test,clf_rf.predict(X_test))
roc_auc = auc(fpr,tpr)
rocrf = RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
ax.plot([0,1],[0,1], '--')
rocrf.plot(ax=ax)
plt.title('ROC Curve with AUC score for Random Forest')
plt.xlim([0,1.03])
plt.ylim([0,1.03])
#plt.show()

"""##### Features Importances for Random Forest"""

features = X_train_res.columns.to_list()
features_val = clf_rf.feature_importances_

features_importances = {}

for key,value in zip(features,features_val):
    features_importances[key] = value

sort_features_rf = dict(sorted(features_importances.items(), key=lambda item: item[1], reverse=True))

# for key,value in zip(sort_features.keys(), sort_features.values()):
    # print(f'{key}:',value)

"""### Trying to improve the scores using the other train/test sets

* With Time feature
"""

clf_rf.fit(X_train_res_t,y_train_res_t)
y_pred = clf_rf.predict(X_test_t)
report_rf_time = classification_report(y_test_t,y_pred, output_dict=True)

"""* Without Time, Discretized"""

clf_rf.fit(X_train_d_res,y_train_d)
y_pred = clf_rf.predict(X_test_d)
report_rf_disc = classification_report(y_test,y_pred,output_dict=True)

"""* With Time, Discretized"""

clf_rf.fit(X_train_dt_res,y_train_dt)
y_pred = clf_rf.predict(X_test_dt)
report_rf_disc_time = classification_report(y_test_t,y_pred,output_dict=True)
