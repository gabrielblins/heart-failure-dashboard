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
import joblib as job
from eda import heart_df,heart_df_new, columns_continuous
warnings.filterwarnings('ignore')

"""## Removing noise from data
   ---

### Removing outliers

Using Z-score outlier metric ( mean +/- 3 * standard deviation )
"""

# Using score Z outlier metric ( mean +/- 3 * standard deviation )
def remove_outliers(df, continuous_features=None, outliers_index=False):

    if continuous_features:
        features_list = continuous_features
    else:
        features_list = df.columns.to_list()
    boolean_mask = np.array([])
    print(df.shape)

    for feature in features_list:
        mean = df[feature].mean()
        std = df[feature].std()
        upper = mean + 3*std
        lower = mean - 3*std
        mask_f = (df[feature] < upper) & (df[feature] > lower)
        if feature == features_list[0]:
            boolean_mask = mask_f
        else:
            boolean_mask = np.vstack((boolean_mask,mask_f.values))

    no_outlier_idx = np.all(boolean_mask.T, axis=1)
    df_no_outlier = df[no_outlier_idx]
    print(df_no_outlier.shape)
    if outliers_index:
        return df_no_outlier, ~no_outlier_idx
    else:
        return df_no_outlier

heart_df_no_outlier, outliers_ind = remove_outliers(heart_df_new, columns_continuous, outliers_index=True)

#outliers = heart_df[outliers_ind]
#outliers

#outliers.DEATH_EVENT.value_counts()

fignout = plt.figure()
ax = fignout.add_subplot(1,1,1)
cplot = sns.countplot(data=heart_df_no_outlier, x='DEATH_EVENT', ax=ax)
valoresnout = heart_df_no_outlier.DEATH_EVENT.value_counts().values
# print('Não morreram: {:.2f}% dos casos'.format(valoresnout[0]/(valoresnout[1]+valoresnout[0])*100))
# print('Morreram: {:.2f}% dos casos'.format(valoresnout[1]/(valoresnout[1]+valoresnout[0])*100))
# print('Proporção: {:.2f}'.format(valoresnout[0]/valoresnout[1]))

figboxnout = plt.figure(figsize=(10,35))
figboxnout.suptitle('Boxplot comparison for data with and without outliers', fontsize=16, x = 0.545)#, dpi=300)
count = 1
for column in columns_continuous:
    #plt.subplot(7,2,count)
    ax = figboxnout.add_subplot(7,2,count)
    ax.set_title('With Outliers')
    sns.boxplot(data=heart_df, y=column, ax=ax)
    count+=1
    #plt.subplot(7,2,count)
    ax = figboxnout.add_subplot(7,2,count)
    ax.set_title('Without Outliers')
    sns.boxplot(data=heart_df_no_outlier, y=column, color='orange', ax=ax)
    count +=1
figboxnout.tight_layout(pad = 1.5)
figboxnout.subplots_adjust(top=.96)
#plt.show()

fighistnout = plt.figure(figsize=(12,35))
fighistnout.suptitle('Histogram comparison for data with and without outliers', fontsize=16, x = 0.545)#, dpi=300)
count = 1

for column in columns_continuous:
    ax = fighistnout.add_subplot(7,2,count)
    ax.set_title(f'With Outliers (Mean: {heart_df_new[column].mean()})')
    #sns.histplot(data=heart_df_new, x=column, hue='DEATH_EVENT', kde=True, ax = ax)
    sns.distplot(a=heart_df_no_outlier[heart_df_no_outlier['DEATH_EVENT'] == 0][column], bins=15, ax=ax, label='Não Faleceu')
    sns.distplot(a=heart_df_no_outlier[heart_df_no_outlier['DEATH_EVENT'] == 1][column], bins=15, ax=ax, label='Faleceu')
    plt.legend()
    count+=1

    ax = fighistnout.add_subplot(7,2,count)
    ax.set_title(f'Without Outliers (Mean: {heart_df_no_outlier[column].mean()})')
    #sns.histplot(data=heart_df_no_outlier, x=column, hue='DEATH_EVENT', color='orange', kde=True, ax = ax)
    sns.distplot(a=heart_df_no_outlier[heart_df_no_outlier['DEATH_EVENT'] == 0][column], bins=15, ax=ax, label='Não Faleceu')
    sns.distplot(a=heart_df_no_outlier[heart_df_no_outlier['DEATH_EVENT'] == 1][column], bins=15,ax=ax, label='Faleceu')
    plt.legend()
    count+=1

fighistnout.tight_layout(pad = 1.5)
fighistnout.subplots_adjust(top=.96)
#plt.show()

remove_outliers_code = """
def remove_outliers(df, continuous_features=None, outliers_index=False):

    if continuous_features:
        features_list = continuous_features
    else:
        features_list = df.columns.to_list()
    boolean_mask = np.array([])
    print(df.shape)

    for feature in features_list:
        mean = df[feature].mean()
        std = df[feature].std()
        upper = mean + 3*std
        lower = mean - 3*std
        mask_f = (df[feature] < upper) & (df[feature] > lower)
        if feature == features_list[0]:
            boolean_mask = mask_f
        else:
            boolean_mask = np.vstack((boolean_mask,mask_f.values))

    no_outlier_idx = np.all(boolean_mask.T, axis=1)
    df_no_outlier = df[no_outlier_idx]
    print(df_no_outlier.shape)
    if outliers_index:
        return df_no_outlier, ~no_outlier_idx
    else:
        return df_no_outlier
"""