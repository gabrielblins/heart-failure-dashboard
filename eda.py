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
warnings.filterwarnings('ignore')

"""## Dataset description
---

| Feature                  | Explanation                                               | Measurement      | Range               |
|--------------------------|-----------------------------------------------------------|------------------|---------------------|
| Age                      | Age of the patient                                        | Years            | [40,..., 95]        |
| Anaemia                  | Decrease of red blood cells or hemoglobin                 | Boolean          | 0, 1                |
| Creatinine phosphokinase | Level of the CPK enzyme in the blood                      | mcg/L            | [23,..., 7861]      |
| Diabetes                 | If the patient has diabetes                               | Boolean          | 0, 1                |
| Ejection fraction        | Percentage of blood leaving the heart at each contraction | Percentage       | [14,..., 80]        |
| High blood pressure      | If a patient has hypertension                             | Boolean          | 0, 1                |
| Platelets                | Platelets in the blood                                    | kiloplatelets/mL | [25.01,..., 850.00] |
| Serum creatinine         | Level of creatinine in the blood                          | mg/dL            | [0.50,..., 9.40]    |
| Serum sodium             | Level of sodium in the blood                              | mEq/L            | [114,..., 148]      |
| Sex                      | Woman or man                                              | Binary           | 0, 1                |
| Smoking                  | If the patient smokes                                     | Boolean          | 0, 1                |
| Time                     | Follow-up period                                          | Days             | [4,...,285]         |
| Death event (target)     | If the patient died during the follow-up period           | Boolean          | 0, 1                |
"""

"""
## Exploratory Data Analysis
---

### Loading the dataset
"""

DATA_PATH = "Dados/heart_failure_clinical_records_dataset.csv"
heart_df = pd.read_csv(DATA_PATH)
heart_df.head()

"""### Looking the data shape
* The dataset has 299 instances and 13 features/variables
"""

heart_df.shape

"""### Checking features data types
* The features are already numerical, so isn't necessary to apply object to numerical conversion
"""

heart_df.info()

"""### Statistical description of the dataset"""

heart_df.describe()

"""### Checking for missing values
* The dataset don't have missing values
"""

msno.bar(heart_df)

"""### Looking at the target variable (DEATH_EVENT)
* The target output is imbalanced
"""

heart_df.DEATH_EVENT.value_counts()

"""#### Count plot of the target variable"""

sns.countplot(data=heart_df, x='DEATH_EVENT')
valores = heart_df.DEATH_EVENT.value_counts().values
print('Don\'t Died: {:.2f}% of cases ({:.0f})'.format(valores[0]/(valores[1]+valores[0])*100, valores[0]))
print('Died: {:.2f}% of cases ({:.0f})'.format(valores[1]/(valores[1]+valores[0])*100, valores[1]))
print('Proportion of the output: {:.2f}'.format(valores[0]/valores[1]))

"""### Correlation between variables"""

plt.figure(figsize=(10,8))
mask = np.triu(np.ones(heart_df.corr().shape[0]))
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if i==j:
            mask[i,j] = 0
        else:
            continue
sns.heatmap(heart_df.corr(), annot=True, cmap='viridis', fmt='.3f')#, mask=mask)

"""### Features correlations for pearson's score higher than 0.1 (absolute)"""

plt.figure(figsize=(10,8))
sns.heatmap(heart_df.corr()[np.abs(heart_df.corr()) > 0.1] , annot=True, cmap='viridis', fmt='.3f')#, mask=mask)

"""### Dropping features with less than 1% of correlation with our target"""

bigger_than_1perc = np.abs(heart_df.corr()['DEATH_EVENT']) > 0.01
new_features_list = heart_df.corr()[bigger_than_1perc]['DEATH_EVENT'].index.to_list()
heart_df_new = heart_df[new_features_list]
heart_df_new.head()

"""### Checking the mean values for the features relative to the DEATH_EVENT variable
* There are slightly differences between the two outputs for DEATH_EVENT
"""

heart_df_new.groupby(['DEATH_EVENT']).mean()

"""### Looking at variance for continuous features
* The values are in a very different scale, for some Machine Learning models the feature scaling will be necessary
"""

# Observando a variância das features contínuas
a = heart_df_new[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']].var()
make_float = lambda x: "{:.2f}".format(x)
a.apply(make_float)

"""### Looking at features Histograms"""

fig = plt.figure(figsize=(16,10))
fig.suptitle('Histogram for all features', fontsize=14, ha='center')
count = 1
for column in heart_df_new.columns.to_list():
    ax = fig.add_subplot(3,4,count)
    ax.set_title(f'{column}')
    sns.histplot(data=heart_df_new, x=column, ax=ax)
    count+=1
fig.tight_layout(pad=1.5)
fig.subplots_adjust(top=.915)
plt.show()

"""### Looking at Boxplots for continuous features
* There are some outliers that need to be removed to not mess with our classification model
"""

fig = plt.figure(figsize=(16,10))
fig.suptitle('Boxplot for continuous features', fontsize=14, ha='center')
count = 1
columns_continuous = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
for column in columns_continuous:
    #plt.subplot(7,2,count)
    ax = fig.add_subplot(2,4,count)
    ax.set_title(f'{column}')
    sns.boxplot(data=heart_df_new, y=column, ax=ax)
    count+=1
fig.tight_layout(pad=1.5)
fig.subplots_adjust(top=.915)
plt.show()
