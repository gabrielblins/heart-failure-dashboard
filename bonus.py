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
from trainandval import X_train_res_t, X_test_t, y_train_res_t, y_test_t, X_train_s_res, y_train_s, X_test_s, y_test, \
                        X_train_st_res, y_train_st, X_test_st
warnings.filterwarnings('ignore')

"""## Bonus Section
---

### Hyperparameter tuning to get better scores

* Saving variables
"""

job.dump([X_train_res_t,X_test_t,y_train_res_t,y_test_t], 'Variables/train_test_for_rf.pkl')

job.dump([X_train_res_t,y_train_res_t],'Variables/X_y_for_rf_val.pkl')

"""#### Using optuna to find the best hyperparameters for Random Forest"""

def objective_rf(trial):

    X,y = job.load('Variables/X_y_for_rf_val.pkl')

    rf_n_estimators = trial.suggest_int("rf_n_estimators", 30, 200)
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 15)
    # Number of features to consider at every split
    rf_max_features = trial.suggest_categorical("rf_max_features", ['auto', 'sqrt'])
    # Minimum number of samples required to split a node
    rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 5)
    # Minimum number of samples required at each leaf node
    rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 5)

    classifier_obj = RandomForestClassifier(
        max_depth=rf_max_depth, n_estimators=rf_n_estimators, max_features=rf_max_features,
        min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf, random_state=5
    )

    for step in range(100):
        # Report intermediate objective value.
        intermediate_value = np.mean(cross_val_score(classifier_obj,X,y,cv=5,scoring='f1'))
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
        return intermediate_value


# study_rf = optuna.create_study(direction="maximize")
# study_rf.optimize(objective_rf, n_trials=500, timeout=300)
study_rf = job.load('Study/study_rf.pkl')

# Calculating the pruned and completed trials
pruned_trials = [t for t in study_rf.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study_rf.trials if t.state == optuna.trial.TrialState.COMPLETE]

# "Number of finished trials: ", len(study_rf.trials))
pruned = f"Number of pruned trials: {len(pruned_trials)}"
complete = f"Number of complete trials: {len(complete_trials)}"
print("Best trial:")
trial_rf = study_rf.best_trial

val_best = "    Value: {}".format(trial_rf.value)
# print("  Params: ")
for key, value in trial_rf.params.items():
    print("    {}: {}".format(key, value))

optimization_hist_rf = optuna.visualization.plot_optimization_history(study_rf)

parallel_coordinate_rf = optuna.visualization.plot_parallel_coordinate(study_rf)

# clf_rf_best = RandomForestClassifier(n_estimators=145, max_depth=10, max_features='sqrt', min_samples_split=3,
#                                      min_samples_leaf=2, random_state=5)
# clf_rf_best.fit(X_train_res_t,y_train_res_t)
clf_rf_best = job.load('Model/clf_rf_best.pkl')

"""* With hyperparameter tuning the model scored better"""

y_pred_b = clf_rf_best.predict(X_test_t)
report_rf_best = classification_report(y_test_t, y_pred_b, output_dict=True)

figrfbestcm = plt.figure(figsize=(8,6))
ax = figrfbestcm.add_subplot(1,1,1)
hmap = sns.heatmap(confusion_matrix(y_test_t, clf_rf_best.predict(X_test_t)),cmap='viridis', annot=True, ax=ax)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.show()

figrocrfbest = plt.figure()
ax = figrocrfbest.add_subplot(1,1,1)
fpr, tpr, thresholds = roc_curve(y_test_t,clf_rf_best.predict(X_test_t))
roc_auc = auc(fpr,tpr)
roc_rfbest = RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
ax.plot([0,1],[0,1],'--')
roc_rfbest.plot(ax=ax)
plt.title('ROC Curve with AUC score for Random Forest')
plt.xlim([0,1.03])
plt.ylim([0,1.03])
#plt.show()

features = X_train_res_t.columns.to_list()
features_val = clf_rf_best.feature_importances_

features_importances = {}

for key,value in zip(features,features_val):
    features_importances[key] = value

sort_features_rf_best = dict(sorted(features_importances.items(), key=lambda item: item[1], reverse=True))

# for key,value in zip(sort_features.keys(), sort_features.values()):
    # print(f'{key}:',value)

best_features = [feature for feature in sort_features_rf_best.keys() if sort_features_rf_best[feature] > 0.08]
#best_features

"""#### Using SVC to predict the target"""

from sklearn.svm import SVC
svm = SVC(random_state=5)
svm.fit(X_train_s_res, y_train_s)

"""* The SVC for data without time scored better than the Decision Tree and similar to the Random Forest"""

y_pred = svm.predict(X_test_s)
report_svm = classification_report(y_test, y_pred, output_dict=True)

"""* The SVC scored better with Time feature"""

svm.fit(X_train_st_res, y_train_st)
y_pred = svm.predict(X_test_st)
report_svm_time = classification_report(y_test_t, y_pred, output_dict=True)

"""#### Using XGBoost

* XGBoost is the state-of-the-art for tabular data
"""

from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.01, n_estimators=180, random_state=5)
xgb.fit(X_train_res_t, y_train_res_t)

"""* With no tuning the XGB scores similar to SVC"""

y_pred = xgb.predict(X_test_t)
report_xgb = classification_report(y_test_t, y_pred, output_dict=True)

"""##### Tuning XGB hyperparameters"""

import xgboost as xgb
from sklearn.metrics import f1_score
def objective(trial):
    train_x, valid_x, train_y, valid_y = job.load('Variables/train_test_for_rf.pkl')
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["n_estimators"] = trial.suggest_int("n_estimators", 50, 500, step=10)
        param["max_depth"] = trial.suggest_int("max_depth", 3, 21, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    f1score = f1_score(valid_y, pred_labels, average='weighted')
    return f1score


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=500, timeout=300)
study = job.load('Study/study_xgb.pkl')

finished = f"Number of finished trials: {len(study.trials)}"
print("Best trial:")
trial = study.best_trial

Value = "  Value: {}".format(trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

optimization_hist_xgb = optuna.visualization.plot_optimization_history(study)

parallel_coordinate_xgb = optuna.visualization.plot_parallel_coordinate(study, params=['n_estimators', 'alpha', 'eta',
                                                                                       'gamma', 'lambda', 'max_depth',
                                                                                       'min_child_weight', 'rate_drop'])

best_param_xgb = trial.params
best_param_xgb["verbosity"] = 0
best_param_xgb["objective"] = "binary:logistic"
best_param_xgb["tree_method"] = "exact"

# xgb_best = XGBClassifier(**best_param_xgb)
# xgb_best.fit(X_train_res_t,y_train_res_t)
xgb_best = job.load('Model/xgb_best.pkl')

"""* With hyperparameter tuning our XGB model made the Best score of all models"""

y_pred_best = xgb_best.predict(X_test_t)
report_xgb_best = classification_report(y_test_t,y_pred_best, output_dict=True)

figxgbbestcm = plt.figure(figsize=(8,6))
ax = figxgbbestcm.add_subplot(1,1,1)
hmap1 = sns.heatmap(confusion_matrix(y_test_t, y_pred_best), cmap='viridis', annot=True, ax=ax)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.show()

figrocxgbbest = plt.figure()
ax = figrocxgbbest.add_subplot()
fpr, tpr, thresholds = roc_curve(y_test_t,y_pred_best)
roc_auc = auc(fpr,tpr)
roc_xgbbest = RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
ax.plot([0,1],[0,1],'--')
roc_xgbbest.plot(ax=ax)
plt.title('ROC Curve with AUC score for Random Forest')
plt.xlim([0,1.03])
plt.ylim([0,1.03])
#plt.show()

features = X_train_res_t.columns.to_list()
features_val = xgb_best.feature_importances_

features_importances = {}

for key,value in zip(features,features_val):
    features_importances[key] = value

sort_features_xgb_best = dict(sorted(features_importances.items(), key=lambda item: item[1], reverse=True))

"""## Conclusions
---

As we can see, without time variable, our models made predictions only based in the biological and life style features, for the decision trees the most important features to predict if the patient will die are
* serum_creatinine: 0.30715518808747294
* ejection_fraction: 0.2603599810712565
* creatinine_phosphokinase: 0.23548816839781692
* platelets: 0.10640856066692554

And for the Random Forest the best are:
* serum_creatinine: 0.24959796944873913
* ejection_fraction: 0.18246742539309144
* platelets: 0.13973738494982005
* age: 0.13552502445670014
* creatinine_phosphokinase: 0.13455543718121094
"""

# for key,value in zip(sort_features.keys(), sort_features.values()):
#     print(f'{key}:',value)

objective_rf_code = """
def objective_rf(trial):

    X,y = job.load('Variables/X_y_for_rf_val.pkl')

    rf_n_estimators = trial.suggest_int("rf_n_estimators", 30, 200)
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 15)
    # Number of features to consider at every split
    rf_max_features = trial.suggest_categorical("rf_max_features", ['auto', 'sqrt'])
    # Minimum number of samples required to split a node
    rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 5)
    # Minimum number of samples required at each leaf node
    rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 5)

    classifier_obj = RandomForestClassifier(
        max_depth=rf_max_depth, n_estimators=rf_n_estimators, max_features=rf_max_features,
        min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf, random_state=5
    )

    for step in range(100):
        # Report intermediate objective value.
        intermediate_value = np.mean(cross_val_score(classifier_obj,X,y,cv=5,scoring='f1'))
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
        return intermediate_value
"""

objective_code = """
import xgboost as xgb
from sklearn.metrics import f1_score
def objective(trial):
    train_x, valid_x, train_y, valid_y = job.load('Variables/train_test_for_rf.pkl')
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["n_estimators"] = trial.suggest_int("n_estimators", 50, 500, step=10)
        param["max_depth"] = trial.suggest_int("max_depth", 3, 21, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    f1score = f1_score(valid_y, pred_labels, average='weighted')
    return f1score
"""