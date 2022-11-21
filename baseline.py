# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
sns.set()
## 3407 is all you need
random_seed = 3407

## load origin data
train = pd.read_csv('./data/dataTrain.csv')
test = pd.read_csv('./data/dataB.csv')
train_appendix = pd.read_csv('./data/dataNoLabel.csv')
submission = pd.read_csv('./data/submit_example_B.csv')
# train.head()

## get features` names
feature_location = ['f1', 'f2', 'f4', 'f5', 'f6']
feature_conversation = ['f43', 'f44', 'f45', 'f46']
feature_internet = ['f3', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17',
                    'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28',
                    'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39',
                    'f40', 'f41', 'f42']

## feature engineering pipelie
## consequently get: train, train_appendix, test
data = pd.concat([train, train_appendix, test]).reset_index(drop=True)
# 1.1. Serial number code
data['f3'] = data['f3'].map({'mid':1, 'high':2, 'low':0})
# 1.2. Features combination
fe_basic_four(feature_location)
fe_basic_four(feature_conversation)
trick_1()
fe_basic_four(['f47','f3','f19','f8'])
# 1.3. Data cleaning
train = data.iloc[0:59872,:]
### do a preliminary to illuminate useless data
#gbc_test_preds = model_train(train_x, train_y, test_x,
#GradientBoostingClassifier(), "GradientBoostingClassifier", 60, False)
###
train = train[:50000] # After data cleaning
train_appendix = data.iloc[59872:99756,:]
test = data.iloc[99756:,:]
#data.describe()

## preparing training and testing data
features = [i for i in train.columns if i not in ['label', 'id']]
train_x = train[features]
train_y = train['label']
test_x = test[features]

## initialize stacking model, which is comprised of (xgbc, lgbm, cbc,...)
xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    gamma=0,
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    random_state=random_seed
)
cbc = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=1,
    loss_function='Logloss',
    verbose=0,
    random_seed=random_seed
)
lgbm = LGBMClassifier(
    objective = 'binary',
    boosting_type = 'gbdt',
    metric = 'auc',
    learning_rate = 0.05,
    num_leaves = 2 ** 6,
    max_depth = 8,
    tree_learner = 'serial',
    colsample_bytree = 0.8,
    subsample_freq = 1,
    subsample = 0.8,
    #num_boost_round = 5000,
    max_bin = 255,
    verbose = -1,
    seed = random_seed,
    bagging_seed = random_seed,
    feature_fraction_seed = random_seed
)
estimators = [
    ('cbc', cbc),
    ('lgbm', lgbm),
    ('xgbc', xgbc)
]
stk = StackingClassifier(
    estimators = estimators,
    final_estimator = LogisticRegression()
)
clf = stk

# 1.4 feature selection
useful_features,_ = rank_features(train_x, train_y)
_,_ = rank_features(train_x[useful_features], train_y, False)
### SAVE&LOAD
# arr_useful_features = np.array(useful_features)
# np.save('./arr_useful_features.npy',arr_useful_features)
# load
# arr_useful_features = np.load('../input/attributes/arr_useful_features.npy')
# useful_features = arr_useful_features.tolist()
###
train_x = train_x[useful_features]
test_x = test_x[useful_features]

### GRID Search: TUNE THE HIPER-PARAMETERS
# xgbc = XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='auc',
#     gamma=0,
#     n_estimators=100,
#     max_depth=5,
#     random_state=random_seed
# )
# learning_rate = [0.05,0.1,0.3]
# param_grid = dict(learning_rate = learning_rate)
# skf = StratifiedKFold(n_splits=12, random_state=random_seed, shuffle=True)

# grid_search = GridSearchCV(xgbc,param_grid,scoring = 'roc_auc',n_jobs=-1,cv = skf)
# #scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
# grid_result = grid_search.fit(train_x, train_y) #运行网格搜索
# print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
# #grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
# #best_score_：成员提供优化过程期间观察到的最好的评分
# #具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
# #注意，“params”键用于存储所有参数候选项的参数设置列表。
# means = grid_result.cv_results_['mean_test_score']
# params = grid_result.cv_results_['params']
# for mean,param in zip(means,params):
#     print("%f  with:   %r" % (mean,param))
###

### semi-supre train
# train_appendix_x = train_appendix[useful_features]
# train_appendix_y_preds_clf = basic_model_train(train_x, train_y, train_appendix_x)
# ## choose no label data
# ta_p90_mask = select_threshold(train_appendix_y_preds_clf, 0.04, 0.96)
# ta_p90_x = train_appendix_x[ta_p90_mask]
# ta_p90_y = train_appendix_y_preds_clf[ta_p90_mask]
# ta_p90_y = np.around(ta_p90_y)
# train_x['label'] = train_y
# ta_p90_x['label'] = ta_p90_y
# train_x = pd.concat([train_x, ta_p90_x]).reset_index(drop=True)
# train_x = train_x[useful_features]
# train_y = train_x['label']
###

## train & predict
test_y_preds_clf = model_train(train_x, train_y, test_x, clf, "StackingClassifier", 12)

## submission
submission['label'] = test_y_preds_clf
submission.to_csv('./results.csv', index=False)

################FUNCTIONS###############
## feature engineering functions
# 1. violence +-*/
def fe_basic_four(feature_names, epsilon=0.05):
    n = len(feature_names)
    for i in range(n):
        for j in range(n):
            if (i != j):
                data[f'{feature_names[i]}+{feature_names[j]}'] = data[feature_names[i]] + data[feature_names[j]]
                data[f'{feature_names[i]}-{feature_names[j]}'] = data[feature_names[i]] - data[feature_names[j]]
            data[f'{feature_names[i]}*{feature_names[j]}'] = data[feature_names[i]] * data[feature_names[j]]
            data[f'{feature_names[i]}/{feature_names[j]}'] = data[feature_names[i]] / (data[feature_names[j]] + epsilon)
# 2. linear conbination
def trick_1():
    data['f47'] = data['f1'] * 10 + data['f2']

## K-fold cross validation/Bagging framework
def model_train(train_x, train_y, test_x, model, model_name, kfold=5, s_switch=True, display_each_fold=True):
    oof_preds = np.zeros((train_x.shape[0]))
    test_preds = np.zeros(test_x.shape[0])
    skf = StratifiedKFold(n_splits=kfold, random_state=random_seed, shuffle=True)
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train_x, train_y)):
        x_train, x_test = train_x.iloc[train_index, :], train_x.iloc[test_index, :]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
        model.fit(x_train,y_train)
        y_pred = model.predict_proba(x_test)[:,1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test,y_pred)
        if display_each_fold:
            print("- KFold = %d, val_auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(test_x)[:,1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.8f" % (model_name, roc_auc_score(train_y, oof_preds)))
    return test_preds / kfold

## get useful features according to whether they benefit the results
def rank_features(train_x, train_y, c_useful_f=True):
    useful_features = []  #
    useful_features_val = []
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, stratify=train_y, random_state=random_seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print('auc = %.8f' % auc)
    if (c_useful_f):
        ff = [i for i in train_x.columns if i not in ['label', 'id']]
        for f in ff:
            print(f)
            X_test_masked = X_test.copy()
            X_test_masked[f] = 0
            auc_masked = roc_auc_score(y_test, clf.predict_proba(X_test_masked)[:, 1])
            if auc_masked < auc:
                useful_features.append(f)
                useful_features_val.append(auc_masked - auc)
                print('%5s | %.8f | %.8f' % (f, auc_masked, auc_masked - auc))
    return useful_features, useful_features_val

## semi-supre train
# 1. predict unlabel dataframe
def basic_model_train(train_x, train_y, test_x):
    clf.fit(train_x, train_y)
    test_y_pred = clf.predict_proba(test_x)[:, 1]
    return test_y_pred
# 2. choose no label data with threshold
def select_threshold(arr, lb=0.1, ub=0.9):
    return np.logical_or(arr<lb, arr>ub)