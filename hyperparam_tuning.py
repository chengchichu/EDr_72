#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:52:56 2021

@author: anpo
"""
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def tune_logistic(X,y):
 	
    # define models and parameters
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc',error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result 

def tune_rf(X, y):

    # define models and parameters
    model = RandomForestClassifier()

    rand_grid = {'bootstrap': [True, False],
      'max_depth': [10, 25, 50, 100],
      'max_features': ['sqrt','log2'],
      'min_samples_leaf': [1, 2, 4],
      'min_samples_split': [2, 5, 10],
      'n_estimators': [10, 100, 500]}
    
    # define grid search
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)    
    grid_search = RandomizedSearchCV(estimator=model, param_distributions=rand_grid, n_jobs=-1, cv=cv, scoring='roc_auc', n_iter=300)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    return grid_result   
        
def xgb_objective(space):
    clf=xgb.XGBClassifier(n_estimators = space['n_estimators'],
                    eta = space['eta'],
                    subsample = space['subsample'],
                    max_depth = space['max_depth'], 
                    gamma = space['gamma'],
                    min_child_weight=space['min_child_weight'],
                    colsample_bytree=space['colsample_bytree'],
                    use_label_encoder=False)
    evaluation = [( train_x, train_y), ( val_x, val_y)]                
    # print(evaluation)                
    clf.fit(train_x, train_y, eval_set=evaluation, eval_metric="auc", 
        early_stopping_rounds=10)
        
    pred = clf.predict_proba(train_x)[:,1]                  
    auc_ = roc_auc_score(train_y, pred)
    
    print("AUC score:", auc_)
    
    return{'loss':str(1-auc_), 'status': STATUS_OK }
    # return STATUS_OK
    


# start tuning
import pickle
with open('abdominal_pain_data.pickle', 'rb') as handle:
     data = pickle.load(handle)

X_train_c = data['Xtrain'] 
y_train_c = data['ytrain']  
preprocessed_X_test = data['Xtest'] 
y_test = data['ytest'] 
encoding_head = data['head']


# lg_params = tune_logistic(X_train_c,y_train_c)

# rf_params = tune_rf(X_train_c,y_train_c)


space={
'max_depth': hp.choice('max_depth', np.arange(3, 18, 1, dtype=int)),
'gamma': hp.uniform('gamma', 1,9),
'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
'subsample' : hp.uniform('subsample', 0.5,1),
'min_child_weight' : hp.choice('min_child_weight', np.arange(0, 10, 1, dtype=int)), 
'eta' : hp.uniform('eta', 0.005, 0.3),
'n_estimators' : hp.choice('n_estimators', np.arange(20, 200, 5, dtype=int)),
'seed': 0,
'reg_alpha' : hp.choice('reg_alpha', np.arange(40, 180, 1, dtype=int)),
'reg_lambda' : hp.uniform('reg_lambda', 0, 1)
}

train_x, val_x, train_y, val_y = train_test_split(X_train_c, y_train_c, test_size=1/8, random_state=40, stratify = y_train_c) 
    
      # define objective function
best_hyperparams = fmin(fn = xgb_objective,
                space = space,
                algo = tpe.suggest,
                max_evals = 100,
                trials = Trials())



        
        
        