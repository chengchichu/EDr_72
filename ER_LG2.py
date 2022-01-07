#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:55:50 2020

@author: chengchichu
"""

# machine learning for predicting 72hour unscheduled revisit in ER


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, average_precision_score, confusion_matrix, roc_auc_score
from numpy.random import randint, seed
from collections import Counter
from sklearn import metrics
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
import seaborn as sn
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
#from scipy.stats import chi2_contingency
from imblearn.under_sampling import OneSidedSelection, CondensedNearestNeighbour
import re
from sklearn.impute import SimpleImputer
#from imblearn.under_sampling import CondensedNearestNeighbour
#from imblearn.under_sampling import NearMiss
fconvert = np.vectorize(float)
from tabulate import tabulate
import xgboost as xgb
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import random


### Functions

def pre_encode(data,tag,mds):
    data_copy = data.copy()  # prevent mutable      
    scale_param = []
    encoder = []
    if tag == 1: # for continuous data   
       data_copy = fconvert(data_copy) # widen the precision      
       if mds:
          # standardize
          scaler = StandardScaler()
          s_data = scaler.fit_transform(data_copy)     
          scale_param = [scaler.mean_, np.sqrt(scaler.var_)]
          out = s_data
       else:
          # -1 constant imputation
          scaler = RobustScaler(with_centering = False)
          not_miss = data_copy != -1 # leave -1 intact
          s_data = scaler.fit_transform(data_copy[not_miss].reshape(-1, 1))
          idx = np.where(not_miss) 
          data_copy[idx[0]] = s_data.copy()              
          scale_param = [scaler.scale_]
          out = data_copy         
    elif tag == 0:   
       # 把這個輸出 用同一個來處理test
       encoder = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
       out = encoder.fit_transform(data_copy)
    elif tag == 2:  
       out = data_copy
    else:
       print('wrong code for preprocessing')       
     
    return out, scale_param, encoder

def test_encode(data,tag,scale_param,mds,enc):
    data_copy = data.copy()  # prevent mutable      
    if tag == 1: # ordinal encoding      
       data_copy = fconvert(data_copy) # widen the precision      
       if mds:       
          s_data = (data_copy-scale_param[0])/scale_param[1]
          out = s_data
       else:
          not_miss = data_copy != -1 # leave -1 intact
          data_true = data_copy[not_miss].reshape(-1,1)
          s_data = data_true/scale_param[0]
          idx = np.where(not_miss) 
          data_copy[idx[0]] = s_data.copy()
          out = data_copy       
    elif tag == 0:   
       #encoder = OneHotEncoder(sparse=False)
       out = enc.transform(data_copy)
    elif tag == 2:  
       out = data_copy
    else:
       print('wrong code') 
 
    return out

# 檢查是否有非數值的符號
def assert_number(data, lab):
    tmp = (data[lab].isnull()) | (data[lab].isna())
    datanot_null = data[lab][~tmp.values]
    for i,j in datanot_null.items():
        try: 
           data.at[i,lab] = pd.to_numeric(j)      
        except:
           #print(j)
           data.at[i,lab] = pd.to_numeric(re.findall("\d+", j)[0])     
    return data

def model_auc(model, preprocessed_X, y_true):
    try: 
        y_score = model.predict_proba(preprocessed_X)[:,1]
    except:       
        y_score = model.decision_function(preprocessed_X)    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(y_true, y_score)
    return roc_auc, auprc, y_score, fpr, tpr

def ml_model(clf,data_X,data_y, set_bootstrap):
    # 10 fold        
    kfold = KFold(10, True, 1)
    aucs = []
    models = []
    k_idx = []
    data_size = np.arange(0,data_X.shape[0])
    for train, test in kfold.split(data_size):
        it_idx = {}
        it_idx['train'] = train
        it_idx['test'] = test
        k_idx.append(it_idx)
        # 只對xtrain做bootstrapping
        xtrain = data_X[data_size[train],:]
        ytrain = data_y[data_size[train]]        
        
        # set this off, if already applied class weight
        if set_bootstrap == True:             
           xtrain, ytrain = bootstrap(xtrain, ytrain)
        xtest = data_X[data_size[test],:]
        ytest = data_y[data_size[test]]
        model = clf.fit(xtrain, ytrain)
        area_under_ROC = model_auc(model, xtest, ytest)
        aucs.append(area_under_ROC[0])
        models.append(model)        
    # selection model with best AUC
    bst = models[np.argmax(aucs)]
    val_mean = np.mean(aucs)
    val_sem = np.std(aucs)/np.sqrt(len(aucs))
    val_out = (val_mean, val_sem)
    return bst, models, k_idx, val_out

def model_xgb(clf,data_X,data_y, params_in):  
     # use this step to balance the data, not sure it is necessary here
     #xtrain, ytrain = bootstrap(xtrain, ytrain)
     #切給 validation set, 10% of total, because train is 80%
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=1/8, random_state=40, stratify = data_y)            
         # because I am tuning the hyperplane parameter, I don't need to do k fold here
    dtrain = clf.DMatrix(X_train, label=y_train)
    dval = clf.DMatrix(X_test, label=y_test)
         # I will fix this initially, see reference Woo Suk Hong and Andrew Talyor, 2019, thier supplementary text
    params = params_in.copy()
    
    params['objective'] = 'binary:logistic'
#              'nrounds': 20,
#              'nthread': 5}
    params['eval_metric'] = 'auc'
    # params['max_depth'] = 3
    # params['max_depth'] = 8
    # params['min_child_weight'] = 7
    # params['min_child_weight'] = 4
    # params['subsample'] = 1
    # params['colsample_bytree'] = 0.51
    # params['colsample_bytree'] = 0.96
    # params['eta'] = 0.05
    # params['n_estimators'] = 15
    # params['gamma'] = 1.02
    # params['gamma'] = 2.5
    # params['reg_alpha'] = 60
    # params['reg_alpha'] = 158
    # params['reg_lambda'] = 0.66
    # params['reg_lambda'] = 0.54
    # params['scale_pos_weight'] = 16
    num_boost_round = 999
    
    clf2 = clf.XGBClassifier(**params, use_label_encoder=False)
    # clf.train(params,dtrain,num_boost_round=num_boost_round,evals=[(dtest, "Test")], \
    #                    early_stopping_rounds=10)
    model = clf.train(params,dtrain,num_boost_round=num_boost_round,evals=[(dval, "Test")], \
                       early_stopping_rounds=10)

    # another model for voting 
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    model2 = clf2.fit(X_train, y_train,
            eval_set=evaluation,
            early_stopping_rounds=10,verbose=False)
    return model, model2

def bootstrap(datax,datay):    
    # combined with bootstrap
    # major class
    cnt = Counter(datay)
    np.random.seed(0)
    major_idx = np.random.choice(cnt[0],int(cnt[0]*1.5),replace = True)
    # minor class
    np.random.seed(1)
    minor_idx = np.random.choice(cnt[1],int(cnt[0]*1.5),replace = True)         
    X_major = datax[0:cnt[0],:] 
    X_minor = datax[cnt[0]:len(datay),:]
    y_major = datay[0:cnt[0]]
    y_minor = datay[cnt[0]:len(datay)]      
    datax_balanced = np.concatenate((X_major[major_idx,:], X_minor[minor_idx,:]), axis = 0)
    datay_balanced = np.concatenate((y_major[major_idx], y_minor[minor_idx]))    
    return datax_balanced, datay_balanced
    
def get_nan_pr(data,cols):
    
    try:
       keys = cols.keys()
    except:
       keys = cols
       
    pr = []
    for i in keys:        
        tmp = (data[i].isnull()) | (data[i].isna())
        pr.append([i,sum(tmp)/data.shape[0]])  
    return pr  

def model_result(y_test, bst, modelname, X_test):

    if modelname == 'XGB':
       dtest = xgb.DMatrix(X_test, label=y_test)
       proba = bst.predict(dtest)
    
       pred = []
       for i in proba:
           if i < 0.5:
              pred.append(0)
           else:
              pred.append(1) 
       cm = confusion_matrix(y_test, pred)
       cp = classification_report(y_test, pred, target_names=['沒回診','有回診'])       
    else:    
       cm = confusion_matrix(y_test, bst.predict(X_test))
       cp = classification_report(y_test, bst.predict(X_test), target_names=['沒回診','有回診'])
         
        # accuracy and specificity
    total=sum(sum(cm))
    accuracy1=(cm[0,0]+cm[1,1])/total
    specificity1 = cm[0,0]/(cm[0,1]+cm[0,0])   
    precision = cm[1,1]/(cm[1,1]+cm[0,1])   
    recall = cm[1,1]/(cm[1,1]+cm[1,0])
    f1 = 2*precision*recall/(precision+recall)
    other_scores = (accuracy1,specificity1,precision,recall,f1) 
    
    return cm, cp, other_scores

def table_r(cp,cm,auc_in, others):
    lines = cp.splitlines()
    c=[np.array(lines[2].split()[1:4]), np.array(lines[3].split()[1:4])]
    out = np.concatenate((cm,c),axis=1)
    cc = np.array([auc_in, np.nan])
    out = np.concatenate((out,cc.reshape(2,-1)),axis=1)
    
    for i in others:
        cc = np.array([i, np.nan])
        out = np.concatenate((out,cc.reshape(2,-1)),axis=1)
        
    tb = tabulate(out, headers=['p0', 'p1','precision','recall','f1','AUC','ACC','SP','PC','RE','F1']) 
    print(tb)
        
    return tb

def preprocess(X_train, y_train, X_test, cols, fs_to_imp):  
    
    X = X_train.copy()      
    y72 = y_train.copy() 
 
    # imputation
    md_strategy = True
    if md_strategy:    
       imp = SimpleImputer(missing_values=np.nan, strategy='median')   
    else: 
       # imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = -1) 
       imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp.fit(X[fs_to_imp]) 
    impdata = imp.transform(X[fs_to_imp])
    impdata_test = imp.transform(X_test[fs_to_imp])          
    cnt = 0
    for i in fs_to_imp:
        #print(i)
        X[i] = impdata[:,cnt]
        X_test[i] = impdata_test[:,cnt]
        cnt+=1           
    # preprocessing encoding
    preprocessed_X = []
    preprocessed_X_test = []
    encoding_head = []
    #encoding_head_ = {}
    encoding_head_flat = []
    # scale_params = {}   
    mcnt0 = []
    cnt = 0 # initial
    # train encoding
    for key, value in cols.items():
        data_col = X[key].values.reshape(-1,1)
        out, scale_param, enc = pre_encode(data_col, value, md_strategy) 
        
        data_col_test = X_test[key].values.reshape(-1,1)
        out_test = test_encode(data_col_test, value, scale_param, md_strategy, enc)
        
        # n-1 for dummy variable, this means reference group is the first column
        if out.shape[1]>1:
           out = out[:,1:(out.shape[1])]
           out_test = out_test[:,1:(out_test.shape[1])]
        
        #encoding_head_[key] = feature_name
        #ec = [key for i in range(out.shape[1])]
        if value == 0:
           feature_name = enc.get_feature_names()
           f0 = [i for i in feature_name]
           L = out.shape[1]
           N = -1-L+1
           x2 = f0[N:]           
           assert(out.shape[1] == len(x2))  
           for j in x2:
               encoding_head.append(j)
           
        else:
           encoding_head.append(key)
        #print(out.shape[1])   
        mcnt0.append(out.shape[1])
        #initialize
        if cnt == 0:
           preprocessed_X = out  
           preprocessed_X_test = out_test
        else:
           preprocessed_X = np.concatenate((preprocessed_X, out), axis = 1)
           preprocessed_X_test = np.concatenate((preprocessed_X_test, out_test), axis = 1)    
                      
        cnt += 1
    return preprocessed_X, y72, preprocessed_X_test, encoding_head
   
def cls_constructor(clfname, params):
    
    if clfname == 'lg':    
       clf = LogisticRegression(**params, random_state=0, max_iter=5000)       
    elif clfname == 'rf':    
       clf = RandomForestClassifier(**params, random_state=0)       
    elif clfname == 'xgb':  
       clf = xgb.XGBClassifier(**params, use_label_encoder=False)
    else:
       print('no matched cls')
    return clf    

    # elif clfname == 'xgb':  
    #    clf = xgb.XGBClassifier(**params,use_label_encoder=False)

def run_models(X_train_c, y_train_c, preprocessed_X_test, y_test, model_strat, encoding_head, data_root_folder, best_hyperparams, class_weight_apply, seed_tag):
    
    # 調整class weight的話, 在training的時候就不bootstrap     
    if class_weight_apply == False:
        set_bootstrap = True
    else:   
        set_bootstrap = False
    
    # set_bootstrap = False
    
    #
    params = {} 
    if class_weight_apply == False:     
        if best_hyperparams:   
          params = best_hyperparams['lg']              
    elif class_weight_apply == True:    
        if best_hyperparams:
          params = best_hyperparams['lg']
          params['class_weight'] = {0:0.05,1:0.95}
        else: 
          params['class_weight'] = {0:0.05,1:0.95}
          
    clf1 = cls_constructor('lg', params)      
    print('running LG')
    bst_lg, _, _, aucs_lg = ml_model(clf1, X_train_c, y_train_c, set_bootstrap)
    
    # LG imp
    imp = pd.DataFrame(data = abs(bst_lg.coef_[0]),columns = ['lg_beta'])
    head = pd.DataFrame(data = encoding_head,columns = ['features'])
    imp_LG = pd.concat([imp, head],axis = 1)


    #    
    params = {} 
    if class_weight_apply == False:     
        if best_hyperparams:   
          params = best_hyperparams['rf']              
    elif class_weight_apply == True:    
        if best_hyperparams:
          params = best_hyperparams['rf']
          params['class_weight'] = {0:0.05,1:0.95}
        else: 
          params['class_weight'] = {0:0.05,1:0.95}
          
    clf2 = cls_constructor('rf', params)      
    print('running RF')
    bst_rf, _, _, aucs_rf = ml_model(clf2, X_train_c, y_train_c, set_bootstrap)
    
    # 
    imp = pd.DataFrame(data = abs(bst_rf.feature_importances_),columns = ['rf_importance'])
    imp_RF = pd.concat([imp, head],axis = 1)
    
    
    #
    print('running XGB')  
    if class_weight_apply == False:     
        if best_hyperparams:   
          bst_xgb, bst_xgb2 = model_xgb(xgb, X_train_c, y_train_c, best_hyperparams['xgb'])      
        else:
          bst_xgb, bst_xgb2 = model_xgb(xgb, X_train_c, y_train_c, best_hyperparams)
    elif class_weight_apply == True:    
        if best_hyperparams:
          best_hyperparams['xgb']['scale_pos_weight'] = 16 
          bst_xgb, bst_xgb2 = model_xgb(xgb, X_train_c, y_train_c, best_hyperparams['xgb'])  
        else:
          best_hyperparams['scale_pos_weight'] = 16   
          bst_xgb, bst_xgb2 = model_xgb(xgb, X_train_c, y_train_c, best_hyperparams)
      
    _, _, _, aucs_xgb = ml_model(bst_xgb2, X_train_c, y_train_c, set_bootstrap)

    # different type of feature imp for xgb 
    dictA = bst_xgb.get_score(importance_type = 'weight')
    dictB = bst_xgb.get_score(importance_type = 'gain')
    dictC = bst_xgb.get_score(importance_type = 'cover')
    dictD = bst_xgb.get_score(importance_type = 'total_gain')
    dictE = bst_xgb.get_score(importance_type = 'total_cover')
    impXGB = [dictA, dictB, dictC, dictD, dictE]
    
    
    #     
    clf4 = LinearSVC(random_state=0, tol=1e-5, dual=False, max_iter = 10000) 
    print('running SVC')
    bst_svm, _, _, aucs_svm = ml_model(clf4, X_train_c, y_train_c, set_bootstrap)


    tune_voting_weight = False
    
    if tune_voting_weight == False: 
       eclf1 = VotingClassifier(estimators=[('lg', bst_lg), ('rf', bst_rf), ('xgb', bst_xgb2)], voting='soft', weights = [1,2,1]) # 2.5, 5, 5
       bst_eclf, models, kidx, aucs_eclf = ml_model(eclf1, X_train_c, y_train_c, set_bootstrap)
    else:
       
       v_weights = [ [1,1,1], [1,2,1], [1,1,2], [1,2,2], [2,1,1], [2,1,2], [2,2,1], [1,1,1], [1,6,1], [1,1,6], [1,6,6], [6,1,1], [6,1,6], [6,6,1], [1,1,1], [1,10,1], [1,1,10], [1,10,10], [10,1,1], [10,1,10], [10,10,1], [1,1,1], [1,14,1], [1,1,14], [1,14,14], [14,1,1], [14,1,14], [14,14,1],[1,1,1], [1,18,1], [1,1,18], [1,18,18], [18,1,1], [18,1,18], [18,18,1] ]
                                                                 
                     # [1,1,1], [1,14,1], [1,1,14], [1,14,14], [14,1,1], [14,1,14], [14,14,1], \
                     # [1,1,1], [1,18,1], [1,1,18], [1,18,18], [18,1,1], [18,1,18], [18,18,1] ]

       best_weight_comb = []
       for i in v_weights:
           eclf1 = VotingClassifier(estimators=[('lg', bst_lg), ('rf', bst_rf), ('xgb', bst_xgb2)], voting='soft', weights = i)
           bst_eclf, models, kidx, aucs_eclf = ml_model(eclf1, X_train_c, y_train_c, set_bootstrap)
           best_weight_comb.append(aucs_eclf[0])
           print(i)
           print(aucs_eclf[0])
             
       bst_weight_idx = np.argmax(best_weight_comb)    
        
       print('best_voting weight')
       print(v_weights[bst_weight_idx])
        
        # apply best voting weight
       eclf1 = VotingClassifier(estimators=[('lg', bst_lg), ('rf', bst_rf), ('xgb', bst_xgb2)], voting='soft', weights = v_weights[bst_weight_idx])
       bst_eclf, models, kidx, aucs_eclf = ml_model(eclf1, X_train_c, y_train_c, set_bootstrap)
    
    cm_lg, cp_lg, others_lg = model_result(y_test, bst_lg, 'LG', preprocessed_X_test)
    cm_rf, cp_rf, others_rf  = model_result(y_test, bst_rf, 'RF', preprocessed_X_test)
    cm_xg, cp_xg, others_xg  = model_result(y_test, bst_xgb, 'XGB', preprocessed_X_test)
    cm_sv, cp_sv, others_sv  = model_result(y_test, bst_svm, 'SVM', preprocessed_X_test)
    cm_ec, cp_ec, others_ec  = model_result(y_test, bst_eclf, 'ECLF', preprocessed_X_test)

    models_ = {}
    lg_auc, lgprc, lg_yscore, fpr, tpr = model_auc(bst_lg, preprocessed_X_test, y_test)
    models_['lg'] = (fpr, tpr, lg_auc, aucs_lg, lg_yscore)
    rf_auc, rfprc, rf_yscore, fpr, tpr = model_auc(bst_rf, preprocessed_X_test, y_test)
    models_['rf'] = (fpr, tpr, rf_auc, aucs_rf, rf_yscore)
    svm_auc, svmprc, svm_yscore, fpr, tpr = model_auc(bst_svm, preprocessed_X_test, y_test)
    models_['svm'] = (fpr, tpr, svm_auc, aucs_svm, svm_yscore)
    ec_auc, ecprc, ec_yscore, fpr, tpr = model_auc(bst_eclf, preprocessed_X_test, y_test)
    print(ec_auc)
    models_['ec'] = (fpr, tpr, ec_auc, aucs_eclf, ec_yscore)
      # xgb test part 跟別人不同分開寫
    dtest = xgb.DMatrix(preprocessed_X_test , label = y_test)
    xgb_yscore = bst_xgb.predict(dtest)
    fpr, tpr, _ = roc_curve(y_test, xgb_yscore)
    xgb_auc = auc(fpr, tpr)
    models_['xgb'] = (fpr, tpr,xgb_auc, aucs_xgb, xgb_yscore)
    
    imps = {}
    imps['lg'] = imp_LG
    imps['rf'] = imp_RF
    imps['xgb'] = impXGB    
    models_['feature_imp'] = imps
     
    # if not model_strat:
    print(model_strat)
    print('LG')
    tb1 = table_r(cp_lg,cm_lg,lg_auc,others_lg)
    print('RF')
    tb2 = table_r(cp_rf,cm_rf,rf_auc,others_rf)
    print('XGB')
    tb3 = table_r(cp_xg,cm_xg,xgb_auc,others_xg)
    print('SVM')
    tb4 = table_r(cp_sv,cm_sv,svm_auc,others_sv)
    print('EC')
    tb5 = table_r(cp_ec,cm_ec,ec_auc,others_ec)
     
    filename = 'M'+model_strat+'result_seed'+str(seed_tag)+'.txt'
    ftb = 'LG' + '\n' + tb1 + '\n' +'RF' + '\n' + tb2 + '\n' +'XGB' + '\n' + tb3 + '\n' +'SVM' + '\n' + tb4 + '\n' +'EC' + '\n' + tb5 + '\n'
                
    ftb2 = ftb+'\n'+'train_size:'+str(X_train_c.shape[0])+'\n'+'test_size:'+str(preprocessed_X_test.shape[0])
    
    with open(data_root_folder+filename, 'w') as f:
          f.write(ftb2)    

    return models_, bst_eclf, bst_xgb2, bst_rf, bst_lg 
        

   ############################### 
    

def rand_selection(xdata, ydata):
    
     y0 = ydata[ydata==0]
     y1 = ydata[ydata==1]
     y0_sub = y0[0:len(y1)] 
     rey = np.concatenate((y0_sub,y1))
     
     x0 = xdata[ydata==0,:]
     x1 = xdata[ydata==1,:]
     x0_sub = x0[0:len(y1),:]     
     reX = np.concatenate((x0_sub,x1))
     
     return reX, rey
          
def deal_miss_nan(X, y):
      
    Xcopy = X.copy()
    ycopy = y.copy()

    # 極端值處理 連續類別之缺失
    cont_cols = ['ER_LOS','age1','TMP','PULSE','BPS','BPB','Dr_VSy','WEIGHT','Bun_value', \
                 'CRP_value','Lactate_value','Procalcitonin_value','Creatine_value','Hb_value', \
                 'Hct_value','RBC_value','WBC_value','BRTCNT','SPAO2','DD_visit_30','DD_visit_365', 'exam_TOTAL', \
                 'lab_TOTAL','ER_visit_30','ER_visit_365','sugar_value','Xrayh_T','MRIh_T','CTh_T', \
                 'in_SPAO2', 'in_BRTCNT','in_BPS','in_BPB','in_TMP','in_PULSE','SBP1','DBP1','SBP2','DBP2']
        
    # 移除明顯極端值
    #examining
    #df_cat['PULSE'].value_counts().sort_index()
    Xcopy = remove_extreme(Xcopy, 'TMP', [20, 45])
    Xcopy = remove_extreme(Xcopy, 'PULSE', [10, 250]) 
    Xcopy = remove_extreme(Xcopy, 'BPS', 0) 
    Xcopy = remove_extreme(Xcopy, 'BPB', 0) 
    Xcopy = remove_extreme(Xcopy, 'WEIGHT', [10, 400]) 
    Xcopy = remove_extreme(Xcopy, 'SBP1', [10, 400]) 
    Xcopy = remove_extreme(Xcopy, 'DBP1', [10, 400])
    Xcopy = remove_extreme(Xcopy, 'BRTCNT', 0) 
    Xcopy = remove_extreme(Xcopy, 'SPAO2', 10) 
    
    Xcopy = remove_extreme(Xcopy, 'in_TMP', [20, 45])
    Xcopy = remove_extreme(Xcopy, 'in_PULSE', [10, 250]) 
    Xcopy = remove_extreme(Xcopy, 'in_BPS', 0) 
    Xcopy = remove_extreme(Xcopy, 'in_BPB', 0) 
    Xcopy = remove_extreme(Xcopy, 'SBP2', [10, 400]) 
    Xcopy = remove_extreme(Xcopy, 'DBP2', [10, 400])
    Xcopy = remove_extreme(Xcopy, 'in_BRTCNT', 0) 
    Xcopy = remove_extreme(Xcopy, 'in_SPAO2', 10)

    fs = ['TMP','PULSE','BPS','BPB','WEIGHT','SBP1','DBP1','BRTCNT','SPAO2', \
          'in_TMP','in_PULSE','in_BPS','in_BPB','SBP2','DBP2','in_BRTCNT','in_SPAO2'] # 檢驗值未考慮移除極端值
        
    for f in fs:
        qs = Xcopy[f].quantile([0.25,0.75])
        itq = qs[0.75]-qs[0.25]
        
        Bound = [qs[0.25]-1.5*itq, qs[0.75]+1.5*itq]
        
        idx = Xcopy[ (Xcopy[f] < Bound[0]) | (Xcopy[f] > Bound[1] )].index  
        # Any observations that are more than 1.5 IQR below Q1 or more than 1.5 IQR above Q3 are considered outliers.         
        Xcopy.at[np.array(idx), f] = np.nan
    
    pr = get_nan_pr(Xcopy, cont_cols)
    print(pr)
    fs_to_imp = [i[0] for i in pr if i[1]>0] # preprocessing的時候 impute       
    # 連續類別確認為數字    
    for i in cont_cols:
        Xcopy = assert_number(Xcopy, i)

    # 不連續類別之缺失
    cats = ['DPT2','SEX','ANISICCLSF_C','INTY','week','weekday','indate_time_gr','GCSE','GCSV','GCSM', \
            'indate_month','ANISICMIGD','ANISICMIGD_1','ANISICMIGD_3','ct','MRI','xray','EKG','Echo', \
            'in_GCSE','in_GCSV','in_GCSM','in_ANISICCLSF_C','in_ANISICMIGD','in_ANISICMIGD_1','in_ANISICMIGD_3', \
            'ICD3'] 
    for i in cats:
        print(sum(Xcopy[i].isna()))     
    
    # 缺失新增類別
    Xcopy['INTY'].fillna(value=10, inplace=True)
    Xcopy['ICD3'].fillna(value='noICD', inplace=True)
    
    # 對類別變項檢查, 如果只有<5 sample移除, 無法平均的分給train and test    
    cat_cols = ['INTY', 'ICD3']   
    # cat_cols = ['INTY'] 
    row_idx = np.empty(0).astype(int)    
    for i in cat_cols:
        table = Xcopy[i].value_counts()
        for j,k in table.items():
            if k <= 5:
               row_idx = np.concatenate((row_idx,Xcopy[Xcopy[i].values == j].index.values),axis = 0)
    Xcopy = Xcopy.drop(row_idx)       
    ycopy = ycopy.drop(row_idx) 
   
    return Xcopy, ycopy, fs_to_imp
    
def remove_extreme(x, f, t):
 
    try:
        idx = x[(x[f]<t[0]) | (x[f]>t[1])].index    
    except:
        if t!=0:
           idx = x[(x[f] < t)].index    
        else:    
           idx = x[(x[f] == t)].index    
    
    x.at[np.array(idx), f] = np.nan
    
    return x
    
# ####### where the code start 

if __name__ == '__main__':

    data_root_folder = '/home/anpo/Desktop/pyscript/EDr_72/'
    #data_root_folder = '/Users/chengchichu/Desktop/py/EDr_72/'
    df = pd.read_csv(data_root_folder+'CGRDER_20210618_v15.csv', encoding = 'big5')
    #df2 = pd.read_csv('/home/anpo/Desktop/pyscript/EDr_72/er72_processed_DATA_v10_ccs_converted.csv')

    cols = {}
    cols['DPT2'] = 0
    # cols['drID'] = 2
    cols['SEX'] = 0 
    cols['ANISICCLSF_C'] = 1
    cols['INTY'] = 0
    cols['week'] = 0
    cols['weekday'] = 2
    cols['indate_time_gr'] = 0
    cols['GCSE'] = 1
    cols['GCSV'] = 1
    cols['GCSM'] = 1
    cols['BRTCNT'] = 1  #極端值處理
    cols['SPAO2'] = 1 #極端值處理
    cols['DD_visit_30'] = 1
    cols['ct'] = 2
    cols['MRI'] = 2
    cols['xray'] = 2
    cols['EKG'] = 2
    cols['Echo'] = 2
    cols['DD_visit_365'] = 1
    cols['indate_month'] = 0
    cols['exam_TOTAL'] = 1
    cols['lab_TOTAL'] = 1
    cols['ANISICMIGD'] = 1
    cols['ANISICMIGD_1'] = 1
    cols['ANISICMIGD_3'] = 1
    
    cols['Free_typing'] = 2
    cols['Panendoscope'] = 2
    cols['Buscopan'] = 2
    cols['Ketorolac'] = 2
    cols['Primperan'] = 2
    cols['Novamin'] = 2
    cols['Codeine'] = 2
    cols['Morphine'] = 2
    cols['Nalbuphine'] = 2
    cols['ICD3'] = 0
    
    cols['ER_LOS'] = 1
    cols['age1'] = 1
    cols['ER_visit_30'] = 1 # 
    cols['ER_visit_365'] = 1
    cols['TMP'] = 1
    cols['PULSE'] = 1
    cols['BPS'] = 1
    cols['BPB'] = 1
    cols['Dr_VSy'] = 1
    cols['WEIGHT'] = 1
    cols['Bun_value'] = 1
    cols['CRP_value'] = 1
    cols['Lactate_value'] = 1
    cols['Procalcitonin_value'] = 1    
    cols['Creatine_value'] = 1
    cols['Hb_value'] = 1
    cols['Hct_value'] = 1
    cols['RBC_value'] = 1
    cols['WBC_value'] = 1
    
    cols['sugar_value'] = 1
    cols['Xrayh_T'] = 1
    cols['MRIh_T'] = 1
    cols['CTh_T'] = 1
    cols['in_SPAO2'] = 1
    cols['in_BRTCNT'] = 1
    cols['in_GCSE'] = 1
    cols['in_GCSV'] = 1
    cols['in_GCSM'] = 1
    cols['in_BPS'] = 1
    cols['in_BPB'] = 1
    cols['in_TMP'] = 1
    cols['in_PULSE'] = 1
    cols['in_ANISICCLSF_C'] = 1
    cols['in_ANISICMIGD'] = 1
    cols['in_ANISICMIGD_1'] = 1
    cols['in_ANISICMIGD_3'] = 1
    cols['SBP1'] = 1
    cols['DBP1'] = 1
    cols['SBP2'] = 1
    cols['DBP2'] = 1
     
#    cols['blood_lab'] # 血液檢查, 是否有其他項
#    cols['urine_lab'] # 尿液檢查, 是否有其他項
    #cols['WBC_rslt'] = 1 檢驗數值之類別化1,0,-1(缺失)
    #cols['細分類'] = 0
    cols['中分類'] = 0
    #cols['大分類'] = 2
    #cols['判別依據'] = 2

    # # make sure you get ccs right in CCS_distribution.py
    # index admission的主診斷
    # with open(data_root_folder+'ccs_distri.txt', 'r') as f:
    #       ccs_ids = f.read().splitlines()       
    #       for i in range(len(ccs_ids)):
    #           cols[ccs_ids[i]] = 2
       
    # # 過去兩年病史
    with open(data_root_folder+'ccsh_distri.txt', 'r') as f:
         ccsh_ids = f.read().splitlines()       
         for i in range(len(ccsh_ids)):
             cols[ccsh_ids[i]] = 2    
        
    # # 用藥
    with open(data_root_folder+'atc_distri.txt', 'r') as f:
         atc_ids = f.read().splitlines()       
         for i in range(len(atc_ids)):
             cols[atc_ids[i]] = 2        
                     
    column_keys = cols.keys()
    df_cat = df[column_keys]
    y72 = df['re72'] 
    
    # bad_outcome = df['next_DEAD'] | df['next_ICU']
    # y72 = df['next_adm'] & df['re72']
 
    # 極端值處理, 確認連續類別為數字      
    df_cat, y72, miss_feature = deal_miss_nan(df_cat, y72)
      
    # 同主訴的子群 或 其他子群 >65, los>mean, 
#    complaint = '細分類'
#    # y72 = (y72.astype(bool)) & (df['細分類'].values == df['下次細分類'].values)  
#    y72 = (y72.astype(bool)) & (df[complaint].values != df['下次'+complaint].values)
#    df_cat = df_cat[~df[complaint].isna()]
#    y72 = y72[~df[complaint].isna()]        
    # next ICU and next DEAD也可以用來stratification
    # strat_params['DPT2_1'] = 1
    # strat_params['DPT2_3'] = 3
           
    # 利用主訴來切子模型
    sub_model = True
    keys_to_remove = [] #
    strat_params = {}
    if sub_model:
       key_to_strat = '中分類'
       keys_to_remove.append(key_to_strat)
       for ii in range(10):
           dfcnts = df[key_to_strat].value_counts()
           strat_params[key_to_strat+str(ii+1)] = dfcnts.index[ii]
       #確定用來分子模的field存在    
       if key_to_strat not in df_cat.columns:
          raise NameError('strat field not exist')   
           #刪掉用來分類的類別
       cols.pop(key_to_strat)    
    else:    
       strat_params['全'] = ''
    
    strat_params = {}
    strat_params['中分類1'] = '腹痛'
   
    # 子模迴圈
    # total_cols = {} 
    for key, val in strat_params.items():
        print(val)
        # total_cols[key] = cols.keys()
        if sub_model:
           # 以主訴再分     
           df_3 = df_cat[df_cat[key[:len(key_to_strat)]] == val]
           y72_3 = y72[df_cat[key[:len(key_to_strat)]] == val]
        else:    
           # 不以主訴再分    
           df_3 = df_cat 
           y72_3 = y72
       
        # 切出submodel之後, 判別依據移除
        df_3 = df_3.drop(keys_to_remove,axis=1)
           
        #=== 切分 train and test set
        X_train, X_test, y_train, y_test = train_test_split(df_3, y72_3, test_size=0.2, random_state=40, stratify = y72_3)
        # l = list(range(41,1000))
        # seed_tag = random.sample(l, 1)[0]
        # X_train, X_test, y_train, y_test = train_test_split(df_3, y72_3, test_size=0.2, random_state=seed_tag, stratify = y72_3)
    
        #了解那些是缺失的 
        pr = get_nan_pr(X_train,cols)
        pr2 = get_nan_pr(X_test,cols)
        col_to_drop = [i[0] for i,j in zip(pr,pr2) if i[1]>0.5 or j[1]>0.5] # 少於50%的就丟吧
        # 移除缺失太多的feature, cols也跟著移掉
        X_train_ = X_train.drop(col_to_drop,axis = 1)
        X_test_ = X_test.drop(col_to_drop,axis = 1)
        cols_copy = cols.copy()
        miss_feature_copy = miss_feature.copy()
        for i in col_to_drop:
            cols_copy.pop(i)
            miss_feature_copy.remove(i)
        
        assert(X_train_.shape[1] == len(cols_copy))
        
        # remove highly correlated feature 
        # X_train_ = X_train_.drop(['ANISICCLSF_C','ANISICMIGD','ANISICMIGD_1','ANISICMIGD_3','BRTCNT','SPAO2','TMP','PULSE','BPS','BPB','SBP2','DBP2'], axis=1)
        # X_test_ = X_test_.drop(['ANISICCLSF_C','ANISICMIGD','ANISICMIGD_1','ANISICMIGD_3','BRTCNT','SPAO2','TMP','PULSE','BPS','BPB','SBP2','DBP2'], axis=1)
        # miss_feature_copy = ['ER_LOS','Dr_VSy','WEIGHT','in_SPAO2','in_BRTCNT','in_BPS','in_BPB','in_TMP','in_PULSE','SBP1','DBP1']
        # col_to_drop = ['ANISICCLSF_C','ANISICMIGD','ANISICMIGD_1','ANISICMIGD_3','BRTCNT','SPAO2','TMP','PULSE','BPS','BPB','SBP2','DBP2']
        # for i in col_to_drop:
        #     cols_copy.pop(i)

        # 前處理 = 缺失處理>轉換
        mice_imp_data = pd.read_csv('X_train_mice.csv')
        X_train_ = mice_imp_data
        preprocessed_X, ytrain, preprocessed_X_test, encoding_head = preprocess(X_train_, y_train, X_test_, cols_copy, miss_feature_copy) 
                
        #======imbalanced 處理？
        balanced = True    
        if balanced:
            
           if not sub_model:
              n_seeds_num = 4000 # original 4000, 500 for 2nd return
           else:                
              n_seeds_num = 50  # original 50
                              
           reX = np.array([0])
           while reX.shape[0] < preprocessed_X_test.shape[0]:            
                 undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=n_seeds_num, random_state = 432)
                        #undersample = CondensedNearestNeighbour(n_neighbors=1, n_seeds_S=n_seeds_num)
                        #undersample = NearMiss(version=1,n_neighbors = 3)                               
                 reX, rey = undersample.fit_resample(preprocessed_X, ytrain.values)         
                 n_seeds_num = n_seeds_num-10
                           
           X_train_c = reX.copy()
           y_train_c = rey.copy()
        
        else:
           X_train_c = preprocessed_X.copy()
           y_train_c = ytrain.values.copy()
           #不用alg去balance, 隨意的挑選
#           reX, rey = rand_selection(preprocessed_X, ytrain.values)
#           X_train_c = reX.copy()
#           y_train_c = rey.copy()
        
        # undersample.sample_indices_
        xtrain_idx = np.array(X_train.index)
        xtest_idx = np.array(X_test.index)

        abp_model_idices = {}
        abp_model_idices['train'] = xtrain_idx
        abp_model_idices['test'] = xtest_idx
        # 就跑模型吧
        # models_out = run_models(X_train_c, y_train_c, preprocessed_X_test, y_test, '腹痛', encoding_head, data_root_folder, best_hyperparams, class_weight_apply)
        # run_models(df_sub_train[top30].values, y_train_c, df_sub_test[top30].values, y_test, key, top30, data_root_folder)

        # save the data
        # whether_save = True
        # if whether_save:
        #     import pickle
            
        #     data = {}
        #     data['Xtrain'] = X_train_c
        #     data['ytrain'] = y_train_c
        #     data['Xtest'] = preprocessed_X_test
        #     data['ytest'] = y_test
        #     data['head'] = encoding_head
            
        #     with open('abdominal_pain_data.pickle', 'wb') as handle:
        #           pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
## plot table 1   
# from tableone import TableOne 
# features = ['age1', 'SEX', 'INTY', 'ER_visit_365', 'ANISICCLSF_C', 'ER_LOS', 'TMP', 'PULSE', 'BPS', 'BPB', 'BRTCNT', 'xray', 'Echo', 'ct']
# groupby = 'DPT2'
# categorical = ['SEX','INTY','ANISICCLSF_C', 'xray', 'Echo', 'ct']  # 'blood_lab'
# # nonnormal = ['bili']
# mytable = TableOne(df_3, columns = features, categorical=categorical, groupby=groupby, pval=True)

# print(mytable.tabulate(tablefmt="github"))
# end of the code

# import shap
# explainer = shap.Explainer(bst_xgb)
# shap_values = explainer(X_train_c)

# # visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])


#    shap_values = shap.TreeExplainer(bst_rf).shap_values(X_train_c)
#  # shap_values = shap.explainers.Linear(bst_lg, X_train_c).shap_values(X_train_c)
# # # shap.summary_plot(shap_values, X_train, plot_type="bar")

# adata = pd.DataFrame(X_train_c,columns = encoding_head)
# shap.summary_plot(shap_values, adata, plot_type="bar")

# # # j = 1

# # explainerModel = shap.TreeExplainer(bst_xgb)
# # shap_values_Model = explainerModel.shap_values(X_train_c)
# # p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], adata.iloc[[j]], matplotlib = True, show = False)
# # plt.savefig('tmp.png')
# # # plt.close()

# shap.summary_plot(shap_values, features=adata, feature_names=adata.columns)
# plt.savefig('shap_imp.png')
# plt.show()



# plt.show()
# NOTE =============================================================================================
 


## save for mat 

#aucs = []
#models = []
#k_idx = [] 
# data_size = np.arange(0,X_train.shape[0])
#for train, test in kfold.split(data_size):
#    it_idx = {}
#    it_idx['train'] = train
#    it_idx['test'] = test
##    k_idx.append(it_idx)  
#erData = {}
#
#erData['X'] = preprocessed_X
#
#erData['y'] = y72_
#
#erData['k_idx'] = k_idx
#savemat('er72data.mat',erData)
############functions#####################



# 隨便抽 match minor class的數量
#def under_sample(y_true, X):
#    
#    assert(y_true.shape[0] == X.shape[0])
#    c = Counter(y_true)
#    print(c)   
#    if c[0] > c[1]:
#       major = 0
#       minor = 1
#    else:   
#       major = 1
#       minor = 0
#       
#    minor_class = X[y_true == minor]
#    major_class = X[y_true == major]
#    seed(1)
#    randvalues = randint(0, major_class.shape[0]-1, minor_class.shape[0])
#    resampled_majorX = X[randvalues,:]
#
#    reX = np.concatenate((resampled_majorX,minor_class),axis = 0)
#    rey = np.squeeze(np.concatenate((np.zeros((minor_class.shape[0],1)),np.ones((minor_class.shape[0],1))),axis = 0))
#   
#    return reX, rey    

#def scale_data(data, icol):
#    scaler = StandardScaler()
#    s_data = scaler.fit_transform(data[:,icol].reshape(-1,1)) 
#    data[:,icol] = np.squeeze(s_data)
#    return data
#
#def cm(model, y_test, X_test):
#    data = {'y_Actual':    y_test,
#        'y_Predicted': model.predict(X_test)
#        }
#    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#    cfm = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
#    sn.heatmap(cfm, annot=True)
#
#def ttestXY(y_label, xdata, cols):
#    x1 = y_label == 1
#    x2 = y_label == 0
#    idx = np.where(x1.values)[0]
#    idx2 = np.where(x2.values)[0]
#    h, p = stats.ttest_ind(xdata[idx,cols],  xdata[idx2,cols])
#    return p
#
#def chi2test(y_label, xdata, cols2):
#    dfx = pd.DataFrame(xdata)
#    chips = []
#    for i in cols2:
#        data_crosstab = pd.crosstab(dfx[i], y_label, margins = False)                                
#        stat, p, dof, expected = chi2_contingency(data_crosstab) 
#        chips.append(p)    
#    return chips


#############
## feature selection, 用所有的data
#sel_ = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))
#sel_.fit(preprocessed_X, np.ravel(y72_, order='C'))
#X_train_df = pd.DataFrame(preprocessed_X)
#selected_feat = X_train_df.columns[(sel_.get_support())]
#
#print('total features: {}'.format((X_train_df.shape[1])))
#print('selected features: {}'.format(len(selected_feat)))
#print('features with coefficients shrank to zero: {}'.format(
#np.sum(sel_.estimator_.coef_ == 0)))

# SVM

#import matplotlib
#xgb.plot_importance(bst, importance_type='cover')
#fig = matplotlib.pyplot.gcf()
#fig.set_size_inches(15, 10)
#fig.savefig('xgbF_cover.png')

## MODEL的可解釋性

#import shap
#def model_proba(x):
#    return model.predict_proba(x)[:,1]
##
#
#dfx = pd.DataFrame(preprocessed_X,columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
##
#fig,ax = shap.partial_dependence_plot(
#    '8', model_proba, dfx, model_expected_value=True,
#    feature_expected_value=True, show=False, ice=False
#)


# compare with statsmodels
#sm_model = sm.Logit(y72_, sm.add_constant(preprocessed_X)).fit(disp=0)
#print(sm_model.pvalues)
#sm_model.summary()


#def logit_pvalue(model, x):
#    """ Calculate z-scores for scikit-learn LogisticRegression.
#    parameters:
#        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
#        x:     matrix on which the model was fit
#    This function uses asymtptics for maximum likelihood estimates.
#    """
#    if len(x.shape) == 1:
#       x = x.reshape(-1,1) 
#    
#    p = model.predict_proba(x)
#    n = len(p)
#    m = len(model.coef_[0]) + 1
#    coefs = np.concatenate([model.intercept_, model.coef_[0]])
#    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
#    ans = np.zeros((m, m))
#    for i in range(n):
#        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
#    vcov = np.linalg.inv(np.matrix(ans))
#    se = np.sqrt(np.diag(vcov))
#    t =  coefs/se  
#    p = (1 - norm.cdf(abs(t))) * 2
#    return p

# def add_cat(data, lab, new_cat):
#     tmp = (data[lab].isnull()) | (data[lab].isna())
#     miss_idx = np.where(tmp.values)[0]
# #    miss_idx = [i for i in boolidx.index if boolidx[i]]
#     print('共有{}個缺失值'.format(len(miss_idx)))
#     if miss_idx.size>0:
#        for i in range(len(miss_idx)):
#            data.at[miss_idx[i],lab] = new_cat
#     return data

# def add_avg(data, lab):
# #     tmp = (data[lab].isnull()) | (data[lab].isna())
# #     miss_idx = np.where(tmp.values)[0]
# #     print('共有{}個缺失值'.format(len(miss_idx)))
# #     avg = np.median(data[lab][~tmp.values])     
# #     if miss_idx.size>0:
# #        for i in range(len(miss_idx)):
# # #           data[lab].set_value(miss_idx[i], avg) 
# #            data.at[miss_idx[i],lab] = avg
#     imp = SimpleImputer(missing_values=np.nan, strategy='median')
#     imp.fit([[1, 2], [np.nan, 3], [7, 6]])


#     return data, avg    

# def fill_avg(data, lab, avg):
#     tmp = (data[lab].isnull()) | (data[lab].isna())
#     miss_idx = np.where(tmp.values)[0]
#     if miss_idx.size>0:
#        for i in range(len(miss_idx)):
#            data.at[miss_idx[i],lab] = avg
#     return data

# def get_ICD_cat(data, icd_tag):
# #    pmatch = re.compile(r'[A-Za-z]') # 是否英文字母
#     for i in range(len(data)):   
# #    for i in data.iterrows():
#         icd_code = data.loc[i, icd_tag]
# #        print(icd_code)
#         matchobj = re.finditer(r'[A-Za-z]', icd_code)
# #        print(len(list(matchobj)))
#         if (len(list(matchobj)) < 3) : # NN 為缺失值, len = 2
#            data.at[i, icd_tag] = icd_code[0]
# #           for j in re.finditer(r'[A-Za-z]', icd_code):
# ##               print(icd_code[j.start()])
# ##               data.loc[i, icd_tag] = icd_code[j.start()]
# #               data.at[i, icd_tag] = icd_code[j.start()]
# #    print(m.start(), m.group())   
#     return data


