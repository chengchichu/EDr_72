#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:55:50 2020

@author: chengchichu
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
#from scipy.stats import norm
#import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, classification_report, average_precision_score, confusion_matrix
from numpy.random import randint, seed
from collections import Counter
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
#import xgboost as xgb
from xgboost.sklearn import XGBClassifier
#from scipy import stats
#from scipy.stats import chi2_contingency
from imblearn.under_sampling import OneSidedSelection
import re
from sklearn.impute import SimpleImputer
#from imblearn.under_sampling import CondensedNearestNeighbour
#from imblearn.under_sampling import NearMiss
fconvert = np.vectorize(float)
from tabulate import tabulate

### Functions
def pre_encode(data,tag,mds):
    data_copy = data.copy()  # prevent mutable      
    scale_param = []
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
       encoder = OneHotEncoder(sparse=False)
       out = encoder.fit_transform(data_copy)
    elif tag == 2:  
       out = data_copy
    else:
       print('wrong code')        
    return out, scale_param

def test_encode(data,tag,scale_param,mds):
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
       encoder = OneHotEncoder(sparse=False)
       out = encoder.fit_transform(data_copy)
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

# 改寫general
def add_cut(data, lab, t):
    tmp = data[lab]
    if len(t) == 1:
        boolidx = tmp>=t[0]
        data[lab] = boolidx.eq(True).mul(1)
    elif len(t) == 2:
        boolidx1 = tmp<=t[0] 
        boolidx2 =(tmp>t[0]) & (tmp<=t[1]) 
        boolidx3 = tmp>t[1]      
        out = boolidx1.eq(True).mul(0)
        out2 = boolidx2.eq(True).mul(1)
        out3 = boolidx3.eq(True).mul(2)      
        data[lab] = out | out2 | out3
    elif len(t) == 3: 
        boolidx1 = tmp<=t[0] 
        boolidx2 =(tmp>t[0]) & (tmp<=t[1]) 
        boolidx3 =(tmp>t[1]) & (tmp<=t[2])  
        boolidx4 = tmp>t[2]
        out = boolidx1.eq(True).mul(0)
        out2 = boolidx2.eq(True).mul(1)
        out3 = boolidx3.eq(True).mul(2) 
        out4 = boolidx4.eq(True).mul(3) 
        data[lab] = out | out2 | out3 | out4
    else:
        print('converting error, out of n cat')
    return data  

def model_auc(model, preprocessed_X, y_true):
    try: 
       y_score = model.decision_function(preprocessed_X)   
    except:
       y_score = model.predict_proba(preprocessed_X)[:,1]  
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(y_true, y_score)
    return roc_auc, auprc

def ml_model(clf,data_X,data_y):
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
        xtrain, ytrain = bootstrap(xtrain, ytrain)
        xtest = data_X[data_size[test],:]
        ytest = data_y[data_size[test]]
        model = clf.fit(xtrain, ytrain)
        area_under_ROC = model_auc(model, xtest, ytest)
        aucs.append(area_under_ROC[0])
        models.append(model)        
    # selection model with best AUC
    bst = models[np.argmax(aucs)]
    return bst, models, k_idx, aucs

# def model_xgb(clf,data_X,data_y):
#     # 10 fold
#     kfold = KFold(10, True, 1)
#     aucs = []
#     models = []
#     k_idx = []
#     data_size = np.arange(0,data_X.shape[0])
#     for train, test in kfold.split(data_size):
#         it_idx = {}
#         it_idx['train'] = train
#         it_idx['test'] = test
#         k_idx.append(it_idx)
#         # 只對xtrain做bootstrapping
#         xtrain = data_X[data_size[train],:]
#         ytrain = data_y[data_size[train]]        
#         xtrain, ytrain = bootstrap(xtrain, ytrain)
#         xtest = data_X[data_size[test],:]
#         ytest = data_y[data_size[test]]        
#         # xgb evaluation set
#         eval_set = [(xtrain,ytrain),(xtest,ytest)]
#         # xgb early stopping
#         model = clf.fit(xtrain, ytrain, early_stopping_rounds=5, eval_metric = "error", eval_set = eval_set)
#         results = model.evals_result()    
#         area_under_ROC = model_auc(model, xtest, ytest)
#         aucs.append(area_under_ROC[0])
#         models.append(model)        
#     # selection model with best AUC
#     bst = models[np.argmax(aucs)]
#     return bst, models, k_idx, aucs, eval_set

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
    pr = []
    for i in cols.keys():        
        tmp = (data[i].isnull()) | (data[i].isna())
        pr.append(sum(tmp)/len(df))  
    return pr  

def model_result(y_test, bst, modelname, X_test):
    # print('conf matrix '+modelname)
    cm = confusion_matrix(y_test, bst.predict(X_test))
    # print(cm)
    cp = classification_report(y_test, bst.predict(X_test), target_names=['沒回診','有回診'])
    # print(cp)
    return cm, cp

def table_r(cp,cm,auc):
    lines = cp.splitlines()
    c=[np.array(lines[2].split()[1:4]), np.array(lines[3].split()[1:4])]
    out = np.concatenate((cm,c),axis=1)
    cc = np.array([auc, np.nan])
    out = np.concatenate((out,cc.reshape(2,-1)),axis=1)
    print(tabulate(out, headers=['p0', 'p1','precision','recall','f1','AUC']))
    
def preprocess(X_train, y_train, X_test, y_test, cols):  
    
    X = X_train.copy()      
    y72 = y_train.copy()
     
    # 新增類別不太適合, 缺失太少, train test split 類別不平均 
    X['INTY'].fillna(value=6, inplace=True)
    X_test['INTY'].fillna(value=6, inplace=True)
    
    # 連續類別確認為數字
    fs_to_imp = []       
    for i,j in cols.items():
        if (j == 1) and (i in miss_feature):
           X = assert_number(X, i)
           X_test = assert_number(X_test, i)
           fs_to_imp.append(i)   
           
    # imputation
    md_strategy = True
    if md_strategy:    
       imp = SimpleImputer(missing_values=np.nan, strategy='median')   
    else: 
       imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = -1)  
    imp.fit(X[fs_to_imp]) 
    impdata = imp.transform(X[fs_to_imp])
    impdata_test = imp.transform(X_test[fs_to_imp])          
    cnt = 0
    for i in fs_to_imp:
        print(i)
        X[i] = impdata[:,cnt]
        X_test[i] = impdata_test[:,cnt]
        cnt+=1
            
    # preprocessing encoding
    preprocessed_X = []
    preprocessed_X_test = []
    encoding_head = []
    scale_params = {}
    cnt = 0 # initial
    mcnt0 = []
    for key, value in cols.items():

        #print(key)

        data_col = X[key].values.reshape(-1,1)
        out, scale_param = pre_encode(data_col, value, md_strategy)
        
        data_col_test = X_test[key].values.reshape(-1,1)
        out_test = test_encode(data_col_test, value, scale_param, md_strategy)
  
        # n-1 for dummy variable, this means reference group is the first column
        if out.shape[1]>1:
           out = out[:,1:(out.shape[1])]
           out_test = out_test[:,1:(out_test.shape[1])]
           
        ec = [key for i in range(out.shape[1])]
        encoding_head.append(ec)
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
           
    encoding_head_flat = [j for i in encoding_head for j in i]      
    
    return preprocessed_X, y72, preprocessed_X_test, encoding_head_flat
    
def run_models(X_train_c, y_train_c, preprocessed_X_test, y_test, model_strat):
        ## 跑model      
    clf1 = LogisticRegression(random_state=0, max_iter=3000)
    bst_lg, models, kidx, aucs_lg = ml_model(clf1, X_train_c, y_train_c)
    
    clf2 = RandomForestClassifier(random_state=0)  ## 隨機森林
    bst_rf, models, kidx, aucs_rf = ml_model(clf2, X_train_c, y_train_c)
    
     # # clf3 = XGBClassifier(use_label_encoder=False, eval_metric="error")    
     # # bst_xgb, models, kidx, aucs_xgb, eval_set = model_xgb(clf3, X_train_c, y_train_c)
    
    clf3 = XGBClassifier(use_label_encoder=False, eval_metric="error")    
    bst_xgb, models, kidx, aucs_xgb = ml_model(clf3, X_train_c, y_train_c)

    eclf1 = VotingClassifier(estimators=[('lg', clf1), ('rf', clf2), ('xgb', clf3)], voting='soft', weights = [2.5,5,2.5])
    bst_eclf, models, kidx, aucs_eclf = ml_model(eclf1, X_train_c, y_train_c)
    
    cm_lg, cp_lg = model_result(y_test, bst_lg, 'LG', preprocessed_X_test)
    cm_rf, cp_rf = model_result(y_test, bst_rf, 'RF', preprocessed_X_test)
    cm_xg, cp_xg = model_result(y_test, bst_xgb, 'XGB', preprocessed_X_test)
    cm_ec, cp_ec = model_result(y_test, bst_eclf, 'ECLF', preprocessed_X_test)

    metrics.plot_roc_curve(bst_lg, preprocessed_X_test, y_test)
    metrics.plot_roc_curve(bst_rf, preprocessed_X_test, y_test) 
    metrics.plot_roc_curve(bst_xgb, preprocessed_X_test, y_test) 
    metrics.plot_roc_curve(bst_eclf, preprocessed_X_test, y_test) 
   
    lg_auc, lgprc = model_auc(bst_lg, preprocessed_X_test, y_test)
    rf_auc, rfprc = model_auc(bst_rf, preprocessed_X_test, y_test)
    xgb_auc, xgbprc = model_auc(bst_xgb, preprocessed_X_test, y_test)
    ec_auc, ecprc = model_auc(bst_eclf, preprocessed_X_test, y_test)
    
    #if not model_strat:
    print(model_strat)
    print('LG')
    table_r(cp_lg,cm_lg,lg_auc)
    print('RF')
    table_r(cp_rf,cm_rf,rf_auc)
    print('XGB')
    table_r(cp_xg,cm_xg,xgb_auc)
    print('EC')
    table_r(cp_ec,cm_ec,ec_auc)
    
    # # finding the best weight for voting classifier
    # weights_comb = [[3,3.5,3.5],[5,2.5,2.5],[7,1.5,1.5],[9,0.5,0.5]]
    # weights_comb = [[3.5,3,3.5],[2.5,5,2.5],[1.5,7,1.5],[0.5,9,0.5]]
    # weights_comb = [[3.5,3.5,3],[2.5,2.5,5],[1.5,1.5,7],[0.5,0.5,9]]
    # for i in weights_comb:        
    #     eclf1 = VotingClassifier(estimators=[('lg', clf1), ('rf', clf2), ('xgb', clf3)], voting='soft', weights = i)
    #     bst_eclf, models, kidx, aucs_eclf = ml_model(eclf1, X_train_c, y_train_c)
    #     cm_ec, cp_ec = model_result(y_test, bst_eclf, 'ec', preprocessed_X_test)
    #     ec_auc, ecprc = model_auc(bst_eclf, preprocessed_X_test, y_test)
    #     print('wlg-%.2f wrf-%.2f wxgb-%.2f'  % (i[0], i[1], i[2]))
    #     table_r(cp_ec,cm_ec,ec_auc)
        # mean_auc = np.mean(aucs_eclf)
        # print(mean_auc)
        # print(confusion_matrix(y_test, bst_eclf.predict(preprocessed_X_test)))
        # fpr, tpr, thresholds = metrics.roc_curve(y_test, bst_eclf.predict_proba(preprocessed_X_test)[:,1])
        # print(metrics.auc(fpr, tpr))

# ####### where the code start 

if __name__ == '__main__':

    data_root_folder = '/Users/chengchichu/Desktop/py/EDr_72/'
    df = pd.read_csv(data_root_folder+'CGRDER_20210422_v11.csv', encoding = 'big5')
    
    #df2 = pd.read_csv('/home/anpo/Desktop/pyscript/EDr_72/er72_processed_DATA_v10_ccs_converted.csv')

    cols = {}
    cols['DPT2'] = 0
    # cols['drID'] = 2
    cols['SEX'] = 0 
    cols['ANISICCLSF_C'] = 2
    cols['INTY'] = 0
    cols['ER_LOS'] = 1
    cols['age1'] = 1
    cols['week'] = 0
    cols['weekday'] = 2
    cols['indate_time_gr'] = 0
    cols['ER_visit_30'] = 1 # 
    cols['ER_visit_365'] = 1
    cols['TMP'] = 1
    cols['PULSE'] = 1
    cols['BPS'] = 1
    cols['BPB'] = 1
    cols['GCSE'] = 2
    cols['GCSV'] = 2
    cols['GCSM'] = 2
    cols['BRTCNT'] = 1
    cols['SPAO2'] = 1
    cols['DD_visit_30'] = 1
    cols['ct'] = 2
    cols['MRI'] = 2
    cols['xray'] = 2
    cols['EKG'] = 2
    cols['Echo'] = 2
    # cols['blood_lab'] = 2 
    # cols['urine_lab'] = 2 
    #cols['WBC_lab'] = 2
    #cols['Hb_Lab'] = 2
    #cols['Hct_lab'] = 2
    #cols['Bun_lab'] = 2
    #cols['Creatine_lab'] = 2
    #cols['CRP_lab'] = 2
    #cols['Procalcitonin_lab'] = 2
    #cols['Lactate_lab'] = 2
    #cols['ICD'] = 0  #cols['ICD3'] = 0
    cols['DD_visit_365'] = 1
    cols['Dr_VSy'] = 1
    cols['WEIGHT'] = 1
    cols['indate_month'] = 0
    cols['SBP'] = 1
    cols['DBP'] = 1
    cols['exam_TOTAL'] = 1
    cols['lab_TOTAL'] = 1
    cols['ANISICMIGD'] = 2
    cols['ANISICMIGD_1'] = 2
    cols['ANISICMIGD_3'] = 2
    cols['Bun_value'] = 1
    cols['CRP_value'] = 1
    cols['Lactate_value'] = 1
    cols['Procalcitonin_value'] = 1    
    cols['Creatine_value'] = 1
    cols['Hb_value'] = 1
    cols['Hct_value'] = 1
    cols['RBC_value'] = 1
    cols['WBC_value'] = 1
    cols['中分類'] = 2
    cols['大分類'] = 2

    # # make sure you get ccs right in CCS_distribution py
    # index admission的主診斷
    with open(data_root_folder+'ccs_distri.txt', 'r') as f:
         ccs_ids = f.read().splitlines()       
         for i in range(len(ccs_ids)):
             cols[ccs_ids[i]] = 2
       
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
    df_cat = df[cols.keys()]
    y72 = df['re72'] 
    
    # 院區切分
    # 台北院區的話
    # df_3 = df_cat[df['DPT2']==1]
    # y72_3 = y72[df['DPT2']==1]
                
    # df_3 = df_cat.iloc[df2['newID'][df2['ccs']=='dx82'].values,:]
    # y72_3 = y72.iloc[df2['newID'][df2['ccs']=='dx82'].values]
        
    # 對類別變項檢查, 如果只有一個sample移除, 無法平均的分給train and test    
    cat_cols = ['SEX','ANISICCLSF_C','INTY','week','weekday','indate_time_gr']   
    row_idx = np.empty(0).astype(int)    
    for i in cat_cols:
        table = df_cat[i].value_counts()
        for j,k in table.items():
            if k <= 5:
               #row_idx.append(df_3[df_3[i].values == j].index.values)
               row_idx = np.concatenate((row_idx,df_cat[df_cat[i].values == j].index.values),axis = 0)
    df_cat = df_cat.drop(row_idx)       
    y72 = y72.drop(row_idx)       
               
    
    # 切分subpopulation to build model
    strat_params = {}
    strat_params['大分類'] = '腸胃系統'
    strat_params['中分類'] = '腹痛'
    sub_model = True
    # 刪掉用來分類的類別
    keys_to_remove = ['大分類', '中分類']
    for key in keys_to_remove:
        cols.pop(key)

    for key, val in strat_params.items():
        print(key)
        df_3 = df_cat[df_cat[key] == val]
        y72_3 = y72[df_cat[key] == val]
        #=== 切分 train and test set
        # 思考在imputation前如何正確stratify 
        #X_train, X_test, y_train, y_test = train_test_split(df_3, y72_3, test_size=0.3, random_state=40, stratify = df_3['INTY'])
        X_train, X_test, y_train, y_test = train_test_split(df_3, y72_3, test_size=0.3, random_state=40)
    
        #了解哪些是缺失的 
        pr = get_nan_pr(X_train,cols)
        miss = [i>0 for i in pr]
        miss_feature = []
        cnt = 0
        for i in column_keys:
            if miss[cnt]:           
               print(i) 
               miss_feature.append(i)
            cnt+=1
        
        #   
        preprocessed_X, ytrain, preprocessed_X_test, encoding_head_flat = preprocess(X_train, y_train, X_test, y_test, cols) 
    
        #======imbalanced 處理？
        unbalanced_corret = True
        if unbalanced_corret:
           n_seeds_num=10000     
           if sub_model:
              n_seeds_num=1000               
           undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=n_seeds_num)
           #undersample = CondensedNearestNeighbour(n_neighbors=1)
           #undersample = NearMiss(version=1,n_neighbors = 3)
           reX, rey = undersample.fit_resample(preprocessed_X, ytrain.values)
           
           X_train_c = reX.copy()
           y_train_c = rey.copy()
        else:
           X_train_c = preprocessed_X.copy()
           y_train_c = ytrain.values.copy()
        
        run_models(X_train_c, y_train_c, preprocessed_X_test, y_test, key)

    # # save data for autoML
    # import pickle

    # ehr_processed = {}
    # ehr_processed['Xtrain'] = X_train_c
    # ehr_processed['ytrain'] = y_train_c
    # ehr_processed['Xtest'] = preprocessed_X_test
    # ehr_processed['ytest'] = y_test

    # with open("ehr_processed.pickle","wb") as f:
    #     pickle.dump( ehr_processed, f)

    # autoML


# after u pull ur images
# (base) anpo@anpo-linux-2f:~$ docker images 
# Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.40/images/json: dial unix /var/run/docker.sock: connect: permission denied
# (base) anpo@anpo-linux-2f:~$ sudo docker images 
# REPOSITORY             TAG                 IMAGE ID            CREATED             SIZE
# mfeurer/auto-sklearn   master              d6bb7d7fbfab        5 days ago          1.23GB
# (base) anpo@anpo-linux-2f:~$ docker run -it mfeurer/auto-sklearn:master
# docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post http://%2Fvar%2Frun%2Fdocker.sock/v1.40/containers/create: dial unix /var/run/docker.sock: connect: permission denied.
# See 'docker run --help'.
# (base) anpo@anpo-linux-2f:~$ sudo docker run -it mfeurer/auto-sklearn:master
# root@0c3438ff51ec:/auto-sklearn# 

# python3

#sudo docker ps 
# 查containerID
#sudo docker cp /home/anpo/Desktop/pyscript/EDr_72/ehr_processed.pickle 97a8233ac77e:/auto-sklearn/autosklearn









    # # 對某些variability 非常小的feature還是作類別化處理 例如體溫
    # X = add_cut(X, 'TMP', [37.5])
    # X = add_cut(X, 'SPAO2', [94])
    # X = add_cut(X, 'BRTCNT', [12, 20])
    # X_test = add_cut(X_test, 'TMP', [37.5])
    # X_test = add_cut(X_test, 'SPAO2', [94])
    # X_test = add_cut(X_test, 'BRTCNT', [12, 20])
               


## if model is LG
# importance = bst.coef_[0]
# assert(len(encoding_head_flat) == len(importance))

# fig, ax = plt.subplots(figsize=(12, 6))
# plt.bar(np.arange(0,len(importance)), importance)
# ax.set_xticks(np.arange(0, len(encoding_head_flat), step=1)) 
# ax.set_xticklabels(encoding_head_flat) 
# plt.setp(ax.get_xticklabels(), rotation=90)

# if XGB
#fig, ax = plt.subplots(figsize=(12, 6))
#sorted_idx = bst.feature_importances_.argsort()
##sorted_idx = np.sort(bst.feature_importances_)
#xh = [encoding_head_flat[u] for u in sorted_idx]
#yh = bst.feature_importances_[sorted_idx]
#plt.bar(np.arange(0,len(yh)),yh)
#ax.set_xticks(np.arange(0, len(encoding_head_flat), step=1)) 
#ax.set_xticklabels(xh) 
#plt.setp(ax.get_xticklabels(), rotation=90)
#plt.xlabel("Xgboost Feature Importance")

#results = model.evals_result()





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

    # 模型performance based on AUC 
#def model_auc(model, preprocessed_X, y_true):
#    try: 
#       y_score = model.decision_function(preprocessed_X)   
#    except:
#       y_score = model.predict_proba(preprocessed_X)[:,1]  
#    fpr, tpr, _ = roc_curve(y_true, y_score)
#    roc_auc = auc(fpr, tpr)
#    auprc = average_precision_score(y_true, y_score)
#    return roc_auc, auprc
#
## 其他模型performance指標
#def model_report(model, preprocessed_X, y_true, class_labels):
#    y_pred = model.predict(preprocessed_X)
#    print(classification_report(y_true, y_pred, target_names=class_labels))

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





#kfold = KFold(10, True, 1)
#aucs = []
#data = np.arange(0,lgx.shape[0])
#for train, test in kfold.split(data):
#    
#    train_xdata = xgbx[data[train],:]
#    train_ydata = xgby[data[train]]
#    # XGB need validation
#    X_train2, X_val, y_train2, y_val = train_test_split(train_xdata, train_ydata, test_size=0.25, random_state=1)
#    
#    model = LogisticRegression(random_state=0, max_iter=1000).fit(lgx[data[train],:], lgy[data[train]])
#    area_under_ROC = model_auc(model, lgx[data[test],:], lgy[data[test]])
#    aucs.append(area_under_ROC[0])


    
#feature correlation

#
#p = ttestXY(y72_, X, ~)
#p = ttestXY(y72_, X, ~)
#
#
#cols2 = [	'LOC', 'SEX',	'DPT',	'ANISICCLSF_C',	'INTY', 'week', 'weekday',	'indate_time_gr', 'ER_visit_30', 'ER_visit_365', 'TMP',	'PULSE', 'BPS',	'GCSE',	'GCSV',	'GCSM', 'BRTCNT','SPAO2']  
#chips = chi2test(y72_, X) 
    
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


## check any missing value, 暫時先排除, 之後想填補策略
#for i in range(len(cols)):
##    print(i)
#    if i == 0:
#       tmp = X[cols[i]].isnull().values 
#    else:
#       tmp = tmp | X[cols[i]].isnull().values 
      
#missing = tmp

# 處理class imbalance的問題                
#[reX, rey] = under_sample(y_train, X_train)              
#clf = RandomForestClassifier(max_depth=2, random_state=0).fit(reX, rey)
#area_under_ROC, auprc = model_auc(clf, X_test, y_test)
#cm(clf, y_test, X_test)
#confusion_matrix(y_test, clf.predict(X_test))
#model_report(clf, reX, rey, ['72小時沒返診', '72小有返診'])

#from imblearn.under_sampling import TomekLinks
#tl = TomekLinks()
#X_res0, y_res0 = tl.fit_resample(X_train, y_train.values)
#
##from imblearn.under_sampling import CondensedNearestNeighbour
##undersample = CondensedNearestNeighbour(n_neighbors=1)
##X_res, y_res = undersample.fit_resample(X_res0, y_res0)


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
