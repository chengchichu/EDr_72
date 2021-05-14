#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:55:50 2020

@author: chengchichu
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, average_precision_score, confusion_matrix
from numpy.random import randint, seed
from collections import Counter
from sklearn import metrics
from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
#import xgboost as xgb
from xgboost.sklearn import XGBClassifier
#from scipy.stats import chi2_contingency
from imblearn.under_sampling import OneSidedSelection, CondensedNearestNeighbour
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
    encoder = []
    #feature_name = []
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
       #feature_name = encoder.get_feature_names()
    elif tag == 2:  
       out = data_copy
    else:
       print('wrong code')       
     
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
        # xtrain, ytrain = bootstrap(xtrain, ytrain)
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
        pr.append([i,sum(tmp)/data.shape[0]])  
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
    
    tb = tabulate(out, headers=['p0', 'p1','precision','recall','f1','AUC'])
    return tb

def preprocess(X_train, y_train, X_test, cols, miss_feature):  
    
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
    scale_params = {}   
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
           
        # L = out.shape[1]
        # print(L)
        # N = -1-L+1
        # print(N)
        # print(atmp[N:][0])
        # print(key)
        # print(encoding_head)
        # if value == 0:
           
        cnt += 1
    #encoding_head_flat = [j for i in encoding_head for j in i]      
    #print(len(encoding_head))    
    
    return preprocessed_X, y72, preprocessed_X_test, encoding_head
    
def run_models(X_train_c, y_train_c, preprocessed_X_test, y_test, model_strat, encoding_head):
        ## 跑model      
    clf1 = LogisticRegression(random_state=0, max_iter=5000)
    print('running LG')
    bst_lg, models, kidx, aucs_lg = ml_model(clf1, X_train_c, y_train_c)
    
    # LG imp
    imp = pd.DataFrame(data = abs(bst_lg.coef_[0]),columns = ['lg_beta'])
    head = pd.DataFrame(data = encoding_head,columns = ['features'])
    imp2 = pd.concat([imp, head],axis = 1)
    print(imp2.sort_values(by=['lg_beta'],ascending = False))
    
    clf2 = RandomForestClassifier(random_state=0)  ## 隨機森林
    print('running RF')
    bst_rf, models, kidx, aucs_rf = ml_model(clf2, X_train_c, y_train_c)
    
    imp = pd.DataFrame(data = abs(bst_rf.feature_importances_),columns = ['rf_importance'])
    imp2 = pd.concat([imp, head],axis = 1)
    print(imp2.sort_values(by=['rf_importance'],ascending = False))
    
     # # clf3 = XGBClassifier(use_label_encoder=False, eval_metric="error")    
     # # bst_xgb, models, kidx, aucs_xgb, eval_set = model_xgb(clf3, X_train_c, y_train_c)
    
    clf3 = XGBClassifier(use_label_encoder=False, eval_metric="error")    
    print('running XGB')
    bst_xgb, models, kidx, aucs_xgb = ml_model(clf3, X_train_c, y_train_c)

    clf4 = LinearSVC(random_state=0, tol=1e-5, dual=False, max_iter = 10000) 
    print('running SVC')
    bst_svm, models, kidx, aucs_svm = ml_model(clf4, X_train_c, y_train_c)

    eclf1 = VotingClassifier(estimators=[('lg', clf1), ('rf', clf2), ('xgb', clf3)], voting='soft', weights = [2.5,5,2.5])
    bst_eclf, models, kidx, aucs_eclf = ml_model(eclf1, X_train_c, y_train_c)
    
    cm_lg, cp_lg = model_result(y_test, bst_lg, 'LG', preprocessed_X_test)
    cm_rf, cp_rf = model_result(y_test, bst_rf, 'RF', preprocessed_X_test)
    cm_xg, cp_xg = model_result(y_test, bst_xgb, 'XGB', preprocessed_X_test)
    cm_sv, cp_sv = model_result(y_test, bst_svm, 'SVM', preprocessed_X_test)
    cm_ec, cp_ec = model_result(y_test, bst_eclf, 'ECLF', preprocessed_X_test)

   # metrics.plot_roc_curve(bst_lg, preprocessed_X_test, y_test)
   # metrics.plot_roc_curve(bst_rf, preprocessed_X_test, y_test) 
   # metrics.plot_roc_curve(bst_xgb, preprocessed_X_test, y_test) 
   # metrics.plot_roc_curve(bst_eclf, preprocessed_X_test, y_test) 
   
    lg_auc, lgprc = model_auc(bst_lg, preprocessed_X_test, y_test)
    rf_auc, rfprc = model_auc(bst_rf, preprocessed_X_test, y_test)
    xgb_auc, xgbprc = model_auc(bst_xgb, preprocessed_X_test, y_test)
    svm_auc, svmprc = model_auc(bst_svm, preprocessed_X_test, y_test)
    ec_auc, ecprc = model_auc(bst_eclf, preprocessed_X_test, y_test)
    
    #if not model_strat:
    print(model_strat)
    print('LG')
    tb1 = table_r(cp_lg,cm_lg,lg_auc)
    print('RF')
    tb2 = table_r(cp_rf,cm_rf,rf_auc)
    print('XGB')
    tb3 = table_r(cp_xg,cm_xg,xgb_auc)
    print('SVM')
    tb4 = table_r(cp_sv,cm_sv,svm_auc)
    print('EC')
    tb5 = table_r(cp_ec,cm_ec,ec_auc)
    
    filename = 'M'+model_strat+'result.txt'
    ftb = 'LG' + '\n' + tb1 + '\n' +'RF' + '\n' + tb2 + '\n' +'XGB' + '\n' + tb3 + '\n' +'SVM' + '\n' + tb4 + '\n' +'EC' + '\n' + tb5 + '\n'
           
    with open('/home/anpo/Desktop/pyscript/EDr_72/'+filename, 'w') as f:
         f.write(ftb+'\n'+'train_size:'+str(X_train_c.shape[0])+'\n'+'test_size:'+str(preprocessed_X_test.shape[0]))
      
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
          
# ####### where the code start 

if __name__ == '__main__':

    data_root_folder = '/home/anpo/Desktop/pyscript/EDr_72/'
    #data_root_folder = '/Users/chengchichu/Desktop/py/EDr_72/'
    df = pd.read_csv(data_root_folder+'CGRDER_20210512_v12.csv', encoding = 'big5')
    
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
    cols['細分類'] = 2
    cols['中分類'] = 2
    cols['大分類'] = 2
    cols['判別依據'] = 2

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
 
    # 同主訴的子群  
    complaint = '細分類'
    # y72 = (y72.astype(bool)) & (df['細分類'].values == df['下次細分類'].values)  
    y72 = (y72.astype(bool)) & (df[complaint].values != df['下次'+complaint].values)
    df_cat = df_cat[~df[complaint].isna()]
    y72 = y72[~df[complaint].isna()]    
           
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
    strat_params['全'] = ''
    sub_model = False
    # strat_params['判別依據1'] = '檢傷判別條件為主訴=>腹痛,急性中樞中度疼痛(4-7)'
    # strat_params['判別依據2'] = '檢傷判別條件為主訴=>眩暈/頭暈,姿勢性，無其他神經學症狀'
    # strat_params['判別依據3'] = '檢傷判別條件為主訴=>胸痛/胸悶,急性中樞中度疼痛(4-7)'
    # strat_params['判別依據4'] = '檢傷判別條件為主訴=>發燒/畏寒,發燒(看起來有病容)'                                           
    # strat_params['判別依據5'] = '檢傷判別條件為主訴=>腰痛,急性中樞中度疼痛(4-7)'
    # strat_params['判別依據6'] = '檢傷判別條件為主訴=>頭痛,急性中樞中度疼痛(4-7)'
    # strat_params['判別依據7'] = '檢傷判別條件為主訴=>噁心/嘔吐,急性持續性嘔吐' #seed 50 above
    # strat_params['判別依據8'] = '檢傷判別條件為主訴=>眼睛疼痛,急性中樞中度疼痛(4-7)'
    # strat_params['判別依據9'] = '檢傷判別條件為主訴=>背痛,急性中樞中度疼痛(4-7)'
    # strat_params['判別依據10'] = '檢傷判別條件為主訴=>腹瀉,輕度脫水'

    # strat_params['細分類1'] = '急性中樞中度疼痛(4-7)'
    # strat_params['細分類2'] = '發燒(看起來有病容)'
    # strat_params['細分類3'] = '急性周邊重度疼痛(8-10)'
    # strat_params['細分類4'] = '姿勢性，無其他神經學症狀'                                           
    # strat_params['細分類5'] = '急性周邊中度疼痛(4-7)'
    # strat_params['細分類6'] = '血壓或心跳有異於病人之平常數值，但血行動力穩定'
    # strat_params['細分類'] = '輕度呼吸窘迫(92-94%)' #seed 50 above
    # strat_params['細分類8'] = '急性持續性嘔吐'       
    # strat_params['細分類9'] = '急性中樞輕度疼痛(＜4)'
    # strat_params['細分類10'] = '輕度脫水'
    
    # strat_params['中分類1'] = '腹痛'
    # strat_params['中分類2'] = '眩暈/頭暈'
    # strat_params['中分類3'] = '胸痛/胸悶'
    # strat_params['中分類4'] = '發燒/畏寒'                                           
    # strat_params['中分類5'] = '噁心/嘔吐'
    # strat_params['中分類6'] = '局部紅腫' 
    # strat_params['中分類7'] = '頭痛' 
    # strat_params['中分類8'] = '腰痛'       
    # strat_params['中分類9'] = '咳嗽'
    # strat_params['中分類10'] = '紅疹'
    
    # strat_params['大分類1'] = '腸胃系統'
    # strat_params['大分類2'] = '神經系統'
    # strat_params['大分類3'] = '心臟血管系統'
    # strat_params['大分類4'] = '一般和其他'                                           
    # strat_params['大分類5'] = '耳鼻喉系統'
    # strat_params['大分類6'] = '泌尿系統' 
    # strat_params['大分類'] = '皮膚系統' 
    # strat_params['大分類8'] = '骨骼系統'       
    # strat_params['大分類9'] = '呼吸系統'
    # strat_params['大分類10'] = '眼科'

    # strat_params['DPT2_1'] = 1
    # strat_params['DPT2_3'] = 3
           
    # 刪掉用來分類的類別
    keys_to_remove = ['判別依據','細分類','中分類','大分類']
    for key in keys_to_remove:
        cols.pop(key)    

    total_cols = {} 
    for key, val in strat_params.items():
        print(val)
        total_cols[key] = cols.keys()
        if sub_model:
           df_3 = df_cat[(df_cat[key[:4]] == val) & (df_cat['age1']<65)]
           y72_3 = y72[(df_cat[key[:4]] == val) & (df_cat['age1']<65)] 
           # df_3 = df_cat[(df_cat[key[:4]] == val) & (df_cat['ER_LOS']<df_cat['ER_LOS'].mean())]
           # y72_3 = y72[(df_cat[key[:4]] == val) & (df_cat['ER_LOS']<df_cat['ER_LOS'].mean())] 
           # df_3 = df_cat[(df_cat[key[:4]] == val) & (df_cat['Dr_VSy']<df_cat['Dr_VSy'].mean())]
           # y72_3 = y72[(df_cat[key[:4]] == val) & (df_cat['Dr_VSy']<df_cat['Dr_VSy'].mean())] 
           
                
           # df_3 = df_cat[df_cat[key[:4]] == val]
           # y72_3 = y72[df_cat[key[:4]] == val]
           # df_3 = df_cat[df_cat[key[:3]] == val]
           # y72_3 = y72[df_cat[key[:3]] == val]
                       
        else:    
           # 老人 
           # df_3 = df_cat[df_cat['age1']>65]   
           # y72_3 = y72[df_cat['age1']>65] 
           # df_3 = df_cat[df_cat['ER_LOS']>df_cat['ER_LOS'].mean()]   
           # y72_3 = y72[df_cat['ER_LOS']>df_cat['ER_LOS'].mean()] 
           df_3 = df_cat 
           y72_3 = y72
       
        # 切出submodel之後, 判別依據移除
        df_3 = df_3.drop(keys_to_remove,axis=1)
           
        #=== 切分 train and test set
        # 思考在imputation前如何正確stratify 
        X_train, X_test, y_train, y_test = train_test_split(df_3, y72_3, test_size=0.2, random_state=40, stratify = y72_3)
        #X_train, X_test, y_train, y_test = train_test_split(df_3, y72_3, test_size=0.3, random_state=40)
    
        #了解哪些是缺失的 
        pr = get_nan_pr(X_train,cols)
        pr2 = get_nan_pr(X_test,cols)
        # col_to_drop = []
        miss_feature = [i[0] for i,j in zip(pr,pr2) if i[1]>0 or j[1]>0]
        #cnt = 0
        col_to_drop = [i[0] for i,j in zip(pr,pr2) if i[1]>0.5 or j[1]>0.5]

        # 移除缺失太多的feature, cols也跟著移掉
        X_train_ = X_train.drop(col_to_drop,axis = 1)
        X_test_ = X_test.drop(col_to_drop,axis = 1)
        cols_copy = cols.copy()
        for i in col_to_drop:
            cols_copy.pop(i)
        
        assert(X_train_.shape[1] == len(cols_copy))
        preprocessed_X, ytrain, preprocessed_X_test, encoding_head = preprocess(X_train_, y_train, X_test_, cols_copy, miss_feature) 
            
    
        #======imbalanced 處理？
    
        n_seeds_num=4000     
        if sub_model:                      
           n_seeds_num = 50
           reX = np.array([0])
           while reX.shape[0] < preprocessed_X_test.shape[0]:            
                 undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=n_seeds_num)
                    #undersample = CondensedNearestNeighbour(n_neighbors=1, n_seeds_S=n_seeds_num)
                    #undersample = NearMiss(version=1,n_neighbors = 3)                               
                 reX, rey = undersample.fit_resample(preprocessed_X, ytrain.values)         
                 n_seeds_num = n_seeds_num-10
                       
           X_train_c = reX.copy()
           y_train_c = rey.copy()
        else:
           # X_train_c = preprocessed_X.copy()
           # y_train_c = ytrain.values.copy()
           
           reX, rey = rand_selection(preprocessed_X, ytrain.values)
           X_train_c = reX.copy()
           y_train_c = rey.copy()
        
        run_models(X_train_c, y_train_c, preprocessed_X_test, y_test, key, encoding_head)

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


# 改寫general
# def add_cut(data, lab, t):
#     tmp = data[lab]
#     if len(t) == 1:
#         boolidx = tmp>=t[0]
#         data[lab] = boolidx.eq(True).mul(1)
#     elif len(t) == 2:
#         boolidx1 = tmp<=t[0] 
#         boolidx2 =(tmp>t[0]) & (tmp<=t[1]) 
#         boolidx3 = tmp>t[1]      
#         out = boolidx1.eq(True).mul(0)
#         out2 = boolidx2.eq(True).mul(1)
#         out3 = boolidx3.eq(True).mul(2)      
#         data[lab] = out | out2 | out3
#     elif len(t) == 3: 
#         boolidx1 = tmp<=t[0] 
#         boolidx2 =(tmp>t[0]) & (tmp<=t[1]) 
#         boolidx3 =(tmp>t[1]) & (tmp<=t[2])  
#         boolidx4 = tmp>t[2]
#         out = boolidx1.eq(True).mul(0)
#         out2 = boolidx2.eq(True).mul(1)
#         out3 = boolidx3.eq(True).mul(2) 
#         out4 = boolidx4.eq(True).mul(3) 
#         data[lab] = out | out2 | out3 | out4
#     else:
#         print('converting error, out of n cat')
#     return data  