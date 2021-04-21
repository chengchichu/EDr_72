#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:51:00 2021

@author: chengchichu
"""

import pandas as pd
# from ER_LG2 import add_cat, add_avg, add_cut, get_ICD_cat
from scipy.stats import chi2_contingency

def anova_one(y_label, df, cols):
    # x1 = y_label == 1
    # x2 = y_label == 0
    # idx = np.where(x1.values)[0]
    # idx2 = np.where(x2.values)[0]
    # h, p = stats.ttest_ind(xdata[idx,cols],  xdata[idx2,cols])
    ps = []
    Fs = []
    for i in cols:
        F, p = stats.f_oneway(df0[i][y_label==0], df[i][y_label==1])
        ps.append(p)
        Fs.append(F)
    return ps, Fs

def chi2test(y_label, df, cols2):
    chips = []
    contingency_tables = []
    CramerV_corrs = []
    for i in cols2:
        confusion_matrix = pd.crosstab(df[i], y_label, margins = False)                                
        stat, p, dof, expected = chi2_contingency(confusion_matrix) 
        chips.append([i,p])    
        contingency_tables.append(confusion_matrix)
        # Cramer V stats    
        chi2 = stat
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        CramerV_corr = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
        CramerV_corrs.append(CramerV_corr)
    return chips, contingency_tables, CramerV_corrs

df0 = pd.read_csv('/home/anpo/Desktop/pyscript/EDr_72/er72_processed_DATA_v10_ccs_converted.csv')


cols = {}
cols['indate_month'] = 0
cols['DPT2'] = 0
cols['SEX'] = 0 
cols['INTY'] = 0
cols['week'] = 0
cols['weekday'] = 2
cols['indate_time_gr'] = 0
cols['ct'] = 2
cols['MRI'] = 2
cols['xray'] = 2
cols['EKG'] = 2
cols['Echo'] = 2
cols['GCSE'] = 2
cols['GCSV'] = 2
cols['GCSM'] = 2
cols['ANISICCLSF_C'] = 2
cols['ANISICMIGD'] = 2
cols['ANISICMIGD_1'] = 2
cols['ANISICMIGD_2'] = 2
cols['ANISICMIGD_3'] = 2

cols2 = list(cols.keys())

chips, contingency_tables, CramerV_corrs = chi2test(df0['re72'], df0, cols2)

cols = {}
cols['ER_LOS'] = 1
cols['age1'] = 1
cols['ER_visit_30'] = 1 # 
cols['ER_visit_365'] = 1
cols['TMP'] = 1
cols['PULSE'] = 1
cols['BPS'] = 1
cols['BPB'] = 1
cols['BRTCNT'] = 1
cols['SPAO2'] = 1
cols['DD_visit_30'] = 1
cols['DD_visit_365'] = 1
cols['Dr_VSy'] = 1
cols['WEIGHT'] = 1
cols['SBP'] = 1
cols['DBP'] = 1
cols['Bun_value'] = 1
cols['CRP_value'] = 1
cols['Lactate_value'] = 1
cols['Procalcitonin_value'] = 1    
cols['Creatine_value'] = 1
cols['Hb_value'] = 1
cols['Hct_value'] = 1
cols['RBC_value'] = 1
cols['WBC_value'] = 1
cols['exam_TOTAL'] = 2
cols['lab_TOTAL'] = 2

cols2 = list(cols.keys())

ps, Fs = anova_one(df0['re72'], df0, cols2)















# cols = {}
# cols['LOC'] = 0 
# cols['SEX'] = 0 
# #cols['DPT'] = 0
# cols['ANISICCLSF_C'] = 2
# cols['INTY'] = 0
# cols['ER_LOS'] = 1
# cols['age1'] = 1
# cols['week'] = 0
# cols['weekday'] = 2
# cols['indate_time_gr'] = 0
# cols['ER_visit_30'] = 2
# cols['ER_visit_365'] = 2
# cols['TMP'] = 0	
# cols['PULSE'] = 0
# cols['BPS'] = 0
# cols['GCSE'] = 2
# cols['GCSV'] = 2
# cols['GCSM'] = 2
# cols['BRTCNT'] = 0
# cols['SPAO2'] = 2
# cols['DD_visit_30'] = 2
# cols['ct'] = 2
# cols['MRI'] = 2
# cols['xray'] = 2
# cols['EKG'] = 2
# cols['Echo'] = 2
# cols['blood_lab'] = 2 
# cols['urine_lab'] = 2 
# cols['WBC_lab'] = 2
# cols['Hb_Lab'] = 2
# cols['Hct_lab'] = 2
# cols['Bun_lab'] = 2
# cols['Creatine_lab'] = 2
# cols['CRP_lab'] = 2
# cols['Procalcitonin_lab'] = 2
# cols['Lactate_lab'] = 2
# cols['ICD3'] = 0

# df_cat = df[cols.keys()]
# X = df_cat

# #y24 = df['re24']
# y72 = df['re72'] 

# ## Data imputation
# # 到院方式, 缺失當new category, 其他中位數避免 outlier, missing的多少？
# X = add_cat(X, 'INTY', 10)
# X = add_cat(X, 'ICD3', 'NNN') #缺失的類別
# X = add_avg(X, 'ER_LOS'	)
# X = add_avg(X, 'TMP')
# X = add_avg(X, 'PULSE')
# X = add_avg(X, 'BPS')
# X = add_avg(X, 'BRTCNT')
# X = add_avg(X, 'SPAO2')
# # vital sign 類別化
# X = add_cut(X, 'TMP', [36, 37.5, 39]) # 體溫
# X = add_cut(X, 'PULSE', [60, 100]) # 
# X = add_cut(X, 'BPS', [90, 130]) # 收縮壓
# X = add_cut(X, 'BRTCNT', [12, 20]) #
# X = add_cut(X, 'SPAO2', [94]) #

# #抓 ICD code的第一個字母
# X = get_ICD_cat(X, 'ICD3')

# # 
# X_ = X
# y72_ = y72

# cat_var = []
# for i, j in cols.items():
#     if j != 1:
#        cat_var.append(i)  

# chips, contingency_tables = chi2test(y72_, X_, cat_var)
