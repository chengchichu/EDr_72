#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:28:16 2021

@author: chengchichu
"""

import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import numpy as np
from imblearn.over_sampling import SMOTE

def anova_one(y_label, df, cols, major_idx, minor_idx):    
    
    # SMOTE algorithm for upsampling minor class
    #oversample = SMOTE() 不確定這樣 balance is for ML. but not for determining the significance of the feature?
    
    # need to sort before using this method
    dfs = df.sort_values(by=['re72'])  
    ps = []
    Fs = []
    for i in cols:
        major = dfs[i][y_label==0].values
        minor = dfs[i][y_label==1].values
        #F, p = f_oneway(major[major_idx], minor[minor_idx]) # bootstrap
        
        # ###
        reX, rey = oversample.fit_resample(df[i].values.reshape(-1,1),y_label.values)
        F, p = f_oneway(reX[rey == 0], reX[rey == 1])
        
        ps.append([i, p])
        Fs.append(F)
    return ps, Fs

def chi2test(y_label, df, cols2):
    chips = []
    contingency_tables = []
    CramerV_corrs = []
    for i in cols2:
        confusion_matrix = pd.crosstab(df[i], y_label, margins = False)                                
        stat, p, dof, expected = chi2_contingency(confusion_matrix) 
        chips.append([i, p])    
        contingency_tables.append(confusion_matrix)
        # Cramer V stats    
        chi2 = stat
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        try:
             CramerV_corr = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))  # can't do zero division
        except:
             CramerV_corr = np.nan
        CramerV_corrs.append(CramerV_corr)
    return chips, contingency_tables, CramerV_corrs

# ##########
df0 = pd.read_csv('/Users/chengchichu/Desktop/py/EDr_72/er72_processed_DATA_v10_ccs_converted.csv')

del df0['Unnamed: 0']

# 類別
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

# 連續
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


# bootstrap to balance the data for testing
minor_cnt = sum(df0['re72']==1)
major_cnt = sum(df0['re72']==0)

# becareful in using this method. you need to sort the data first
np.random.seed(0)
major_idx = np.random.choice(major_cnt,int(major_cnt*1.5),replace = True)

np.random.seed(1)
minor_idx = np.random.choice(minor_cnt,int(major_cnt*1.5),replace = True)

ps, Fs = anova_one(df0['re72'], df0, cols2, major_idx, minor_idx)