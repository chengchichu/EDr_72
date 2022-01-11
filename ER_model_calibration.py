#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:16:30 2021

@author: anpo
"""
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc, classification_report, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import pickle
import numpy as np
from scipy.stats import chi2
# get y_scores output from ER_LG2.py
# with open('y_scores.pickle', 'wb') as handle:
#      pickle.dump(y_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

file_to_read = open("model_out_abdominal_pain.pickle", "rb")
modelout = pickle.load(file_to_read)   

file_to_read = open("y_test.pickle", "rb")
y_test = pickle.load(file_to_read) 


cols = ['lg', 'rf', 'xgb', 'ec'] # 0.15 0.18 0.2, 0.15, 0.2
colors = ['black','red','blue','green']
cnt = 0

mAP = []

for i in cols:
   
    precision, recall, thresholds = precision_recall_curve(y_test, modelout[i][4], pos_label = 1)
    mAP.append(average_precision_score(y_test, modelout[i][4]))
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.nanargmax(fscore)
    f1_max = np.nanmax(fscore)

    # plot the roc curve for the model
    # no_skill = len(testy[testy==1]) / len(testy)
    # pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label=i, markersize=0.3, color=colors[cnt])
    # pyplot.scatter(recall[ix], precision[ix], marker='o', color=colors[cnt], label='Best', s =50)
    cnt+= 1

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()


brier_scores = []
HL_test = []
cnt = 0
for i in cols:
   
    # fop, mpv = calibration_curve(y_scores['test_label'].values, y_scores[i], n_bins=20, normalize =True)
    fop, mpv = calibration_curve(y_test, modelout[i][4], n_bins=20, normalize =True)
    br_loss = brier_score_loss(y_test, modelout[i][4])
    brier_scores.append(br_loss)
    
    pyplot.plot(mpv, fop, marker='.', color=colors[cnt])

    cnt+=1
    pyplot.xlabel('predicted prob')
    pyplot.ylabel('observed prob')
    # legned()
    # plot the roc curve for the model
    # no_skill = len(testy[testy==1]) / len(testy)
    # pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    # pyplot.plot(recall, precision, marker='.', label=i, markersize=0.3, color=colors[cnt])
    # pyplot.scatter(recall[ix], precision[ix], marker='o', color=colors[cnt], label='Best', s =50)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.legend(['lg','rf','xgb','ec'])
# show the plot
# pyplot.show()   
# axis labels
# pyplot.xlabel('Recall')
# pyplot.ylabel('Precision')
# pyplot.legend()
# # show the plot
# pyplot.show()

# This could be made into a neat function of Hosmer-Lemeshow. 
def HosmerLemeshow (data,g):
    # pihat=yscore
    # # bins = [0,25,50,75,100]
    # bins = list(range(0,100,5))
    # nbin = len(bins)-1
    # pihatcat=pd.cut(pihat, np.percentile(pihat,bins),labels=False,include_lowest=True) #here we've chosen only 4 groups
    # meanprobs =[0]*nbin
    # expevents =[0]*nbin
    # obsevents =[0]*nbin 
    # meanprobs2=[0]*nbin 
    # expevents2=[0]*nbin
    # obsevents2=[0]*nbin 

    data_st = data.sort_values('ys')
    data_st['dcl'] = pd.qcut(data_st['ys'],g)
    
    ys = data_st['y'].groupby(data_st.dcl).sum()
    yt = data_st['y'].groupby(data_st.dcl).count()
    yn = yt - ys
    
    yps = data_st['ys'].groupby(data_st.dcl).sum()
    ypt = data_st['ys'].groupby(data_st.dcl).count()
    ypn = ypt - yps
    
    hltest = (((ys - yps) ** 2 / yps) + ((yn - ypn) ** 2 / ypn)).sum()
    pval = 1 - chi2.cdf(hltest, g - 2)
    
    
    return pval
    # for i in range(nbin):
    #     meanprobs[i]=np.mean(pihat[pihatcat==i])
    #     expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
    #     obsevents[i]=np.sum(Y[pihatcat==i])
    #     meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
    #     expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
    #     obsevents2[i]=np.sum(1-Y[pihatcat==i]) 


    # data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
    # data2={'expevents':expevents,'expevents2':expevents2}
    # data3={'obsevents':obsevents,'obsevents2':obsevents2}
    # m=pd.DataFrame(data1)
    # e=pd.DataFrame(data2)
    # o=pd.DataFrame(data3)
    
    # # The statistic for the test, which follows, under the null hypothesis,
    # # The chi-squared distribution with degrees of freedom equal to amount of groups - 2. Thus 4 - 2 = 2
    # tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) 
    # pvalue=1-chi2.cdf(tt,nbin-2)
    
    

    # return pd.DataFrame([[chi2.cdf(tt,2).round(2), pvalue.round(2)]],columns = ["Chi2", "p - value"])

data = {}
data['ys'] = modelout['ec'][4]
data['y'] = y_test.values
dataP = pd.DataFrame.from_dict(data)
p= HosmerLemeshow(dataP, 30)
print(p)