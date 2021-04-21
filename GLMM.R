library(lme4)
v8_ccs = read.csv("er72_processed_DATA_v8_ccs_converted.csv", header = T)
v8_ccs = v8_ccs[,-1]
for (i in c(2,3,6,9,11,14,19,22,25,26,29:34,36,37,40,41,43,44,46,48,50,52,53,56,58,59,61,63,65:67,69:71,73)) {
  v8_ccs[,i]=as.factor(v8_ccs[,i])
}


# v8_ccs - model0
library(MuMIn)
fm0 <- glmer(re72 ~ RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS
             + (1 | LOC), data = v8_ccs, family="binomial", nAGQ=0)
r.squaredGLMM(fm0)
summary(fm0)
#                    R2m        R2c
# theoretical 0.08288988 0.16191151
# delta       0.01971201 0.03850411

# v8_ccs - model1
fm1 <- glmer(re72 ~ RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS
             + (1 | drID), data = v8_ccs, family="binomial", nAGQ=0)
r.squaredGLMM(fm1)
summary(fm1)


# v8_ccs - model2
fm2 <- glmer(re72 ~ RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS
             + (1 | ccs), data = v8_ccs, family="binomial", nAGQ=0)
r.squaredGLMM(fm2)
summary(fm2)


# # model3(過久->停止)
# fm3 <- glmer(re72 ~ RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS
#              + (1 | LOC)
#              + (0 + RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS | LOC)
#              , data = v8_ccs, family="binomial", nAGQ=0)
# r.squaredGLMM(fm3)
# summary(fm3)
# 
# 
# # model4(過久->停止)
# fm4 <- glmer(re72 ~ RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS
#              + (1 | drID)
#              + (0 + RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS | drID)
#              , data = v8_ccs, family="binomial", nAGQ=0)
# r.squaredGLMM(fm4)
# summary(fm4)
# 
# 
# # model5(過久->停止)
# fm5 <- glmer(re72 ~ RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS
#              + (1 | ccs)
#              + (0 + RBC_value+INTY.1+INTY.2+Dr_VSy+ER_visit_365+week.4+DD_visit_365+GCSV+weekday+ANISICMIGD+TMP+ER_visit_30+Hct_value+week.2+GCSE+Creatine_value+Bun_value+DBP+EKG+DD_visit_30+exam_TOTAL+GCSM+BRTCNT+week+ANISICMIGD_2+ANISICCLSF_C+indate_month.1+MRI+week.1+indate_month+Echo+INTY.7+PULSE+xray+indate_month.6+BPB+Procalcitonin_value+week.3+indate_month.9+WBC_value+indate_month.5+indate_time_gr+ANISICMIGD_3+INTY.5+CRP_value+indate_time_gr.1+Hb_value+INTY+WEIGHT+week.5+INTY.3+SBP+ANISICMIGD_1+ct+lab_TOTAL+indate_month.3+indate_month.8+age1+indate_month.4+Lactate_value+indate_month.7+ER_LOS+INTY.6+SEX+indate_month.2+SPAO2+INTY.4+indate_month.10+BPS | ccs)
#              , data = v8_ccs, family="binomial", nAGQ=0)
# r.squaredGLMM(fm5)
# summary(fm5)


## === 20210401 ===
v8 = read.csv("er72_processed_DATA_v8.csv",header = T)
v8 = v8[,-1]
for (i in c(1:3,5:12,15:23,26,34,36:40,44:54,73:151) ) {
 v8[,i]=as.factor(v8[,i])
}

# v8 - model0
m0 <- glmer(re72 ~ SEX+ANISICCLSF_C+INTY+INTY.1+INTY.2+INTY.3+INTY.4+INTY.5+INTY.6+INTY.7+ER_LOS+age1+week+week.1+week.2+week.3+week.4+week.5+weekday+indate_time_gr+indate_time_gr.1+ER_visit_30+ER_visit_365+TMP+PULSE+BPS+BPB+GCSE+GCSV+GCSM+BRTCNT+SPAO2+DD_visit_30+ct+MRI+xray+EKG+Echo+DD_visit_365+Dr_VSy+WEIGHT+indate_month+indate_month.1+indate_month.2+indate_month.3+indate_month.4+indate_month.5+indate_month.6+indate_month.7+indate_month.8+indate_month.9+indate_month.10+SBP+DBP+exam_TOTAL+lab_TOTAL+ANISICMIGD+ANISICMIGD_1+ANISICMIGD_2+ANISICMIGD_3+Bun_value+CRP_value+Lactate_value+Procalcitonin_value+Creatine_value+Hb_value+Hct_value+RBC_value+WBC_value+dx82+dx142+dx239+dx115+dx147+dx129+dx91+dx185+dx148+dx241+dx199+dx234+dx73+dx232+dx143+dx193+dx111+dx112+dx122+dx151+dx125+dx95+dx79+dx227+dx87+dx80+dx141+dx249+dx146+dx123+dx238+dx113+dx192+dx170+dx7+dx247+dx114+dx127+dx233+dx240+dx138+dx116+dx90+dx55+dx134+dx84+dx124+dx54+dx59+dx220+dx76+dx225+dx163+dx117+dx226+dx139+dx159+dx128+dx11+dx188+dx83+dx255+dx126+dx2+dx156+dx109+dx100+dx97+dx88+dx49+dx72+dx51+dx140+dx50+dx243+dx77+dx78+dx258+dx19
             + (1 | LOC), data = v8, family="binomial", nAGQ=0)
r.squaredGLMM(m0)
summary(m0)
#                    R2m        R2c
# theoretical 0.11248011 0.19111224
# delta       0.02747858 0.04668819

# v8 - model1
m1 <- glmer(re72 ~ SEX+ANISICCLSF_C+INTY+INTY.1+INTY.2+INTY.3+INTY.4+INTY.5+INTY.6+INTY.7+ER_LOS+age1+week+week.1+week.2+week.3+week.4+week.5+weekday+indate_time_gr+indate_time_gr.1+ER_visit_30+ER_visit_365+TMP+PULSE+BPS+BPB+GCSE+GCSV+GCSM+BRTCNT+SPAO2+DD_visit_30+ct+MRI+xray+EKG+Echo+DD_visit_365+Dr_VSy+WEIGHT+indate_month+indate_month.1+indate_month.2+indate_month.3+indate_month.4+indate_month.5+indate_month.6+indate_month.7+indate_month.8+indate_month.9+indate_month.10+SBP+DBP+exam_TOTAL+lab_TOTAL+ANISICMIGD+ANISICMIGD_1+ANISICMIGD_2+ANISICMIGD_3+Bun_value+CRP_value+Lactate_value+Procalcitonin_value+Creatine_value+Hb_value+Hct_value+RBC_value+WBC_value+dx82+dx142+dx239+dx115+dx147+dx129+dx91+dx185+dx148+dx241+dx199+dx234+dx73+dx232+dx143+dx193+dx111+dx112+dx122+dx151+dx125+dx95+dx79+dx227+dx87+dx80+dx141+dx249+dx146+dx123+dx238+dx113+dx192+dx170+dx7+dx247+dx114+dx127+dx233+dx240+dx138+dx116+dx90+dx55+dx134+dx84+dx124+dx54+dx59+dx220+dx76+dx225+dx163+dx117+dx226+dx139+dx159+dx128+dx11+dx188+dx83+dx255+dx126+dx2+dx156+dx109+dx100+dx97+dx88+dx49+dx72+dx51+dx140+dx50+dx243+dx77+dx78+dx258+dx19
             + (1 | drID), data = v8, family="binomial", nAGQ=0)
r.squaredGLMM(m1)
summary(m1)
#                    R2m        R2c
# theoretical 0.11920529 0.11993574
# delta       0.01793204 0.01804192




## === 20210414 ===
setwd("~/Downloads/CGMH/re72")
v10 = read.csv("er72_processed_DATA_v10.csv",header = T)
v10 = v10[,-1]
for (i in c(1:3,5:11,14:22,35:40,43:53,71:374) ) {
  v10[,i]=as.factor(v10[,i])
}

# v10 - model0
m0 <- glmer(re72 ~ SEX+ANISICCLSF_C+INTY+INTY.1+INTY.2+INTY.3+INTY.4+INTY.5+INTY.6+ER_LOS+age1+week+week.1+week.2+week.3+week.4+week.5+weekday+indate_time_gr+indate_time_gr.1+ER_visit_30+ER_visit_365+TMP+PULSE+BPS+BPB+GCSE+GCSV+GCSM+BRTCNT+SPAO2+DD_visit_30+ct+MRI+xray+EKG+Echo+DD_visit_365+Dr_VSy+WEIGHT+indate_month+indate_month.1+indate_month.2+indate_month.3+indate_month.4+indate_month.5+indate_month.6+indate_month.7+indate_month.8+indate_month.9+indate_month.10+SBP+DBP+exam_TOTAL+lab_TOTAL+ANISICMIGD+ANISICMIGD_1+ANISICMIGD_2+ANISICMIGD_3+Bun_value+CRP_value+Lactate_value+Procalcitonin_value+Creatine_value+Hb_value+Hct_value+RBC_value+WBC_value+dx82+dx142+dx239+dx115+dx147+dx129+dx91+dx185+dx148+dx241+dx199+dx234+dx73+dx232+dx143+dx193+dx111+dx112+dx122+dx151+dx125+dx95+dx79+dx227+dx87+dx80+dx141+dx249+dx146+dx123+dx238+dx113+dx192+dx170+dx7+dx247+dx114+dx127+dx233+dx240+dx138+dx116+dx90+dx55+dx134+dx84+dx124+dx54+dx59+dx220+dx76+dx225+dx163+dx117+dx226+dx139+dx159+dx128+dx11+dx188+dx83+dx255+dx126+dx2+dx156+dx109+dx100+dx97+dx88+dx49+dx72+dx51+dx140+dx50+dx243+dx77+dx78+dx258+dx19+dxh87+dxh49+dxh146+dxh88+dxh247+dxh147+dxh111+dxh50+dxh55+dxh59+dxh42+dxh90+dxh6+dxh53+dxh3+dxh139+dxh95+dxh185+dxh2+dxh143+dxh127+dxh145+dxh97+dxh84+dxh141+dxh45+dxh138+dxh226+dxh98+dxh152+dxh225+dxh193+dxh116+dxh149+dxh122+dxh128+dxh106+dxh199+dxh255+dxh119+dxh11+dxh58+dxh151+dxh117+dxh184+dxh54+dxh279+dxh16+dxh102+dxh47+dxh72+dxh115+dxh258+dxh19+dxh134+dxh24+dxh245+dxh48+dxh148+dxh89+dxh63+dxh197+dxh118+dxh123+dxh183+dxh191+dxh85+dxh234+dxh60+dxh120+dxh71+dxh62+dxh132+dxh251+dxh129+dxh4+dxh137+dxh51+dxh14+dxh140+dxh219+dxh7+dxh249+dxh44+dxh150+dxh142+dxh82+dxh198+dxh64+dxh265+dxh181+dxh237+dxh46+dxh195+dxh169+dxh107+dxh241+dxh221+dxh246+dxh109+dxh278+dxh86+dxh68+dxh232+dxh94+dxh217+dxh112+dxh156+dxh1+dxh103+dxh29+dxh262+dxh101+dxh218+dxh257+dxh52+dxh91+dxh15+dxh157+dxh130+dxh190+dxh124+dxh131+dxh214+dxh135+dxh144+dxh92+dxh188+dxh187+dxh269+dxh73+dxh233+dxh125+dxh126+dxh104+dxh32+dxh163+dxh26+dxh186+dxh99+dxh110+dxh80+dxh12+dxh13+dxh239+dxh70+dxh83+dxh227+dxh200
            +atc66+atc59+atc3+atc79+atc19+atc49+atc78+atc2+atc69+atc6+atc71+atc7+atc61+atc81+atc17+atc43+atc45+atc35+atc26+atc77+atc75+atc21+atc16+atc27+atc23+atc10+atc31+atc28+atc12+atc18+atc67+atc36+atc62+atc44+atc42+atc25+atc52+atc41+atc54+atc84+atc72+atc1+atc82+atc70+atc11+atc5+atc29+atc68+atc22+atc30+atc40+atc60+atc46+atc57+atc33+atc20+atc58+atc9+atc4+atc56+atc38+atc55+atc24+atc50+atc51+atc65+atc34+atc74+atc86+atc63+atc32+atc73+atc47+atc39+atc53
            + (1 | DPT2), data = v10, family="binomial", nAGQ=0)
r.squaredGLMM(m0)
#                    R2m        R2c
# theoretical 0.18010418 0.18231375
# delta       0.02754388 0.02788179
summary(m0)


# v10 - model1
m1 <- glmer(re72 ~ SEX+ANISICCLSF_C+INTY+INTY.1+INTY.2+INTY.3+INTY.4+INTY.5+INTY.6+ER_LOS+age1+week+week.1+week.2+week.3+week.4+week.5+weekday+indate_time_gr+indate_time_gr.1+ER_visit_30+ER_visit_365+TMP+PULSE+BPS+BPB+GCSE+GCSV+GCSM+BRTCNT+SPAO2+DD_visit_30+ct+MRI+xray+EKG+Echo+DD_visit_365+Dr_VSy+WEIGHT+indate_month+indate_month.1+indate_month.2+indate_month.3+indate_month.4+indate_month.5+indate_month.6+indate_month.7+indate_month.8+indate_month.9+indate_month.10+SBP+DBP+exam_TOTAL+lab_TOTAL+ANISICMIGD+ANISICMIGD_1+ANISICMIGD_2+ANISICMIGD_3+Bun_value+CRP_value+Lactate_value+Procalcitonin_value+Creatine_value+Hb_value+Hct_value+RBC_value+WBC_value+dx82+dx142+dx239+dx115+dx147+dx129+dx91+dx185+dx148+dx241+dx199+dx234+dx73+dx232+dx143+dx193+dx111+dx112+dx122+dx151+dx125+dx95+dx79+dx227+dx87+dx80+dx141+dx249+dx146+dx123+dx238+dx113+dx192+dx170+dx7+dx247+dx114+dx127+dx233+dx240+dx138+dx116+dx90+dx55+dx134+dx84+dx124+dx54+dx59+dx220+dx76+dx225+dx163+dx117+dx226+dx139+dx159+dx128+dx11+dx188+dx83+dx255+dx126+dx2+dx156+dx109+dx100+dx97+dx88+dx49+dx72+dx51+dx140+dx50+dx243+dx77+dx78+dx258+dx19+dxh87+dxh49+dxh146+dxh88+dxh247+dxh147+dxh111+dxh50+dxh55+dxh59+dxh42+dxh90+dxh6+dxh53+dxh3+dxh139+dxh95+dxh185+dxh2+dxh143+dxh127+dxh145+dxh97+dxh84+dxh141+dxh45+dxh138+dxh226+dxh98+dxh152+dxh225+dxh193+dxh116+dxh149+dxh122+dxh128+dxh106+dxh199+dxh255+dxh119+dxh11+dxh58+dxh151+dxh117+dxh184+dxh54+dxh279+dxh16+dxh102+dxh47+dxh72+dxh115+dxh258+dxh19+dxh134+dxh24+dxh245+dxh48+dxh148+dxh89+dxh63+dxh197+dxh118+dxh123+dxh183+dxh191+dxh85+dxh234+dxh60+dxh120+dxh71+dxh62+dxh132+dxh251+dxh129+dxh4+dxh137+dxh51+dxh14+dxh140+dxh219+dxh7+dxh249+dxh44+dxh150+dxh142+dxh82+dxh198+dxh64+dxh265+dxh181+dxh237+dxh46+dxh195+dxh169+dxh107+dxh241+dxh221+dxh246+dxh109+dxh278+dxh86+dxh68+dxh232+dxh94+dxh217+dxh112+dxh156+dxh1+dxh103+dxh29+dxh262+dxh101+dxh218+dxh257+dxh52+dxh91+dxh15+dxh157+dxh130+dxh190+dxh124+dxh131+dxh214+dxh135+dxh144+dxh92+dxh188+dxh187+dxh269+dxh73+dxh233+dxh125+dxh126+dxh104+dxh32+dxh163+dxh26+dxh186+dxh99+dxh110+dxh80+dxh12+dxh13+dxh239+dxh70+dxh83+dxh227+dxh200
            +atc66+atc59+atc3+atc79+atc19+atc49+atc78+atc2+atc69+atc6+atc71+atc7+atc61+atc81+atc17+atc43+atc45+atc35+atc26+atc77+atc75+atc21+atc16+atc27+atc23+atc10+atc31+atc28+atc12+atc18+atc67+atc36+atc62+atc44+atc42+atc25+atc52+atc41+atc54+atc84+atc72+atc1+atc82+atc70+atc11+atc5+atc29+atc68+atc22+atc30+atc40+atc60+atc46+atc57+atc33+atc20+atc58+atc9+atc4+atc56+atc38+atc55+atc24+atc50+atc51+atc65+atc34+atc74+atc86+atc63+atc32+atc73+atc47+atc39+atc53
            + (1 | drID), data = v10, family="binomial", nAGQ=0)
r.squaredGLMM(m1)
#                    R2m        R2c
# theoretical 0.18406169 0.18439330
# delta       0.02952307 0.02957626
summary(m1)



## ==== v10 - ccs ====
v10_ccs = read.csv("er72_processed_DATA_v10_ccs_converted.csv", header = T)
v10_ccs = v10_ccs[,-1]
for (i in c(1,2,4:10,13:21,34:38,42:52,70:294)) {
  v10_ccs[,i]=as.factor(v10_ccs[,i])
}



# v10_ccs - model2
fm2<-glmer(re72 ~ SEX+ANISICCLSF_C+INTY+INTY.1+INTY.2+INTY.3+INTY.4+INTY.5+INTY.6+ER_LOS+age1+week+week.1+week.2+week.3+week.4+week.5+weekday+indate_time_gr+indate_time_gr.1+ER_visit_30+ER_visit_365+TMP+PULSE+BPS+BPB+GCSE+GCSV+GCSM+BRTCNT+SPAO2+DD_visit_30+ct+MRI+xray+EKG+Echo+DD_visit_365+Dr_VSy+WEIGHT+indate_month+indate_month.1+indate_month.2+indate_month.3+indate_month.4+indate_month.5+indate_month.6+indate_month.7+indate_month.8+indate_month.9+indate_month.10+SBP+DBP+exam_TOTAL+lab_TOTAL+ANISICMIGD+ANISICMIGD_1+ANISICMIGD_2+ANISICMIGD_3+Bun_value+CRP_value+Lactate_value+Procalcitonin_value+Creatine_value+Hb_value+Hct_value+RBC_value+WBC_value+dxh87+dxh49+dxh146+dxh88+dxh247+dxh147+dxh111+dxh50+dxh55+dxh59+dxh42+dxh90+dxh6+dxh53+dxh3+dxh139+dxh95+dxh185+dxh2+dxh143+dxh127+dxh145+dxh97+dxh84+dxh141+dxh45+dxh138+dxh226+dxh98+dxh152+dxh225+dxh193+dxh116+dxh149+dxh122+dxh128+dxh106+dxh199+dxh255+dxh119+dxh11+dxh58+dxh151+dxh117+dxh184+dxh54+dxh279+dxh16+dxh102+dxh47+dxh72+dxh115+dxh258+dxh19+dxh134+dxh24+dxh245+dxh48+dxh148+dxh89+dxh63+dxh197+dxh118+dxh123+dxh183+dxh191+dxh85+dxh234+dxh60+dxh120+dxh71+dxh62+dxh132+dxh251+dxh129+dxh4+dxh137+dxh51+dxh14+dxh140+dxh219+dxh7+dxh249+dxh44+dxh150+dxh142+dxh82+dxh198+dxh64+dxh265+dxh181+dxh237+dxh46+dxh195+dxh169+dxh107+dxh241+dxh221+dxh246+dxh109+dxh278+dxh86+dxh68+dxh232+dxh94+dxh217+dxh112+dxh156+dxh1+dxh103+dxh29+dxh262+dxh101+dxh218+dxh257+dxh52+dxh91+dxh15+dxh157+dxh130+dxh190+dxh124+dxh131+dxh214+dxh135+dxh144+dxh92+dxh188+dxh187+dxh269+dxh73+dxh233+dxh125+dxh126+dxh104+dxh32+dxh163+dxh26+dxh186+dxh99+dxh110+dxh80+dxh12+dxh13+dxh239+dxh70+dxh83+dxh227+dxh200
           +atc66+atc59+atc3+atc79+atc19+atc49+atc78+atc2+atc69+atc6+atc71+atc7+atc61+atc81+atc17+atc43+atc45+atc35+atc26+atc77+atc75+atc21+atc16+atc27+atc23+atc10+atc31+atc28+atc12+atc18+atc67+atc36+atc62+atc44+atc42+atc25+atc52+atc41+atc54+atc84+atc72+atc1+atc82+atc70+atc11+atc5+atc29+atc68+atc22+atc30+atc40+atc60+atc46+atc57+atc33+atc20+atc58+atc9+atc4+atc56+atc38+atc55+atc24+atc50+atc51+atc65+atc34+atc74+atc86+atc63+atc32+atc73+atc47+atc39+atc53
           + (1 | ccs), data = v10_ccs, family="binomial", nAGQ=0)
r.squaredGLMM(fm2)
#                    R2m        R2c
# theoretical 0.15113226 0.16926484
# delta       0.02439935 0.02732674
summary(fm2)





library(lmtest)
model1 <- glm(re72 ~ SEX+ANISICCLSF_C+INTY+INTY.1+INTY.2+INTY.3+INTY.4+INTY.5+INTY.6+ER_LOS+age1+week+week.1+week.2+week.3+week.4+week.5+weekday+indate_time_gr+indate_time_gr.1+ER_visit_30+ER_visit_365+TMP+PULSE+BPS+BPB+GCSE+GCSV+GCSM+BRTCNT+SPAO2+DD_visit_30+ct+MRI+xray+EKG+Echo+DD_visit_365+Dr_VSy+WEIGHT+indate_month+indate_month.1+indate_month.2+indate_month.3+indate_month.4+indate_month.5+indate_month.6+indate_month.7+indate_month.8+indate_month.9+indate_month.10+SBP+DBP+exam_TOTAL+lab_TOTAL+ANISICMIGD+ANISICMIGD_1+ANISICMIGD_2+ANISICMIGD_3+Bun_value+CRP_value+Lactate_value+Procalcitonin_value+Creatine_value+Hb_value+Hct_value+RBC_value+WBC_value+dx82+dx142+dx239+dx115+dx147+dx129+dx91+dx185+dx148+dx241+dx199+dx234+dx73+dx232+dx143+dx193+dx111+dx112+dx122+dx151+dx125+dx95+dx79+dx227+dx87+dx80+dx141+dx249+dx146+dx123+dx238+dx113+dx192+dx170+dx7+dx247+dx114+dx127+dx233+dx240+dx138+dx116+dx90+dx55+dx134+dx84+dx124+dx54+dx59+dx220+dx76+dx225+dx163+dx117+dx226+dx139+dx159+dx128+dx11+dx188+dx83+dx255+dx126+dx2+dx156+dx109+dx100+dx97+dx88+dx49+dx72+dx51+dx140+dx50+dx243+dx77+dx78+dx258+dx19+dxh87+dxh49+dxh146+dxh88+dxh247+dxh147+dxh111+dxh50+dxh55+dxh59+dxh42+dxh90+dxh6+dxh53+dxh3+dxh139+dxh95+dxh185+dxh2+dxh143+dxh127+dxh145+dxh97+dxh84+dxh141+dxh45+dxh138+dxh226+dxh98+dxh152+dxh225+dxh193+dxh116+dxh149+dxh122+dxh128+dxh106+dxh199+dxh255+dxh119+dxh11+dxh58+dxh151+dxh117+dxh184+dxh54+dxh279+dxh16+dxh102+dxh47+dxh72+dxh115+dxh258+dxh19+dxh134+dxh24+dxh245+dxh48+dxh148+dxh89+dxh63+dxh197+dxh118+dxh123+dxh183+dxh191+dxh85+dxh234+dxh60+dxh120+dxh71+dxh62+dxh132+dxh251+dxh129+dxh4+dxh137+dxh51+dxh14+dxh140+dxh219+dxh7+dxh249+dxh44+dxh150+dxh142+dxh82+dxh198+dxh64+dxh265+dxh181+dxh237+dxh46+dxh195+dxh169+dxh107+dxh241+dxh221+dxh246+dxh109+dxh278+dxh86+dxh68+dxh232+dxh94+dxh217+dxh112+dxh156+dxh1+dxh103+dxh29+dxh262+dxh101+dxh218+dxh257+dxh52+dxh91+dxh15+dxh157+dxh130+dxh190+dxh124+dxh131+dxh214+dxh135+dxh144+dxh92+dxh188+dxh187+dxh269+dxh73+dxh233+dxh125+dxh126+dxh104+dxh32+dxh163+dxh26+dxh186+dxh99+dxh110+dxh80+dxh12+dxh13+dxh239+dxh70+dxh83+dxh227+dxh200
            +atc66+atc59+atc3+atc79+atc19+atc49+atc78+atc2+atc69+atc6+atc71+atc7+atc61+atc81+atc17+atc43+atc45+atc35+atc26+atc77+atc75+atc21+atc16+atc27+atc23+atc10+atc31+atc28+atc12+atc18+atc67+atc36+atc62+atc44+atc42+atc25+atc52+atc41+atc54+atc84+atc72+atc1+atc82+atc70+atc11+atc5+atc29+atc68+atc22+atc30+atc40+atc60+atc46+atc57+atc33+atc20+atc58+atc9+atc4+atc56+atc38+atc55+atc24+atc50+atc51+atc65+atc34+atc74+atc86+atc63+atc32+atc73+atc47+atc39+atc53
            , data = v10, family="binomial")
lrtest(model1,m0)
#    Df LogLik Df  Chisq Pr(>Chisq)    
# 1 390 -29599                         
# 2 391 -29583  1 32.165  1.416e-08 ***


lrtest(model1,m1)
# 不行


model2<-glm(re72 ~ SEX+ANISICCLSF_C+INTY+INTY.1+INTY.2+INTY.3+INTY.4+INTY.5+INTY.6+ER_LOS+age1+week+week.1+week.2+week.3+week.4+week.5+weekday+indate_time_gr+indate_time_gr.1+ER_visit_30+ER_visit_365+TMP+PULSE+BPS+BPB+GCSE+GCSV+GCSM+BRTCNT+SPAO2+DD_visit_30+ct+MRI+xray+EKG+Echo+DD_visit_365+Dr_VSy+WEIGHT+indate_month+indate_month.1+indate_month.2+indate_month.3+indate_month.4+indate_month.5+indate_month.6+indate_month.7+indate_month.8+indate_month.9+indate_month.10+SBP+DBP+exam_TOTAL+lab_TOTAL+ANISICMIGD+ANISICMIGD_1+ANISICMIGD_2+ANISICMIGD_3+Bun_value+CRP_value+Lactate_value+Procalcitonin_value+Creatine_value+Hb_value+Hct_value+RBC_value+WBC_value+dxh87+dxh49+dxh146+dxh88+dxh247+dxh147+dxh111+dxh50+dxh55+dxh59+dxh42+dxh90+dxh6+dxh53+dxh3+dxh139+dxh95+dxh185+dxh2+dxh143+dxh127+dxh145+dxh97+dxh84+dxh141+dxh45+dxh138+dxh226+dxh98+dxh152+dxh225+dxh193+dxh116+dxh149+dxh122+dxh128+dxh106+dxh199+dxh255+dxh119+dxh11+dxh58+dxh151+dxh117+dxh184+dxh54+dxh279+dxh16+dxh102+dxh47+dxh72+dxh115+dxh258+dxh19+dxh134+dxh24+dxh245+dxh48+dxh148+dxh89+dxh63+dxh197+dxh118+dxh123+dxh183+dxh191+dxh85+dxh234+dxh60+dxh120+dxh71+dxh62+dxh132+dxh251+dxh129+dxh4+dxh137+dxh51+dxh14+dxh140+dxh219+dxh7+dxh249+dxh44+dxh150+dxh142+dxh82+dxh198+dxh64+dxh265+dxh181+dxh237+dxh46+dxh195+dxh169+dxh107+dxh241+dxh221+dxh246+dxh109+dxh278+dxh86+dxh68+dxh232+dxh94+dxh217+dxh112+dxh156+dxh1+dxh103+dxh29+dxh262+dxh101+dxh218+dxh257+dxh52+dxh91+dxh15+dxh157+dxh130+dxh190+dxh124+dxh131+dxh214+dxh135+dxh144+dxh92+dxh188+dxh187+dxh269+dxh73+dxh233+dxh125+dxh126+dxh104+dxh32+dxh163+dxh26+dxh186+dxh99+dxh110+dxh80+dxh12+dxh13+dxh239+dxh70+dxh83+dxh227+dxh200
           +atc66+atc59+atc3+atc79+atc19+atc49+atc78+atc2+atc69+atc6+atc71+atc7+atc61+atc81+atc17+atc43+atc45+atc35+atc26+atc77+atc75+atc21+atc16+atc27+atc23+atc10+atc31+atc28+atc12+atc18+atc67+atc36+atc62+atc44+atc42+atc25+atc52+atc41+atc54+atc84+atc72+atc1+atc82+atc70+atc11+atc5+atc29+atc68+atc22+atc30+atc40+atc60+atc46+atc57+atc33+atc20+atc58+atc9+atc4+atc56+atc38+atc55+atc24+atc50+atc51+atc65+atc34+atc74+atc86+atc63+atc32+atc73+atc47+atc39+atc53
           , data = v10_ccs, family="binomial")
lrtest(model2,fm2)
#    Df LogLik Df  Chisq Pr(>Chisq)    
# 1 289 -29849                         
# 2 290 -29733  1 231.69  < 2.2e-16 ***


