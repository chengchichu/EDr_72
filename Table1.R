setwd("~/Downloads/CGMH/re72")

# read.csv(..., fileEncoding="big5") 或者預先將資料存成UTF-8格式
data <- read.csv("CGRDER_v15.csv", header = T)
v15 <- subset(data, select = -c(IDCODE, OPDNO, DRNO, ADMDAT, DGDAT, indate, outdate, 
                                ipdat, CHTNO, next_indate, dur, next_dpt, indate_time, itnm_gr,
                                drID, ICD, 主治醫師, 大分類, 中分類, 細分類, 判別依據, 
                                下次主治醫師, 下次大分類, 下次中分類, 下次細分類, 
                                下次判別依據, ICD3, 同大分類, 同中分類, 同細分類))

# names(v15)
continuous <- c("TMP","PULSE","BPS","BPB","BRTCNT","SPAO2","ER_LOS","age1","ER_visit_30",
                "ER_visit_365","DD_visit_30","DD_visit_365","Dr_VSy","HEIGHT","WEIGHT",
                "exam_TOTAL","lab_TOTAL","Bun_value","Creatine_value","Hb_value",
                "Hct_value","RBC_value","WBC_value","CRP_value","Lactate_value",
                "Procalcitonin_value","SBP1","DBP1","SBP2","DBP2","in_TMP","in_PULSE",
                "in_BPS","in_BPB","in_BRTCNT","in_SPAO2","CTh_T","MRIh_T","Xrayh_T",
                "sugar_value")


category <- names(v15)[which((names(v15)  %in% continuous)==FALSE)]

v15[continuous] <- lapply(v15[continuous], as.numeric)
v15[category] <- lapply(v15[category], as.factor)

library(tableone)
tableOne = CreateTableOne(vars = c(continuous, category), strata = "re72", data = v15, factorVars = category)
table1 = print(tableOne)
write.csv(table1, "/Users/jhih-fonglin/Downloads/CGMH/re72/table1.csv", fileEncoding="big5")
# 在輸出資料時，增加fileEncoding="big5"即可輸出中文


# 以重症(住院 or ICU or Dead)去敘述Table1
v15$Critical <- as.factor(apply(apply(v15[c("next_adm","next_ICU","next_DEAD")],2,as.numeric),1,max))
tableOne_2 = CreateTableOne(vars = c(continuous, category, "Critical"), strata = c("Critical","re72"), data = v15, factorVars = category)
table1_2 = print(tableOne_2)
write.csv(table1_2, "/Users/jhih-fonglin/Downloads/CGMH/re72/table1_2.csv", fileEncoding="big5")






# logistic regression
glm_data <- v15
# for (i in 1:ncol(glm_data)) {
#   if (length(levels(glm_data[,i]))<2) {
#     glm_data[,i] <- as.numeric(as.character(glm_data[,i]))
#     # dele[i] <- names(glm_data[,i])
#   } else {glm_data[,i] <- glm_data[,i]}
# }


Odds_Ratio <- data.frame()
rname <- c()
for (i in 1:length(glm_data)) {
  coef <- tryCatch(summary(glm(re72 ~ glm_data[, i], data = glm_data, family = "binomial") ), error=function(e) NULL)$coef[-1,]
  colnames(Odds_Ratio) <- colnames(coef)

  Odds_Ratio <- rbind(Odds_Ratio, coef)
  
  rownames(Odds_Ratio) <- NULL
  rname <- if (is.factor(glm_data[, i])==T & length(levels(glm_data[, i])[-1])<1) {
    c(rname)
  } else if (is.factor(glm_data[, i])==T & length(levels(glm_data[, i])[-1])>=1) {
    c(rname,paste0(names(glm_data)[i], levels(glm_data[, i])[-1]))
  } else { c(rname,names(glm_data)[i]) }
  
  rownames(Odds_Ratio) <- rname

}
colnames(Odds_Ratio) <- c("Estimate","Std. Error","z value","Pr(>|z|)")
Odds_Ratio$Odds_Ratio <- exp(Odds_Ratio$Estimate)
Odds.Ratio <- subset(Odds_Ratio,select = c(Odds_Ratio, `Pr(>|z|)`))
write.csv(Odds.Ratio, "/Users/jhih-fonglin/Downloads/CGMH/re72/Odds Ratio.csv", fileEncoding="big5")


## =======================================================================================================================
# 針對腹痛族群計算Table1
## =======================================================================================================================
train <- read.csv("V15_ABPAIN_TRAIN.csv", header = T, fileEncoding="big5")
test <- read.csv("V15_ABPAIN_TEST.csv", header = T, fileEncoding="big5")
train$ER_LOS <- train$ER_LOS*60
test$ER_LOS <- test$ER_LOS*60
# test_ol <- read.csv("V15_ABPAIN_TEST1.csv", header = T, fileEncoding="big5")
abpain <- rbind(train, test)
# abpain <- rbind(train, test_ol)


# INTY=5 救護車到院
train1 <- subset(train, select = c(age1, SEX, INTY, ER_visit_365, ANISICCLSF_C, ER_LOS,
                                   TMP, PULSE, BPS, BPB, BRTCNT, blood_lab, xray, 
                                   Echo, ct, next_adm, next_DEAD, re72))

test1 <- subset(test, select = c(age1, SEX, INTY, ER_visit_365, ANISICCLSF_C, ER_LOS,
                                   TMP, PULSE, BPS, BPB, BRTCNT, blood_lab, xray, 
                                   Echo, ct, next_adm, next_DEAD, re72))

abpain1 <- subset(abpain, select = c(age1, SEX, INTY, ER_visit_365, ANISICCLSF_C, ER_LOS,
                                   TMP, PULSE, BPS, BPB, BRTCNT, blood_lab, xray, 
                                   Echo, ct, next_adm, next_DEAD, re72))



continuous <- c("age1", "ER_visit_365", "ER_LOS", "TMP", "PULSE", "BPS", "BPB", "BRTCNT")

category <- c("SEX", "ANISICCLSF_C", "INTY", "blood_lab", "xray", "Echo", "ct", "next_adm", "next_DEAD", "re72")

train1[continuous] <- lapply(train1[continuous], as.numeric)
train1[category] <- lapply(train1[category], as.factor)
test1[continuous] <- lapply(test1[continuous], as.numeric)
test1[category] <- lapply(test1[category], as.factor)
abpain1[continuous] <- lapply(abpain1[continuous], as.numeric)
abpain1[category] <- lapply(abpain1[category], as.factor)
# All total
table(train1$re72)+table(test1$re72)

# Training set & no 72-hour return 
t(data.frame(lapply(train1[continuous][train1$re72==0,], function(x) paste0(median(x, na.rm=T), " (", round(quantile(x, 1/4, na.rm=T), 1), "-", round(quantile(x, 3/4, na.rm=T), 1), ")"))))
lapply(train1[category][train1$re72==0,], function(x) paste0(table(x), " (", round(prop.table(table(x))*100,2), "%)"))


# Training set & 72-hour return 
t(data.frame(lapply(train1[continuous][train1$re72==1,], function(x) paste0(median(x, na.rm=T), " (", round(quantile(x, 1/4, na.rm=T), 1), "-", round(quantile(x, 3/4, na.rm=T), 1), ")"))))
lapply(train1[category][train1$re72==1,], function(x) paste0(table(x), " (", round(prop.table(table(x))*100,2), "%)"))
CreateTableOne(vars = c(continuous, category), strata = c("re72"), data = train1, factorVars = category)

# ======================================================================================================================================
# Testing set & no 72-hour return 
t(data.frame(lapply(test1[continuous][test1$re72==0,], function(x) paste0(median(x, na.rm=T), " (", round(quantile(x, 1/4, na.rm=T), 1), "-", round(quantile(x, 3/4, na.rm=T), 1), ")"))))
table(test1$ANISICCLSF_C)
lapply(test1[category][test1$re72==0,], function(x) paste0(table(x), " (", round(prop.table(table(x))*100,2), "%)"))

# Testing set & 72-hour return 
t(data.frame(lapply(test1[continuous][test1$re72==1,], function(x) paste0(median(x, na.rm=T), " (", round(quantile(x, 1/4, na.rm=T), 1), "-", round(quantile(x, 3/4, na.rm=T), 1), ")"))))
lapply(test1[category][test1$re72==1,], function(x) paste0(table(x), " (", round(prop.table(table(x))*100,2), "%)"))
CreateTableOne(vars = c(continuous, category), strata = c("re72"), data = test1, factorVars = category)


# All, p-value
All_Table1 <- CreateTableOne(vars = c(continuous, category), strata = c("re72"), data = abpain1, factorVars = category)
summary(All_Table1)
All_Table1

# All, median + IQR
paste0(median(abpain1["BPB"][abpain1$re72==0,], na.rm = T)," (",quantile(abpain1["BPB"][abpain1$re72==0,],1/4, na.rm = T),"-",quantile(abpain1["BPB"][abpain1$re72==0,],3/4, na.rm = T),")")
paste0(median(abpain1["BPB"][abpain1$re72==1,], na.rm = T)," (",quantile(abpain1["BPB"][abpain1$re72==1,],1/4, na.rm = T),"-",quantile(abpain1["BPB"][abpain1$re72==1,],3/4, na.rm = T),")")


# 樣本住院的比例
paste0(table(train1$next_adm)[[2]], " (", round(prop.table(table(train1$next_adm))[[2]]*100,2), "%)")
paste0(table(test1$next_adm)[[2]], " (", round(prop.table(table(test1$next_adm))[[2]]*100,2), "%)")
paste0(table(abpain1$next_adm)[[2]], " (", round(prop.table(table(abpain1$next_adm))[[2]]*100,2), "%)")














