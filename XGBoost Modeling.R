library(tidyverse)
library(readxl)
library(xgboost)
library(caret)
setwd("~/Desktop/Kaggle Compatition/Credit Card Fraud/archive/raw_data")

#1.Import Data
train<-read_csv('train_smote.csv')
str(train)
colnames(train)
train$category<-as.factor(train$category)
train$gender<-as.factor(train$gender)
train$city<-as.factor(train$city)
train$zip<-as.factor(train$zip)
train$state<-as.factor(train$state)
train$job<-as.factor(train$job)
train$is_fraud<-as.factor(train$is_fraud)
train$weekday<-as.factor(train$weekday)
which(is.na(train))

val<-read_csv('validation.csv')
str(val)
colnames(val)
val$category<-as.factor(val$category)
val$gender<-as.factor(val$gender)
val$city<-as.factor(val$city)
val$zip<-as.factor(val$zip)
val$state<-as.factor(val$state)
val$job<-as.factor(val$job)
val$is_fraud<-as.factor(val$is_fraud)
val$weekday<-as.factor(val$weekday)
which(is.na(val))

#2.XGBoost
setwd("~/Desktop/Kaggle Compatition/Credit Card Fraud/archive/model")
y<-train$is_fraud
#Default: eta=0.3, gamma=0, max_depth=6 
XGB_default<-xgboost(data = data.matrix(train[,-13]),
                     label = y,
                     nrounds = 25,
                     eval_metric = "merror",
                     num_class=2)

y_pred <- predict(XGB_default, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_default,'XGBoost_eta0.3_maxdepth6.rds')

#Second Try: eta=0.3, gamma=0, max_depth=8
XGB_eta0.3_maxdepth8<-xgboost(data = data.matrix(train[,-13]),
                               label = y,
                               nrounds = 25,
                               eval_metric = "merror",
                               num_class=2,
                               max_depth=8)

y_pred <- predict(XGB_eta0.3_maxdepth8, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_eta0.3_maxdepth8,'XGBoost_eta0.3_maxdepth8.rds')

#Third Try: eta=0.3, gamma=0, max_depth=10
XGB_eta0.3_maxdepth10<-xgboost(data = data.matrix(train[,-13]),
                              label = y,
                              nrounds = 25,
                              eval_metric = "merror",
                              num_class=2,
                              max_depth=10)

y_pred <- predict(XGB_eta0.3_maxdepth10, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_eta0.3_maxdepth10,'XGBoost_eta0.3_maxdepth10.rds')

#Forth Try: eta=0.5, gamma=0, max_depth=6
XGB_eta0.5_maxdepth6<-xgboost(data = data.matrix(train[,-13]),
                               label = y,
                               nrounds = 25,
                               eval_metric = "merror",
                               num_class=2,
                               eta=0.5,
                               max_depth=6)

y_pred <- predict(XGB_eta0.5_maxdepth6, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_eta0.5_maxdepth6,'XGBoost_eta0.5_maxdepth6.rds')

#Fifth Try: eta=0.5, gamma=0, max_depth=8
XGB_eta0.5_maxdepth8<-xgboost(data = data.matrix(train[,-13]),
                              label = y,
                              nrounds = 25,
                              eval_metric = "merror",
                              num_class=2,
                              eta=0.5,
                              max_depth=8)

y_pred <- predict(XGB_eta0.5_maxdepth8, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_eta0.5_maxdepth8,'XGBoost_eta0.5_maxdepth8.rds')

#Sixth Try: eta=0.5, gamma=0, max_depth=10
XGB_eta0.5_maxdepth10<-xgboost(data = data.matrix(train[,-13]),
                               label = y,
                               nrounds = 25,
                               eval_metric = "merror",
                               num_class=2,
                               eta=0.5,
                               max_depth=10)

y_pred <- predict(XGB_eta0.5_maxdepth10, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_eta0.5_maxdepth10,'XGBoost_eta0.5_maxdepth10.rds')

#Seventh Try: eta=0.1, gamma=0, max_depth=6
XGB_eta0.1_maxdepth6<-xgboost(data = data.matrix(train[,-13]),
                              label = y,
                              nrounds = 25,
                              eval_metric = "merror",
                              num_class=2,
                              eta=0.1,
                              max_depth=6)

y_pred <- predict(XGB_eta0.1_maxdepth6, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_eta0.1_maxdepth6,'XGBoost_eta0.1_maxdepth6.rds')

#Eigth Try: eta=0.1, gamma=0, max_depth=8
XGB_eta0.1_maxdepth8<-xgboost(data = data.matrix(train[,-13]),
                              label = y,
                              nrounds = 25,
                              eval_metric = "merror",
                              num_class=2,
                              eta=0.1,
                              max_depth=8)

y_pred <- predict(XGB_eta0.1_maxdepth8, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_eta0.1_maxdepth8,'XGBoost_eta0.1_maxdepth8.rds')

#Ninth Try: eta=0.1, gamma=0, max_depth=10
XGB_eta0.1_maxdepth10<-xgboost(data = data.matrix(train[,-13]),
                               label = y,
                               nrounds = 25,
                               eval_metric = "merror",
                               num_class=2,
                               eta=0.1,
                               max_depth=10)

y_pred <- predict(XGB_eta0.1_maxdepth10, data.matrix(val[,-13]))
confusionMatrix(as.factor(y_pred),val$is_fraud,mode = 'prec_recall',positive = '1')
saveRDS(XGB_eta0.1_maxdepth10,'XGBoost_eta0.1_maxdepth10_final.rds')
#Eventually we choose the ninth model.


