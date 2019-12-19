rm(list=ls(all=T))

# setting the working directory 
setwd("C:/Users/Yashwanth/Desktop/Project 2")
# check the current working directory 
getwd()

# load required libraries 
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','fastDummies', 'psych')

#install.packages(x)
lapply(x, require, character.only = TRUE)

library("dplyr")
library("plyr")
library("ggplot2")
library("data.table")
library("GGally")

#load data into r 
bank = read.csv("bank-loan.csv", header = T, na.strings = c(" ","","NA"))
# Summarizing  data 
# dim helps to see no of observations and variables in the dataset.
# dataset contains 850 obs. of 9 variables 
dim(bank)

### TAble helps us to see how many for default and non default
table(bank$default)

#### unique values #########
unique(bank$default)

# getting first 5 rows of the datasets 
head(bank, 5)

tail(bank, 5)

# getting the column names of the dataset.
colnames(bank)

# str of the data 
str(bank)

bank$default = as.factor(bank$default)

str(bank)

describe(bank)

############## Checking the distribution of the age ### 
#### age is normally distributed.
plot(density(bank$age))
hist(bank$age, main = " age histogram" , xlab = 'age', ylab = "freq")

#####

ggplot(bank) +
  geom_bar(aes(x=ed),fill="grey")

################ missing value analysis ###################

sum(is.na(bank$default))
missing_val = data.frame(apply(bank, 2, function(x){sum(is.na(x))}))

missing_val$columns = row.names(missing_val)

row.names(missing_val) = NULL

names(missing_val)[1] = "missing_percentage"

missing_val$missing_percentage = (missing_val$missing_percentage/nrow(bank))*100.

missing_val = missing_val[order(-missing_val$missing_percentage),]

missing_val = missing_val[,c(2,1)]

##################### outlier chech #####################
##### All the given variables has outliers. I assume that the income and related 
# debt values are dependent on the observer. more the education more the values
### might be. 

boxplot(bank$age, main = " outlier check @ age", ylab = " age", col = 5)

boxplot(bank$income, main = " outlier check @ income", ylab = "income", col = 5)

boxplot(bank$creddebt, main = " outlier check @ creddebt", ylab = "creddebt", col = 5)

boxplot(bank$othdebt, main = " outlier check @ othdebt", ylab = "othdebt", col = 5)

### or 
ggplot(data = bank, aes(x = "", y = age)) + 
  geom_boxplot() 

ggplot(data = bank, aes(x = "", y = income)) + 
  geom_boxplot() 

ggplot(data = bank, aes(x = "", y = creddebt)) + 
  geom_boxplot() 

ggplot(data = bank, aes(x = "", y = othdebt)) + 
  geom_boxplot()


## Correlation Plot
numeric_index = sapply(bank, is.numeric)
corrgram(bank[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

###### standardisation###########

cnames = colnames(bank)
for (i in cnames) {
  print(i)
  bank[,i] = (bank[,i]-mean(bank[,i]))/sd(bank[,i])
  
}

str(bank)
###################### separating the labeled and not labeled observations. 
##### last 150 observations are not labled. Those observations to be 
### predicted after the model building. we will predict those values after 
### chossing best model. 

bank_train = bank[1:700,1:9]
dim(bank_train)
sum(is.na(bank$default))
bank_test  = bank[701:850,1:8]
dim(bank_test)

############# Model buildig##############
library(caTools)

#Splitting into training and testing data
set.seed(123)
sample = sample.split(bank_train, SplitRatio = 0.8)
sample
training = subset(bank_train, sample==TRUE)
str(training)
testing = subset(bank_train, sample==FALSE)
str(testing)


######################logistic regression #########################

model = glm(default~.,training, family = "binomial")

summary(model)

model1 = glm(default~age,training, family = "binomial")

summary(model1)

model2 = glm(default~ed,training, family = "binomial")

summary(model2)

model3 = glm(default~employ,training, family = "binomial")

summary(model3)

model4 = glm(default~address,training, family = "binomial")

summary(model4)

model5 = glm(default~creddebt,training, family = "binomial")

summary(model5)

#########################model with high important variables############

model6 = glm(default~creddebt+debtinc+address+employ,training, family = "binomial")

summary(model6)

res = predict(model6, testing, type = "response")

range(res)

confusion_matric = table(Actualvalue=testing$default, predictedvalue=res>0.5)

print(confusion_matric)

accuracy = (104+20)/(104+20+24+7)

print(accuracy)

# precision = 0.74
# recall = 0.45

####### threshold evaluation ################
### ROC CURVE ##########
######AUC####
library(ROCR)
pred_log = prediction(res,testing$default)
acc = performance(pred_log,"acc")
plot(acc)

roc_curve = performance(pred_log, "tpr" , "fpr")
plot(roc_curve)
plot(roc_curve , colorize = T, print.cutoffs.at=seq(0.1,by=0.1))

###### using threshold value of 0.4 we can incraese the true positive rate 

confusion_matric = table(Actualvalue=testing$default, predictedvalue=res>0.4)

print(confusion_matric)

accuracy = (107+18)/(107+18+4+26)

print(accuracy)

# accuracy = 0.80
# precision = 0.58
# recall =  0.56

############ AUC##############

auc = performance(pred_log, "auc")
auc

# AUC = 0.82

#############Precision recall curve ##############
library(PRROC)
PRC_curve = performance(pred_log, "prec" , "rec")
plot(PRC_curve, colorize = T)

############################## DEcision tree###############
library(tree)
deci_model = tree(default~., data = training)
summary(deci_model)

### plotting 
plot(deci_model)
text(deci_model,pretty = 0)

#### prediction 

deci_pred = predict(deci_model, testing, type = "class")

confusion_matric = table(Actualvalue=testing$default, predictedvalue=deci_pred)

print(confusion_matric)

#### cross validation 
cv.deci_model = cv.tree(deci_model, FUN = prune.misclass)
cv.deci_model
plot(cv.deci_model)

####pruning
prune.deci_model = prune.misclass(deci_model, best = 10)
plot(prune.deci_model)
text(prune.deci_model)

#### prediction of values again
deci_pred_1 = predict(prune.deci_model, testing, type = "class")

Confusion_matrix_1= table(testing$default, deci_pred_1)

print(Confusion_matrix_1)
# accuracy = 0.74 # precision = 0.54 # recall = 0.43

##############################Random Forest#################

#####random forest 1

library(randomForest)
rf = randomForest(default~., data = training)
print(rf)

## prediction 

rf_pred = predict(rf,testing)

confusion_matric = table(Actualvalue=testing$default, predictedvalue=rf_pred)

print(confusion_matric)

## tune mtry

tuneRF(training[,-9], training[,9],stepfactor = 0.5, 
        plot = TRUE , ntreeTry = 1000, 
        trace = TRUE , 
        improve = 0.05)

rf1 = randomForest(default~.,data = training, ntree = 1000, mtry = 2)

rf1

# predict 

rf_pred1 = predict(rf1,testing)
confusion_matric1 = table(Actualvalue=testing$default, predictedvalue=rf_pred1)

print(confusion_matric1)

# no. of nodes for the trees
hist(treesize(rf1),main = " no. of nodes for the trees", col = "green")

# variable importance
varImpPlot(rf1,
           sort = T,
           main = "variable importance")

importance(rf1)
varUsed(rf1)

############## we will build random forest by taking only max meandecreaseGini
### considering debtinc, employ, creddebt, othdeb, income.
### build model 
rf_final = randomForest(default~debtinc+employ+creddebt+othdebt+income ,
                        data = training,
                        ntree = 1000, mtry = 2)
rf_final

# prediction 
rf_pred_final = predict(rf_final,testing)

confusion_matric_f = table(Actualvalue=testing$default, predictedvalue=rf_pred_final)

print(confusion_matric_f)

# accuracy = 0.735
# precision = 0.55
# recall = 0.340

#### we can not decide the perfomance of model only based on the accuracy
# we need to have a good trade off between precision and recall.
# logistic model has 80.0 % accuracy with good trade-off b/t prec and recall.

##### Conclusion ======= logistic model is the best suited model on this dataset. 

#### predicting for the test data. 
res = predict(model6, bank_test, type = "response")
range(res)
