library(caret)
library(FNN)
library(e1071)
library(forecast)
library(neuralnet)
library(dummies)
library(ROSE)
library(rpart)
library(rpart.plot)
library(randomForest)
library(adabag)

original.df <- read.csv("Loan payments data EDIT.csv")
df <- original.df[c(3,4,6,13,15,16)]


df$loan_status_D <- as.factor(df$loan_status_D)
df$Gender <- as.factor(df$Gender)

test.df <- upSample(df, df$loan_status_D)
df <- test.df[-c(7)]

df$Gender <- ifelse(df$Gender=="male",1,0)

#Training and validating data
train.index <- sample(row.names(df), .6*dim(df)[1])
valid.index <- setdiff(row.names(df), train.index)
train.df <- df[train.index, ]
valid.df <- df[valid.index, ]

#Normalizing data
train.norm.df <- train.df
valid.norm.df <- valid.df
norm.df <- df

norm.values <- preProcess(train.df[, 2:3], method=c("center", "scale"))
train.norm.df[, 2:3] <- predict(norm.values, train.df[, 2:3])
valid.norm.df[, 2:3] <- predict(norm.values, valid.df[, 2:3])
norm.df[, 2:3] <- predict(norm.values, df[, 2:3])


#Neural Net
set.seed(1)
nn <- neuralnet(loan_status_D ~ Principal + terms +  Gender + education_D +
                  age_D, data = train.norm.df, linear.output=FALSE, hidden = 5)
nn$weights
plot(nn)

valid.prediction <- compute(nn, valid.norm.df)
valid.class <- ifelse(apply(valid.prediction$net.result,1,which.max)==1,"0",
                      ifelse(apply(valid.prediction$net.result,1,which.max)==2,"1","2"))
confusionMatrix(as.factor(valid.class), as.factor(df[valid.index,]$loan_status_D))

nn2 <- neuralnet(loan_status_D ~ Principal + terms +  Gender + education_D +
                  age_D, data = train.norm.df, linear.output=FALSE, hidden = c(2,3))
nn2$weights
plot(nn2)

valid.prediction2 <- compute(nn2, valid.norm.df)
valid.class2 <- ifelse(apply(valid.prediction2$net.result,1,which.max)==1,"0",
                      ifelse(apply(valid.prediction2$net.result,1,which.max)==2,"1","2"))
confusionMatrix(as.factor(valid.class2), as.factor(df[valid.index,]$loan_status_D))

#Tree
default.ct <- rpart(loan_status_D ~ ., data = train.df, method = "class")
prp(default.ct)
pred_default <- predict(default.ct,newdata=valid.df,type = "class")
confusionMatrix(pred_default, valid.df$loan_status_D)

rf <- randomForest(loan_status_D ~ ., data = train.df, ntree = 500)

rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, valid.df$loan_status_D)

#Bayes
df2 <- original.df[c(3,5,7,13,15,16)]

df2$loan_status_D <- as.factor(df2$loan_status_D)
df2$Gender <- as.factor(df2$Gender)

test.df2 <- upSample(df2, df2$loan_status_D)
df2 <- test.df[-c(7)]

df2$Gender <- ifelse(df2$Gender=="male",1,0)

#Training and validating data
train.index2 <- sample(row.names(df2), .6*dim(df2)[1])
valid.index2 <- setdiff(row.names(df2), train.index2)
train.df2 <- df[train.index2, ]
valid.df2 <- df[valid.index2, ]



loan.nb <- naiveBayes(loan_status_D ~ ., data = train.df2)
loan.nb

pred.prob <- predict(loan.nb, newdata = valid.df2, type = "raw")
pred_valid.df<-cbind(valid.df2,pred.prob)

pred.class <- predict(loan.nb, newdata = valid.df2)
pred_valid.df<-cbind(pred_valid.df,pred.class)

pred.class <- predict(loan.nb, newdata = valid.df2)
confusionMatrix(pred.class, valid.df2$loan_status_D)

#boosting
boost <- boosting(loan_status_D ~ ., data = train.df)
pred_bst <- predict(boost, valid.df, type = "class")
confusionMatrix(as.factor(pred_bst$class), valid.df$loan_status_D)

prp(boost)






