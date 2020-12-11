# Title     : TODO
# Objective : TODO
# Created by: coultonf
# Created on: 2020-12-09
#Libraries for...
#Linear Classifier
library(FNN)
#Variable selection
library(dplyr)
library(leaps)
# log reg
library(glmnet)
# multinomial log linear model
library(nnet)
#svm
library(e1071)
#random forest
library(randomForest)
#lda
library(MASS)
#rpart
library(rpart)
#gbm
library(gbm)

knn.classifier = function(X.train, X.valid, Y.train, Y.valid){
  K.max = 40 # Maximum number of neighbours

  ### Container to store CV misclassification rates
  mis.CV = rep(0, times = K.max)

  for(i in 1:K.max){
    ### Fit leave-one-out CV
    this.knn = knn.cv(X.train, Y.train, k=i)

    ### Get and store CV misclassification rate
    this.mis.CV = mean(this.knn != Y.train)
    mis.CV[i] = this.mis.CV
  }
  k.min = which.min(mis.CV)
  SE.mis.CV = sapply(mis.CV, function(r){
    sqrt(r*(1-r)/nrow(X.train))
  })
  thresh = mis.CV[k.min] + SE.mis.CV[k.min]
  k.1se = max(which(mis.CV <= thresh))
  knn.1se = knn(X.train, X.valid, Y.train, k.1se)
  return (misclassify.rate(knn.1se, Y.valid))
}

glm.classifier = function(X.train, X.valid, Y.train, Y.valid){
  logit.fit <- cv.glmnet(x=as.matrix(X.train),
                    y=Y.train, family="multinomial")
  lambda.min = logit.fit$lambda.min
  lambda.1se = logit.fit$lambda.1se
  Y.hat <- predict(logit.fit, type="class",
                             s=logit.fit$lambda.1se,
                             newx=as.matrix(X.valid))
  return(misclassify.rate(Y.valid, Y.hat))

}
glm.min.classifier = function(X.train, X.valid, Y.train, Y.valid){
  logit.fit <- cv.glmnet(x=as.matrix(X.train),
                    y=Y.train, family="multinomial")
  lambda.min = logit.fit$lambda.min
  lambda.1se = logit.fit$lambda.1se
  Y.hat <- predict(logit.fit, type="class",
                             s=logit.fit$lambda.min,
                             newx=as.matrix(X.valid))
  return(misclassify.rate(Y.valid, Y.hat))

}

lda.classifier = function(X.train, X.valid, Y.train, Y.valid){
  fit.lda = lda(X.train, Y.train)
  pred.class <- predict(fit.lda,X.valid)$class
  return(misclassify.rate(Y.valid, pred.class))
}

qda.classifier = function(X.train, X.valid, Y.train, Y.valid){
  fit.lda = qda(X.train, Y.train)
  pred.class <- predict(fit.lda,X.valid)$class
  return(misclassify.rate(Y.valid, pred.class))
}

mll.classifier = function(X.train, X.valid, Y.train, Y.valid){
  fit.log.nnet = multinom(Y.train ~ ., data = cbind(X.train, Y.train))
  Y.hat = predict(fit.log.nnet, newdata=X.valid, type="class")
  table(Y.hat, Y.valid, dnn = c("<MLL> Predicted", "Observed"))
  return(misclassify.rate(Y.valid, Y.hat))

}
rf.classifier = function (X.train, X.valid, Y.train, Y.valid){
  all.mtry = c(2,4,8,12,16,20)
  all.pars = expand.grid(mtry = all.mtry)
  n.pars = nrow(all.pars)
  CV.ERR = array(0, dim = n.pars)
  for(j in 1:n.pars){
    ### Get current parameter values
    this.mtry = all.pars[j,]

    this.rf = randomForest(data=cbind(X.train, Y.train), Y.train~., mtry=this.mtry, nodesize=1,
                      importance=TRUE, keep.forest=TRUE, ntree=1000)
    Y.hat = predict(this.rf, X.train)

    CV.ERR[j] = misclassify.rate(Y.train, Y.hat)
  }
  optimal = all.pars[which.min(CV.ERR),]
  optimal.rf = randomForest(data=cbind(X.train, Y.train), Y.train~., mtry=optimal, nodesize=1,
                      importance=TRUE, keep.forest=TRUE, ntree=500)
  Y.hat = predict(optimal.rf, newdata = X.valid)
  return(misclassify.rate(Y.valid, Y.hat))
}

regtree.classifier = function(X.train, X.valid, Y.train, Y.valid){
  reg.tree = rpart(data=cbind(Y.train, X.train), method="class", Y.train ~ ., cp=0)
  cpt = reg.tree$cptable
  minrow <- which.min(cpt[,4])
  # Take geometric mean of cp values at min error and one step up
  cplow.min <- cpt[minrow,1]
  cpup.min <- ifelse(minrow==1, yes=1, no=cpt[minrow-1,1])
  cp.min <- sqrt(cplow.min*cpup.min)

  # Find smallest row where error is below +1SE
  se.row <- min(which(cpt[,4] < cpt[minrow,4]+cpt[minrow,5]))
  # Take geometric mean of cp values at min error and one step up
  cplow.1se <- cpt[se.row,1]
  cpup.1se <- ifelse(se.row==1, yes=1, no=cpt[se.row-1,1])
  cp.1se <- sqrt(cplow.1se*cpup.1se)
  reg.tree.cv.1se <- prune(reg.tree, cp=cp.1se)
  Y.hat <- predict(reg.tree.cv.1se, newdata=X.valid, type="class")
  return(misclassify.rate(Y.valid, Y.hat))

}
lognet.classifier = function(X.train, X.valid, Y.train, Y.valid){
  Y.train.num = class.ind(Y.train)
  MSE.best = Inf    ### Initialize sMSE to largest possible value (infinity)
  M = 20            ### Number of times to refit.

  for(i in 1:M){
    ### For convenience, we stop nnet() from printing information about
    ### the fitting process by setting trace = F.
    this.nnet = nnet(X.train, Y.train.num, size = 6, decay = 0.1, maxit = 2000,
      softmax = T, trace = F)
    this.MSE = this.nnet$value
    if(this.MSE < MSE.best){
      NNet.best.0 = this.nnet
      MSE.best = this.MSE
    }
  }
  ### Now we can evaluate the validation-set performance of our naive neural
  ### network. We can get the predicted class labels using the predict()
  ### function and setting type to "class"
  Y.hat = predict(NNet.best.0, X.valid, type = "class")
  return(misclassify.rate(Y.valid, Y.hat))
}

svm.classifier = function (X.train, X.valid, Y.train, Y.valid){
  all.cost = c(1,10,20,50,100)
  all.gamma = c(0.001,0.01,0.1,1,10)
  all.pars = expand.grid(cost = all.cost, gamma = all.gamma)
  n.pars = nrow(all.pars)
  CV.ERR = array(0, dim = n.pars)
  for(j in 1:n.pars){
    ### Get current parameter values
    this.cost = all.pars[j,1]
    this.gamma = all.pars[j,2]

    this.svm = svm(Y.train ~ ., data = cbind(Y.train, X.train), kernel = "radial", cost = this.cost, gamma = this.gamma)
    Y.hat = predict(this.svm, X.train)

    CV.ERR[j] = misclassify.rate(Y.train, Y.hat)
  }
  optimal = all.pars[which.min(CV.ERR),]
  optimal.svm = svm(Y.train ~ ., data = cbind(Y.train, X.train), kernel = "radial",
                    cost = optimal$cost, gamma = optimal$gamma)
  Y.hat = predict(optimal.svm, X.valid)
  return(misclassify.rate(Y.valid, Y.hat))
}
