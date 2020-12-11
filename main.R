# Title     : Cellphones data classification
# Objective :
# Created by: coultonf
# Created on: 2020-12-09

source("utils.R")
source("models.R")
source("importance.R")

# data prep
data = na.omit(read.csv("datasets/train.csv"))
data$Y = factor(data$price_range, labels=c("Low", "Med", "High", "Exp"))
data = subset(data, select = -price_range)

set.seed (8646824, kind="Mersenne-Twister")
perm = sample ( x = nrow ( data ))
trainval = data [ which ( perm <= 3* nrow ( data )/4) , ]
test = data [ which ( perm > 3* nrow ( data )/4) , ]
Y = trainval[21]
X = trainval[-21]
trainval.scaled = cbind(rescale(X, X), Y)

Y = test[21]
X = test[-21]
test.scaled = cbind(rescale(X, X), Y)

glm.importance(as.matrix(trainval.scaled[-21]), as.matrix(trainval.scaled[21]))
plot.importance()
K=10
n=nrow(trainval)
folds = get.folds(n, K)

all.models = c("LDA", "MLL", "NN", "GLM", "GLM.min", "SVM")
all.misclassify.rate = array(0, dim = c(K,length(all.models)))
colnames(all.misclassify.rate) = all.models
for(i in 1:K){
  print(paste0(i, " of ", K))
  X.train = trainval[folds != i,-21]
  X.valid = trainval[folds == i,-21]
  Y.train = trainval[folds != i,21]
  Y.valid = trainval[folds == i,21]

  X.train.scaled = rescale(X.train, X.train)
  X.valid.scaled = rescale(X.valid, X.train)

  # all.misclassify.rate[i, "KNN"] = knn.classifier(X.train, X.valid, Y.train, Y.valid)
  # all.misclassify.rate[i, "RF"] = rf.classifier(X.train, X.valid, Y.train, Y.valid)
  all.misclassify.rate[i, "SVM"] = svm.classifier(X.train, X.valid, Y.train, Y.valid)
  all.misclassify.rate[i, "LDA"] = lda.classifier(X.train, X.valid, Y.train, Y.valid)
  # all.misclassify.rate[i, "QDA"] = qda.classifier(X.train, X.valid, Y.train, Y.valid)
  all.misclassify.rate[i, "MLL"] = mll.classifier(X.train, X.valid, Y.train, Y.valid)
  # all.misclassify.rate[i, "REG"] = regtree.classifier(X.train, X.valid, Y.train, Y.valid)
  all.misclassify.rate[i, "GLM"] = glm.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)
  all.misclassify.rate[i, "GLM.min"] = glm.min.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)
  all.misclassify.rate[i, "NN"] = lognet.classifier(X.train, X.valid, Y.train, Y.valid)

}
par(mfrow=c(1,1))
boxplot(all.misclassify.rate, main = paste0("CV Misclassifiers over ", K, " folds"))