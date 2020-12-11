# Title     : TODO
# Objective : TODO
# Created by: coultonf
# Created on: 2020-12-10

glm.importance = function (X.train,Y.train){
  logit.cv = cv.glmnet(X.train, Y.train, family="multinomial")
  plot(logit.cv)
  lambda.min = logit.cv$lambda.min
  lambda.1se = logit.cv$lambda.1se
  cmin = coef(logit.cv, s = lambda.min)
  cs1e = coef(logit.cv, s = lambda.1se)
  print(cmin)
}