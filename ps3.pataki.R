##PS3

Q1

When lamda approaches infinity the penalty term for roughness becomes so large that the function g(x) 
is forced to be smooth everywhere relative to the order of the derivative contained in the penalty term. 
It’s helpful for me to think about this from the simplest case and then work our way up. Let m = the 
order of derivative contined in the penalty term and use the Boston data set from the ISLR libary as a visual aid.

options(digits=3)
library(MASS)
library(graphics)
library(boot)
library(splines)
library(pls)
library(leaps)
boston=Boston
log.crim=log(boston$crim)
medv=boston$medv

#When lamda approaches infinity for:

#m=0 then the 0th derivative is forced to zero, i.e. g(x)=0 so we have no degrees of freedom and can’t fit to 
#the data at all. 

#m=1 then the 1st derivative is forced to zero, i.e. the slope of the function will be zero. We are therefore free 
#to choose some intercept that will minimize the loss function but we have no latitude over the slope. So the fitted 
#form is just a horizontal line where g(x) = y_bar. 

plot(log.crim, medv, ylim = c(0, max(medv))) 
abline(h=mean(medv), lwd=2, col="blue")

#m=2 then the 2nd derivative is forced to zero, i.e. the slope is constant but nonzero. g(x) is a one-degree polynomial 
#that passes through the data and minimizes the residual sum of squares (least squares solution)

plot(log.crim, medv, ylim = c(0, max(medv))) 
abline(lm(medv~log.crim,data=boston), lwd=2, col="blue") 

#m=3 then the 3rd derivative is forced to zero. The solution will be least squares on a quadratic form, i.e. a 
#second-degree polynomial.

fit = lm(medv~poly(log.crim,2),data=Boston)
log.crim.lims=range(log.crim)
log.crim.grid = seq(from=log.crim.lims[1], to=log.crim.lims[2])
preds = predict(fit,newdata=list(log.crim=log.crim.grid),se=TRUE)
se.bands=cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)
par(mfrow=c(1,1),mar=c(4.5,4.5,1,1) ,oma=c(0,0,4,0))
plot(log.crim,medv,xlim=log.crim.lims ,cex=.5,col="darkgrey")
lines(log.crim.grid,preds$fit,lwd=2,col="blue")
matlines(log.crim.grid,se.bands,lwd=1,col="blue",lty=3)

#m=4 then the 4th derivative is forced to zero, The solution will be least squares on a cubic form, i.e. a 
#third degree polynomial.

fit = lm(medv~poly(log.crim,3),data=Boston)
log.crim.lims=range(log.crim)
log.crim.grid = seq(from=log.crim.lims[1], to=log.crim.lims[2])
preds = predict(fit,newdata=list(log.crim=log.crim.grid),se=TRUE)
se.bands=cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)
par(mfrow=c(1,1),mar=c(4.5,4.5,1,1) ,oma=c(0,0,4,0))
plot(log.crim,medv,xlim=log.crim.lims ,cex=.5,col="darkgrey")
lines(log.crim.grid,preds$fit,lwd=2,col="blue")
matlines(log.crim.grid,se.bands,lwd=1,col="blue",lty=3)

#Given this, for g1_hat (m=3) g2_hat (m=4):
        
#a. g2_hat is more flexible than g1_hat. It can more closely follow the training observations and therefore 
#g2_hat will have smaller training RSS. It reduces bias but of course with a price paid in variance.

#b. g2_hat will be prone to overfitting and higher variance given its flexibility and so we’d expect the less 
#flexible model, g1_hat, to have lower test RSS.

#c. When lamda = 0 there is no penalty term and g1_hat = g2_hat therefore we’d expect them to have the same 
#training and test RSS if performed on the same set of data.


#Question 2

#For each of the three approaches we are considering a sequence of models with k=0 up through k=p predictors. 

#a. Without more information is impossible to answer this question definitively. As I explain below best subset is guaranteed to return the p+1 models with the lowest training RSS for each level of k=0 through k=p predictors however training RSS is a notoriously poor estimator of test RSS. It’s entirely possible that the best subset returns a model which fits the test data well, while forward and backwards stepwise selection return a slightly different set of p+1 models and the one chosen from among those perform poorly on the test data. The opposite is entirely possible as well. 

#b.  A full model, that is where k=p, will have the lowest training RSS and this will be the same for all three approaches. However if we arbitrarily choose a level of k and compare across methods then forward and backwards stepwise selection are not guaranteed to have the lowest training RSS. In order to reduce the computational price of best subset, forward stepwise selection keeps the predictor which gives the greatest additional improvement to the fit of the model at each level of k while backwards stepwise selection iteratively removes the elast useful predictor. They are therefore not guaranteed to find the best possible model out of all the 2^p models containing subsests of the p predictors.  

#For example, if the best model where k=1 contains X1 but the best model where k=3 contains X2 and X3, forward selection woudl miss the best model at k=3 since it would retain X1 as it worked its way through each level of k. We could construct a similar example for backwards stepwise selection. Best subset considers (p choose k) models at each level of k and is exhaustive so it is guaranteed to find the model with lowest training RSS for every approach with k predictors.

#c. 

#i. False. Let {X1,X2, X3, X4}  be our predictors and thus p=4. Let the best model chosen by forward stepwise when k=2 be {X1 & X2}. It is entirely possible that when backwards stepwise moves from the full model to one where it considers k+1=3 predictors that it eliminates either X1 or X2 from the model. In this case the predictors of the 2-variable model chosen by forward stepwise will not be a subset of the predictors of the 3-variable model chosen by backwards stepwise selection. The T/F statement does not hold true for all possible cases. 

#ii. False. Let {X1,X2, X3, X4, X5}  be our predictors and thus p=5. If the best model chosen by forward stepwise when k+1=3 is {X3, X4, X5} but the best model chosen by backwards stepwise when k=2 is {X1,X2} then the predictors identified by backwards stepwise are not a subset of the predictors identified by forward stepwise. The T/F statement does not hold true for all possible cases. 

#iii. False. Let {X1,X2, X3, X4, X5} be our predictors and thus p=5. It’s possible that when k=2 best subset identifies {X1, X2} as the best model but when k=3 it identifies {X3,X4,X5} as the best model. This could be due to correlation among variables and is a result of best subset considering all (p choose k) models at every level of k=0 up through k=p. The T/F statement does not hold true for all possible cases. 

#iv. True. Backwards stepwise iteratively removes the predictor which is the least useful at each level of k, starting where k=p and decrementing k by 1 for each round through k=0. Each set of predictors in the (k-1)-variable model will therefore be a subset of the predictors at the k-variable model. 

#v.  True. Forwards stepwise iteratively adds the predictor which is the most useful at each level of k, starting where k=0 and incrementing k by 1 for each round through k=p. Each set of predictors in the k-variable model will therefore be a subset of the predictors at the (k+1)-variable model. 


#Question 3



#dis = the weighted mean of distances to five Boston employment centers (predictor)
#nox = nitrogen oxides concentration in parts per ten million (response)

#check to make sure no observations are missing 
sum(is.na(dis))
sum(is.na(nox))

#a

fit = lm(nox~ poly(dis,3),data=Boston)
summary(fit)


dis.lims=range(dis)
dis.grid = seq(from=dis.lims[1], to=dis.lims[2])
preds = predict(fit,newdata=list(dis=dis.grid),se=TRUE)
se.bands=cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)
par(mfrow=c(1,1),mar=c(4.5,4.5,1,1) ,oma=c(0,0,4,0))
plot(dis,nox,xlim=dis.lims ,cex=.5,col="darkgrey")
title("Degree -3 Polynomial ",outer=T)
lines(dis.grid,preds$fit,lwd=2,col="blue")
matlines(dis.grid,se.bands,lwd=1,col="blue",lty=3)

#b

library(graphics)
dis.lims=range(dis)
dis.grid = seq(from=dis.lims[1], to=dis.lims[2])
par(mfrow=c(1,1),mar=c(4.5,4.5,1,1) ,oma=c(0,0,4,0))
plot(dis,nox,xlim=dis.lims ,cex=.5,col="darkgrey")
title("i-th Degree Polynomial ",outer=T)
names = seq(1:10)
col = rainbow(10)
residuals = c()
for (i in 1:10){
        fit = lm(nox~ poly(dis,i),data=Boston)
        preds = predict(fit,newdata=list(dis=dis.grid),se=TRUE)
        residuals = c(residuals,deviance(fit))
        lines(dis.grid,preds$fit,lwd=2,col=col[i])
}
legend("topright",legend=names,lty=c(1,1), col=col)

#report residual sum of squares for polynomial fits
residuals

#c
library(boot)
set.seed(17)
names=seq(1:10)
cv.error=rep(0,10)
for (i in 1:10) {
        glm.fit=glm(nox~poly(dis,i,raw=TRUE),data=Boston)
        cv.error[i]=cv.glm(Boston,glm.fit,K=10)$delta[1]
}
plot(cv.error, type="o")

#d
#allow df within bs() to produce uniform quantiles for knots
library(splines)
fit=lm(nox~bs(dis,df=4),data=Boston)
pred=predict(fit,newdata=list(dis=dis.grid),se=T)
plot(dis,nox,col="gray")
lines(dis.grid,pred$fit,lwd=2)
lines(dis.grid,pred$fit+2*pred$se ,lty="dashed")
lines(dis.grid,pred$fit-2*pred$se ,lty="dashed")

#why only one knot - bs() defaults to uniform quantiles and since there's only
#four degrees of freedom we have one know and R chooses the median
attr(bs(dis,df=4),"knots")


#e
dis.lims=range(dis)
dis.grid = seq(from=dis.lims[1], to=dis.lims[2])
par(mfrow=c(1,1),mar=c(4.5,4.5,1,1) ,oma=c(0,0,4,0))
plot(dis,nox,xlim=dis.lims ,cex=.5,col="darkgrey")
title("i-th Degrees of Freedom ",outer=T)
names = c(3,4,5,6,7,8,9,10)
col = rainbow(8)
residuals = c()
for (i in 3:10){
        fit = lm(nox~bs(dis,df=i),data=Boston)
        preds = predict(fit,newdata=list(dis=dis.grid),se=TRUE)
        residuals = c(residuals,deviance(fit))
        lines(dis.grid,preds$fit,lwd=2,col=col[i])
}
legend("topright",legend=names,lty=c(1,1), col=col)

#report residual sum of squares for degrees of freedom fits
residuals

plot(seq(from=3, to=10),residuals,type="o")
#as we increase the degrees of freedom we're allowing the model to interpolate
#the data more and more closely. the rss decreases monotonically but again, this
#is not a good estimator of test rss and so we should utilize cross-validation to 
#choose the appropriate degrees of freedom

#f
set.seed(21)
cv.error=rep(0,10)
for (i in 3:10) {
        fit = glm(nox~bs(dis,df=i),data=Boston)
        cv.error[i]=cv.glm(Boston,fit,K=10)$delta[1]
}
cv.error
plot(seq(from=3, to=10),cv.error[3:10], type="o",xlab="df",ylab="rss")

#the marginal difference between degrees of freedom is quite small. When we plot
#the data out we see that while ten degrees of freedom has the lowers test
#rss under cross-validation, choosing 6 degrees of freedom would result in a 
#simpler model with ony a minimal increase in the test MSE

set.seed(21)
cv.error=rep(0,50)
for (i in 3:50) {
        fit = glm(nox~bs(dis,df=i),data=Boston)
        cv.error[i]=cv.glm(Boston,fit,K=10)$delta[1]
}
cv.error
plot(seq(from=3, to=50),cv.error[3:50], type="o",cex=0.25,xlab="df",ylab="rss")


#Question 4


a. 

library(pls)
setwd("~/Downloads/")
load("body.Rdata")
plot(X$Bicep.Girth, Y$Gender, main="Bicep versus Gender", xlab="Bicep", ylab="Gender", pch=5)
plot(X$Thigh.Girth, Y$Gender, main="Thigh versus Gender", xlab="Thigh", ylab="Gender", pch=5)
plot(Y$Height, Y$Gender, main="Height versus Gender", xlab="Height", ylab="Gender", pch=5)

#looking at the documentation we see that men tend to be taller and so we can 
#assume that men are coded as 1 and women as 0 in the dataset.

b.

n = nrow(Y)
set.seed(100)
train = sample(n, 307) 
test = -train

#pcr
library(pls)
set.seed(100)
data = data.frame(Y$Weight, X)
pcr.fit=pcr(Y.Weight~., data=data, subset=train, scale=TRUE, validation="CV")
pls.fit=plsr(Y.Weight~., data=data, subset=train, scale=TRUE, validation="CV")

#Setting scale=TRUE has the effect of standardizing each predictor, using (6.6), 
#prior to generating the principal components, so that the scale on which each variable 
#is measured will not have an effect. (see p.217 as well)

##OH QUESTION

#PCR has the effect of dimensionr reduction and so it would help if we rescale the variables
#so that the model is more interpretable. 

c.

summary(pcr.fit)
summary(pls.fit)

num.components = seq(1:21)
pcr.var = cumsum(explvar(pcr.fit))
pls.var = cumsum(explvar(pls.fit))

plot(num.components, pcr.var,type="l", col="blue", main="PCR & PLSR: % of Variance Explained")
lines(pls.var,col="red")
legend("bottomright", legend=c("pcr.var", "plsr.var"), lty=c(1,1), col=c("blue", "red"))

#The summary() function also provides the percentage of variance explained in the 
#predictors and in the response using different numbers of compo- nents. This concept 
#is discussed in greater detail in Chapter 10. Briefly, we can think of this as the 
#amount of information about the predictors or the response that is captured using M 
#principal components. For example, setting M = 1 only captures 38.31 % of all the 
#variance, or information, in the predictors. In contrast, using M = 6 increases the 
#value to 88.63 %. If we were to use all M = p = 19 components, this would increase 
#to 100 %.

#1. they're both monotically increasing functions.
#2. PCR always outperforms PLSR.
#3. Actually this isn't all that surprising. PSLR is a supervised learning method so 
# it decreases bias at a price to variance. There is actually no absolute advantage to 
# to be gained from using either method exclusively due ot this bias-variance tradeoff.

d.

validationplot(pcr.fit, val.type="MSEP", main="Test MSE for PCR on Training Dataset", xaxt="n", ylim=c(0,100))
axis(1,at=0:21)
validationplot(pls.fit, val.type="MSEP", main="Test MSE for PLSR on Training Dataset", xaxt="n", ylim=c(0,100))
axis(1,at=0:21)

#The cross-validation error approaches its minimum for both PCR and PLSR at 3 principle 
#components but we'd like to use as simple a model as possible and the marginal improvement moving 
#from 2 to 3 components is minimal at best. I'd use two components moving forward.

e. 

#neither opf these methods perform variable selection. We could use best subset
#but we'd have to consider 2^21 models so we choose backwards stepwise regression since
#n is much larger than p in this case and we can start with the full model.

trainingX=data[train,]
testX=data[-train,]
regfit.bwd=regsubsets(Y.Weight~., data=trainingX, nvmax=21, method="backward")
regfit.fwd=regsubsets(Y.Weight~., data=trainingX, nvmax=21, method="forward")
regfit.bwd.summary=summary(regfit.bwd)
regfit.fwd.summary=summary(regfit.fwd)

par(mfrow=c(2,2))
#BIC
plot(regfit.bwd.summary$bic ,xlab="Number of Variables ",ylab="BIC", type="l",col="blue")
lines(regfit.fwd.summary$bic,col="red")
legend("topright", legend=c("bwd.step", "fwd.step"), lty=c(1,1), col=c("blue", "red"))
#AIC
plot(regfit.bwd.summary$cp ,xlab="Number of Variables ",ylab="Mallow's Cp", type="l",col="blue")
lines(regfit.fwd.summary$cp,col="red")
legend("topright", legend=c("bwd.step", "fwd.step"), lty=c(1,1), col=c("blue", "red"))
#ADJR2
plot(regfit.bwd.summary$adjr2 ,xlab="Number of Variables ",ylab="Adjusted R^2", type="l",col="blue")
lines(regfit.fwd.summary$adjr2,col="red")
legend("bottomright", legend=c("bwd.step", "fwd.step"), lty=c(1,1), col=c("blue", "red"))
#RSS
plot(regfit.bwd.summary$rsq ,xlab="Number of Variables ",ylab="r^2", type="l",col="blue")
lines(regfit.fwd.summary$rsq,col="red")
legend("bottomright", legend=c("bwd.step", "fwd.step"), lty=c(1,1), col=c("blue", "red"))

## ok. Neither backwards and forwards seem to significantly outperform oneanother. We
#choose not to use best subset because we'd rather not consider 2^21 models. We move forward
# with backward stepwise selection and use cross-validation to determine the number of predictors.

predict.regsubsets= function(object, newdata, id, ...) {
        form=as.formula(object$call[[2]])
        mat=model.matrix(form, newdata)
        coefi=coef(object, id=id)
        xvars=names(coefi)
        mat[,xvars] %*% coefi
}

k=10
num.var=21
set.seed(100)
folds=sample(1:k, nrow(trainingX), replace=TRUE)
folds <- sample(1:k, nrow(trainingX), replace = TRUE)
cv.errors = matrix(NA, k, 21, dimnames = list(NULL, paste(1:21)))

#for loop for cross validation

for(j in 1:k) {
        best.fit <-regsubsets(Y.Weight~., data=trainingX[folds != j,], nvmax=21, method="backward")
        for (i in 1:num.var) {
                pred = predict(best.fit, trainingX[folds==j,], id=i)
                cv.errors[j,i]=mean((trainingX$Y.Weight[folds==j] - pred)^2)
        }
}

#This has given us a 10×21 matrix, of which the (i, j)th element corresponds
#to the test MSE for the ith cross-validation fold for the best j-variable 
#model. We use the apply() function to average over the columns of this matrix
#in order to obtain a vector for which the jth element is the cross-validation 
#error for the j-variable model.

mean.cv.errors=apply(cv.errors ,2,mean)
par(mfrow=c(1,1))
plot(mean.cv.errors, type="b")
which.min(mean.cv.errors)
points(15,mean.cv.errors[15],col="red",cex=2,pch=20)
points(10,mean.cv.errors[15],col="blue",cex=2,pch=20)

# under cross-validation we see that the lowest cv error is the 15 variable model.
# The reduction in mean cv error moving from the 10-variable (coded in blue on the 
#plot above) to 15-variable model is minimal so the simpler model will suffice.

#now we rerun the 10-variable backward stepwise model on the training data
regfit.best <- regsubsets(Y.Weight ~ ., data = trainingX, nvmax = 10, method="backward")
coef(regfit.best, 10)

f. test models on test data

#10-variable model under backward stepwise regression 
pred.bwd.10 <- rep(0, 200)
for (i in 1:200) {
        pred.bwd.10[i] <- predict(regfit.best, testX[i, ], id = 10)
}
mean((pred.bwd.10 - testX$Y.Weight)^2)

#PCR

pred.pcr = predict(pcr.fit, newdata = data[test, ], ncomp = 2)
mean((pred.pcr - Y$Weight[test])^2)

#PLSR
pred.pls = predict(pls.fit, newdata = data[test, ], ncomp = 2)
mean((pred.pls - Y$Weight[test])^2)


