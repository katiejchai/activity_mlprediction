---
output:
  html_document:
    keep_md: yes
---

# Qualitative Activity Recognition of Weight Lifting Exercises, Machine Learning Predictions

The analysis detailed in this report is based on a study investigating the human activity recognition data of participants performing different executions of weightlifting, [Qualitative Activity Recognition of Weight Lifting Exercises](<http://web.archive.org/web/20170519033209/http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf>).

The goal of this report is to predict the qualitative class of weightlifting performed by a subject based on a variety of quantitative features, measured by accelerometers attached to different points of the subjects' bodies.

## Loading and Cleaning Data

Before anything, we will load all necessary libraries.


```r
library(tidyverse); library(caret); 
library(parallel); library(doParallel)
library(corrplot); library(rattle)
```

Now, load the given datasets with NA values set as blank observations or "NA"s.


```r
pmldf <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings=c("","NA"), stringsAsFactors=TRUE)
validation <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings=c("","NA"), stringsAsFactors=TRUE)
```

To clean the dataset, we will remove the variables with a majority of NA values (using a threshold of 80%), as well as any variables that have near zero variance, as these will add minimal, if any, information to the models. We will also remove the descriptive variables that do not affect the outcome (subject name, time, etc.)


```r
pmldf <- pmldf[,!(apply(is.na(pmldf), 2, mean) > 0.8)]
pmldf <- pmldf[,-nearZeroVar(pmldf)]
pmldf <- pmldf[,-c(1:6)]
```

The pmldf dataset will be split again so that we can test the accuracy of models developed on the training set onto the testing set, and choose the best model of these to form final predictions on the validation set. The seed has been set to ensure reproducibility of the results.


```r
set.seed(1234)
inTrain <- createDataPartition(y=pmldf$classe, p=0.6, list=FALSE)
training <- pmldf[inTrain,]
testing <- pmldf[-inTrain,]
```

We will also update the validation dataset to contain only the selected variables in the training set.


```r
validation <- validation %>% select(colnames(training)[-length(training)])
```

## Exploratory Data Analysis

As an initial exploratory analysis on the data, we will plot the correlations between the remaining predictors of the training dataset.


```r
corrplot(cor(training[,-length(training)]), 
         method="color", type="lower", order="alphabet", 
         tl.cex=0.5, tl.col="black", mar=c(0,0,1,0),
         title="Corrlation Plot of Training Dataset Predictors")
```

![](weightliftingprediction_files/figure-html/corrplot-1.png)<!-- -->

Based on this plot, it is evident that many of the predictors are strongly correlated with one another, based on the number of darkly colored squares in the plot above. Therefore, simple regression may not be the best of choices for modeling the variables' effects on classe.

## Model Building

To build the best model to predict the type of exercise based on these variables, four different methods will be used: decision tree, random forest, linear discriminant analysis, and gradient boosting.

We will use 5-fold cross-validation, along with parallel processing. A k-value of 5 was chosen in an attempt to balance between the less bias but more variance characteristic of large k-values, and the less variance but more bias characteristic of smaller k-values. The setup of parallel processing was included after referencing [Improving Performance of Random Forest with caret::train()](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md) by Len Greski.


```r
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)
fitControl <- trainControl(method="cv", number=5, allowParallel=TRUE)
```

### Decision Tree

A plot of the final decision tree is shown below.


```r
set.seed(55)
dtFit <- train(classe ~ ., data=training, method="rpart", 
               trControl=fitControl)
fancyRpartPlot(dtFit$finalModel, sub="")
```

![](weightliftingprediction_files/figure-html/decisiontree-1.png)<!-- -->

### Random Forest

A summary of the final model is shown below.


```r
set.seed(55)
rfFit <- train(classe ~ ., data=training, method="rf", 
               trControl=fitControl, verbose=FALSE)
rfFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = min(param$mtry, ncol(x)), verbose = FALSE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.83%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3344    4    0    0    0 0.001194743
## B   13 2255   11    0    0 0.010530935
## C    0   20 2032    2    0 0.010710808
## D    0    0   38 1889    3 0.021243523
## E    0    0    0    7 2158 0.003233256
```

### Linear Discriminant Analysis

No visualizations are available, but the respective code is shown below.


```r
set.seed(55)
ldFit <- train(classe ~ ., data=training, method="lda", 
               trControl=fitControl, verbose=FALSE)
```

### Gradient Boosting Machine Model

A plot of the model accuracy vs. number of boosting iterations is shown below.


```r
set.seed(55)
gbFit <- train(classe ~ ., data=training, method="gbm", 
               trControl=fitControl, verbose=FALSE)
plot(gbFit)
```

![](weightliftingprediction_files/figure-html/gradientboostingmachine-1.png)<!-- -->

## Testing Accuracies

To test the out-of-sample accuracies, the models were used to predict the classes of the testing set, and these predictions were compared with the actual observed values to calculate model accuracy.


```r
dtPred <- predict(dtFit, newdata=testing)
rfPred <- predict(rfFit, newdata=testing)
ldPred <- predict(ldFit, newdata=testing)
gbPred <- predict(gbFit, newdata=testing)
```

The confusion matrices of each of the models, which provide multiple different measures of model accuracy, are shown below. We will be using the measure, "Accuracy", for comparisons.

### Decision Tree


```r
confusionMatrix(factor(testing$classe), dtPred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1997   38  163    0   34
##          B  635  503  380    0    0
##          C  642   44  682    0    0
##          D  585  244  457    0    0
##          E  192  180  396    0  674
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4915          
##                  95% CI : (0.4803, 0.5026)
##     No Information Rate : 0.5163          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3357          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4930  0.49851  0.32820       NA  0.95198
## Specificity            0.9381  0.85154  0.88107   0.8361  0.89241
## Pos Pred Value         0.8947  0.33136  0.49854       NA  0.46741
## Neg Pred Value         0.6341  0.92004  0.78450       NA  0.99469
## Prevalence             0.5163  0.12860  0.26485   0.0000  0.09024
## Detection Rate         0.2545  0.06411  0.08692   0.0000  0.08590
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7155  0.67503  0.60463       NA  0.92219
```

### Random Forest


```r
confusionMatrix(factor(testing$classe), rfPred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B   13 1504    1    0    0
##          C    0   11 1356    1    0
##          D    0    0   18 1267    1
##          E    0    0    3    4 1435
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9934         
##                  95% CI : (0.9913, 0.995)
##     No Information Rate : 0.2861         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9916         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9942   0.9927   0.9840   0.9961   0.9993
## Specificity            1.0000   0.9978   0.9981   0.9971   0.9989
## Pos Pred Value         1.0000   0.9908   0.9912   0.9852   0.9951
## Neg Pred Value         0.9977   0.9983   0.9966   0.9992   0.9998
## Prevalence             0.2861   0.1931   0.1756   0.1621   0.1830
## Detection Rate         0.2845   0.1917   0.1728   0.1615   0.1829
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9971   0.9953   0.9911   0.9966   0.9991
```

### Linear Discriminant Analysis


```r
confusionMatrix(factor(testing$classe), ldPred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1795   48  172  204   13
##          B  238  950  183   68   79
##          C  119  139  890  184   36
##          D   64   59  151  971   41
##          E   58  212  128  154  890
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7005          
##                  95% CI : (0.6902, 0.7106)
##     No Information Rate : 0.2898          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6214          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7894   0.6747   0.5840   0.6142   0.8404
## Specificity            0.9216   0.9118   0.9244   0.9497   0.9187
## Pos Pred Value         0.8042   0.6258   0.6506   0.7551   0.6172
## Neg Pred Value         0.9147   0.9276   0.9021   0.9070   0.9736
## Prevalence             0.2898   0.1795   0.1942   0.2015   0.1350
## Detection Rate         0.2288   0.1211   0.1134   0.1238   0.1134
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8555   0.7932   0.7542   0.7819   0.8795
```

### Gradient Boosting Machine Model


```r
confusionMatrix(factor(testing$classe), gbPred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2201   23    4    3    1
##          B   46 1432   35    1    4
##          C    0   51 1299   17    1
##          D    0    5   28 1241   12
##          E    4   28   10   19 1381
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9628          
##                  95% CI : (0.9584, 0.9669)
##     No Information Rate : 0.2869          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9529          
##                                           
##  Mcnemar's Test P-Value : 1.273e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9778   0.9305   0.9440   0.9688   0.9871
## Specificity            0.9945   0.9864   0.9893   0.9931   0.9905
## Pos Pred Value         0.9861   0.9433   0.9496   0.9650   0.9577
## Neg Pred Value         0.9911   0.9831   0.9881   0.9939   0.9972
## Prevalence             0.2869   0.1962   0.1754   0.1633   0.1783
## Detection Rate         0.2805   0.1825   0.1656   0.1582   0.1760
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9861   0.9584   0.9667   0.9810   0.9888
```

### Best Model

Based on these results, all models excluding the linear discriminant analysis model seem to have very high (greater than 95%) accuracy values. However, the random forest method had the highest out-of-sample accuracy of 99.34%. Consequently, this method will be used in the final predictions for the validation set.

## Final Predictions

The final predictions for the classes for the 20 observations in the validation set, based on the random forest model, are shown below. The expected out-of-sample accuracy has a 95% confidence interval of 99.13% to 99.5%. Consequently, the out-of-sample error, which is defined as $1-accuracy$, has a 95% confidence interval of 0.5% to 0.87%.


```r
predict(rfFit, newdata=validation)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
