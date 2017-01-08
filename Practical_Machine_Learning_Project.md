Executive Summary - Predicting Exercise Performance
---------------------------------------------------

The purpose of this study predict the manner in which an individual
exercise was performed.

A considerable amount of data is collected from wearable devices which
capture how much specific activities are performed. For example, an
individual may be interested in recording how many weight lifting
repetitions they accomplished. However, it should be possible to attest
to whether or not they did each repetition properly and record the
number of mistakes.

We shall model movemements collected from accelerometers during these
exercises located at 4 locations on the each participant:

1.  The belt
2.  The upper arm
3.  The forearm
4.  The weight itself.

Six participants performed 10 repetitions of unilateral dumbbell biceps
curls in 5 different manners:

1.  Exactly according to the specification (Class A),
2.  Throwing the elbows to the front (Class B),
3.  Lifting the dumbbell only halfway (Class C),
4.  Lowering the dumbbell only halfway (Class D)
5.  Throwing the hips to the front (Class E).

Class 'A' corresponds to the exercise being performed correctly. The
remaining activity classes correpond to common mistakes. Each method is
specified in the source data by the variable "classe".

*The data used was kindly made available from
<http://groupware.les.inf.puc-rio.br/har> . Thank you to the
collaborators involved during the study.*

### Building Our Model

We began our analysis by exploring the source data. The data contains a
mixture of both granular time series of movements as well as summary
statistics pertaining to the entire range of motion per repetition. We
begin my eliminating the sparse values from our data.

    # Load the source data.
    #options(stringsAsFactors = TRUE)
    pml_training <- read.csv("./pml-training.csv", header = TRUE)
    pml_testing  <- read.csv("./pml-testing.csv",  header = TRUE)

Since our test data only contains specific points and not the full range
of motion in a repetition, we shall exclude these rows from our training
data denoted by the "new window" value.

    pml_training <- subset(pml_training, new_window == "no")

Also, we shall exclude the sparsity of the data caused by empty columns.

    temp <- as.data.table(pml_training)
    temp <- temp[,which(unlist(lapply(temp, function(x)!all( is.na(x)) ))), with=F]

    # clean residual levels that are of 1 distinct value
    temp <- droplevels(temp)
    temp <- temp[, (names(temp)[which(sapply(temp, uniqueN) == 1)]) := NULL]

The test cases we will be classifying do not account for the full range
of motion and are granular points in time recorded within a repetition.
Our model will be compiled respective of the same level so we remove the
time related metrics. Also, since our model should be capable of
predicting what action was performed, "num\_window", should absolutely
be removed as it was left in the test data and has a direct correlation
to the classe in the data available for training.

    # remove the row ordinal value
    # remove the timeseries details
    temp <- temp[,-c('X', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'num_window')]
    training <- as.data.frame(temp)
    dim(training)

    ## [1] 19216    54

### Training Data

We shall train our model using 70% of the observations in our source
data. Later, we will use the remaining 30% of observations to
cross-validate our predictive model.

    train_part <- createDataPartition(y=training$classe, p=0.70, list=FALSE)
    training <- training[train_part,]
    validation <- training[-train_part,]

### Model using Random Forest

The model is compiled using random forest.

    rf.model <- randomForest(classe ~ ., data=training, mtry=2, ntree=500)
    print(rf.model)

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = training, mtry = 2,      ntree = 500) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.74%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3824    3    2    0    1 0.001566580
    ## B   15 2584    4    0    0 0.007299270
    ## C    0   16 2327    4    0 0.008521517
    ## D    0    0   47 2153    3 0.022696323
    ## E    0    0    1    4 2465 0.002024291

### Variable Importance

Observing variable importance, we can see that "roll belt"" contributes
to the most variance in the outcome. Here we plot the belt roll measure
with the 2nd principal component, belt yaw.

    varImpPlot(rf.model)

![](Practical_Machine_Learning_Project_files/figure-markdown_strict/unnamed-chunk-7-1.png)

Here we plot the belt roll measure with the 2nd principal component,
belt yaw.

    qplot(roll_belt, yaw_belt, colour=classe, data=validation)

![](Practical_Machine_Learning_Project_files/figure-markdown_strict/unnamed-chunk-8-1.png)

### Cross Validation

A validation dataset was created as an independent partition of the
source data. We shall use this to observe the out of sample error.

    predictions <- predict(rf.model, validation)
    confusionMatrix <- confusionMatrix(predictions, validation[,c("classe")])
    confusionMatrix

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1149    0    0    0    0
    ##          B    0  796    0    0    0
    ##          C    0    0  706    0    0
    ##          D    0    0    0  637    0
    ##          E    0    0    0    0  763
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9991, 1)
    ##     No Information Rate : 0.2836     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2836   0.1965   0.1743   0.1572   0.1883
    ## Detection Rate         0.2836   0.1965   0.1743   0.1572   0.1883
    ## Detection Prevalence   0.2836   0.1965   0.1743   0.1572   0.1883
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

### Expected out of sample error

Our out-of-sample error is estimated by substracting the total accuracy
from a potential 1 (100% accuracy).

    accuracy <- confusionMatrix$overall[1]
    outOfSampleError <- 1 - accuracy
    paste("Expected out of sample error:", round(100 * outOfSampleError, 5), "%")

    ## [1] "Expected out of sample error: 0 %"

### Prediction Results

The test data, containing 20 test cases, provided is applied to the
predictive model.

    prediction_results  <- predict(rf.model, pml_testing)
    prediction_results

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
