#----------------------------------------------#
#Title:    "Type 2 Diabetes Mellitus Prediction"
#Author:   "Alex Abal"
#Date:      17th February 2024
#--------------------------------------------#

library(ROSE)
library(xgboost)
library(reshape)
library(janitor)
library(visdat)
library(kernlab)
library(caret) #ML Model buidling package
library(tidyverse) #ggplot and dplyr
library(mlbench) #data sets from the UCI repository.
library(summarytools)
library(corrplot) #Correlation plot
library(gridExtra) #Multiple plot in single grip space
library(pROC) #ROC
library(caTools) #AUC
library(rpart.plot) #CART Decision Tree
library(e1071) #imports graphics, grDevices, class, stats, methods, utils
library(graphics) #fourfoldplot
library(naniar)# for missing values
library(mice)# for Imputation using mice
library(dplyr)# For Glimpse
library(Boruta)
library(rattle)
library(gtsummary)

# Loading the dataset
data <- read.csv("Dataset/trial.csv")

df <- data[sample(nrow(data)), ]

# Code for checking for null value
sum(is.na(df))

#Plot of missing values
vis_miss(df) 


#Cleaning the dataset 
df1 <- clean_names(df)

#Encoding the target variable as a factor
df1$diabetes_mellitus = factor(df1$diabetes_mellitus, 
                               levels = c(0,1),
                               labels = c("Negative", "Positive"))

#Encoding sex variable
df1$sex = factor(df1$sex, 
                 levels = c(0,1),
                 labels = c("Male", "Female"))

#Encoding Physical activity variable
df1$physical_activity = factor(df1$physical_activity, 
                               levels = c(0,1),
                               labels = c("Not active", "Active"))

#Encoding Smoking status variable
df1$smoking_status = factor(df1$smoking_status, 
                            levels = c(0,1),
                            labels = c("non smoker", "smoker"))

#Encoding Alcohol status variable
df1$alcohol_status = factor(df1$alcohol_status, 
                            levels = c(0,1),
                            labels = c("Non user", "user"))

#Encoding Hypertension variable
df1$hypertension = factor(df1$hypertension, 
                          levels = c(0,1),
                          labels = c("Not sick", "Sick"))

#Describing the Statistics of the dataset
demo_stat <- df1

data_set_Stat <- demo_stat %>% dplyr::select(age, sex, bmi, weight, insulin, glucose,
                                  systolic, diastolic, alcohol_status, 
                                  smoking_status, physical_activity,
                                  hypertension, diabetes_mellitus)

tbl_summary(data_set_Stat, 
            by =  diabetes_mellitus,
            statistic = list(
              all_continuous() ~ "{mean} ({sd})"),
            missing = "no") %>%
  as_gt(include = -cols_align) %>%
  gt::tab_source_note(gt::md("*This data is Demographic Statitistics of Dataset*"))

#data_set_Stat %>% tbl_summary()

#Imputation of missing values using the Mice package.
data_imp = mice(df1,
                m=5, 
                method =c('pmm','','pmm',
                          'pmm','pmm','pmm','pmm',
                          'pmm','','','','',''), maxit = 50)
#Getting plot of Imputed values
plot(data_imp)

#Checking the data fit to see if the imputed values are a good fit to the data
stripplot(data_imp, pch = 20, cex = 1.2)

#Comparing the output
data_imp$imp$weight 

#Selection of final dataset to be used
final_data = complete(data_imp, 3)
df3 <- final_data

#Checking for missing values if still do exit in the final dataset
is.na(df3)

#Developing a balanced dataset using oversampling technique
data_df3 <- ovun.sample(diabetes_mellitus ~ ., data = df3, 
                                  method = "both", p=0.5,
                                  N=1920, seed = 1)$data
df4 <- data_df3

#Checking to see if my data is normalized, if normalized SD= 1 , mean = 0
sd(df4$bmi)

#Normalizing the dataset so that the mean is Zero and SD = 1
process <- preProcess(as.data.frame(df4), method=c("range"))
norm_scale <- predict(process, as.data.frame(df4))
df5 <- norm_scale

#Calculating mean of scaled variables
sd(df5$bmi)

#Data Exploration Process starts here


# Convert categorical variables
data_new <- sapply(df5, unclass)          
M <- data_new

# Default Heatmap
corrplot(cor(M), method="shade", main = "Heat Map Showing Relationships between Variables",
         sub = "sub", type = "full", shade.col=TRUE, tl.col="darkblue", tl.srt=45)


#Data preparation
cutoff <- caret::createDataPartition(df5$diabetes_mellitus, times = 1, p=0.80, list=FALSE)

# 20% of the data used for validation
testdf <- df5[-cutoff,]

# 80% of data used for model training
traindf <- df5[cutoff,]

#Identifying variable of importance
boruta_output <- Boruta(diabetes_mellitus~., data=traindf, doTrace=0)
rough_fix_mod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(rough_fix_mod)
importances <- attStats(rough_fix_mod)
importances <- importances[importances$decision !="Rejected", c("meanImp", "decision")]
importances[order(-importances$meanImp), ]

# look at the attribute statistics
attStats(boruta_output)

plot(boruta_output, las =2, cex.axis=0.7, xlab="",
     main= "Important variables used development of Prediction models")


#Diabetes Distribution
ggplot(traindf, aes(traindf$diabetes_mellitus, fill = diabetes_mellitus)) + 
  geom_bar() +
  theme_bw() +
  labs(title = "Diabetes Mellitus Classification", x = "Diabetes Mellitus") +
  theme(plot.title = element_text(hjust = 0.5))

#Univariate Analysis



#Bivariate analysis




# Outlier Detection, the Caret function will handle outliers
boxplot(df5,
        main = "Identification of Outliers",
        xlab = "Variables",
        ylab = "Percentiles",
        col = "pink",
        border = "brown",
        horizontal = F,
        notch = F
)

# Training the SVM, I set the 10 fold cross validation 
set.seed(223)
control <- trainControl(method="cv",number=10,  summaryFunction = twoClassSummary, classProbs = TRUE)
metric <- "ROC"
model_svm <- caret::train(diabetes_mellitus ~., 
                           
                          data = traindf,
                          method = "svmRadial",
                          metric = metric,
                          tuneLength = 8,
                          trControl =control,
                          preProcess = c("center","scale"))
model_svm


#Code Let’s plot the accuracy graph.
plot((model_svm),main="ROC for Support Vector Model during Training")

# final ROC value
model_svm$results["ROC"]

#Prediction of SVM model using the Test Data.
predict_svm <- predict(model_svm, newdata = testdf)

#Confusion Matrix 
cm_svm <-confusionMatrix(predict_svm, testdf$diabetes_mellitus, positive="Positive")
cm_svm

# Prediction Probabilities
pred_prob_svm <- predict(model_svm, testdf, type="prob")

#ROC value
roc_svm <- roc(testdf$diabetes_mellitus, pred_prob_svm$Positive)
roc_svm

# Plot the ROC curve with AUC value
plot(roc_svm, 
     colorize=TRUE,
     avg='horizontal',
     spread.estimate='boxplot',
     lwd=3,
     col='purple',
     print.auc = TRUE, auc.polygon = TRUE, 
     legacy.axes = TRUE, main = "Support Vector Model ROC Curve with AUC Value")


#Random Forest Model
set.seed(223)
control <- trainControl(method="cv",number=10,  summaryFunction = twoClassSummary, classProbs = TRUE)
metric <- "ROC"
model_rfm<- caret::train(diabetes_mellitus ~., 
                        data = traindf,
                        method = "ranger",
                        metric = metric,
                        trControl = control,
                        preProcess = c("center","scale","pca"))
model_rfm

#Final ROC value
model_rfm$results["ROC"]

# Plotting model
plot(model_rfm, main="ROC for Random Forest Model during Training")

#Prediction/ validating the Random Forest model using the test dataset
predict_rfm <- predict(model_rfm, newdata = testdf)

#Confusion Matrix 
cm_rfm <-confusionMatrix(predict_rfm, testdf$diabetes_mellitus, positive="Positive")
cm_rfm

#Prediction Probabilities
pred_prob_rfm <- predict(model_rfm, testdf, type="prob")

#ROC value
roc_rfm <- roc(testdf$diabetes_mellitus, pred_prob_rfm$Positive)
roc_rfm

# Plot the ROC curve with AUC value
plot(roc_rfm, 
     colorize=TRUE,
     avg='horizontal',
     spread.estimate='boxplot',
     lwd=4,
     col='darkblue',
     print.auc = TRUE, auc.polygon = TRUE, 
     legacy.axes = TRUE, main = "Random Forest ROC Curve with AUC Value")


# EXtreme Gradient Boosting 
xgb_grid  <-  expand.grid(
  nrounds = 150,
  eta = c(0.03),
  max_depth = 1,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1,
  subsample = 0.5
)
set.seed(223)
control <- trainControl(method="cv", number=10,  summaryFunction = twoClassSummary, classProbs = TRUE)
metric <- "ROC"
model_xgb <- caret::train(diabetes_mellitus ~., 
                          data = traindf,
                          method = "xgbTree",
                          metric = metric,
                          tuneGrid=xgb_grid,
                          trControl = control,
                          preProcess = c("center","scale","pca"))
model_xgb

#Final ROC value
model_xgb$results["ROC"]

# Plotting model
#plot(model_xgb)

#XGBOOST - eXtreme Gradient BOOSTing model using the test dataset
predict_xgb <- predict(model_xgb, newdata = testdf)

#Confusion Matrix 
cm_xgb <-confusionMatrix(predict_xgb, testdf$diabetes_mellitus, positive="Positive")
cm_xgb

#Prediction Probabilities
pred_prob_xgb <- predict(model_xgb, testdf, type="prob")

#ROC value
roc_xgb <- roc(testdf$diabetes_mellitus, pred_prob_xgb$Positive)
roc_xgb

# Plot the ROC curve with AUC value
plot(roc_xgb, 
     colorize=TRUE,
     avg='horizontal',
     spread.estimate='boxplot',
     lwd=3,
     col='black',
     print.auc = TRUE, auc.polygon = TRUE, 
     legacy.axes = TRUE, main = "Extreme Gradient Boosting ROC Curve with AUC Value")


#K Nearest Neighbor Code

set.seed(223)
control <- trainControl(method="cv", number=10,  summaryFunction = twoClassSummary, classProbs = TRUE)
metric <- "ROC"
model_knn <- caret::train(diabetes_mellitus ~., 
                          data = traindf,
                          method = "knn",
                          metric = metric,
                          tuneGrid = expand.grid(.k = c(5,7,9,11)),
                          trControl = control,
                          preProcess = c("center","scale","pca"))
model_knn
plot((model_knn), main="ROC for K_Nearest Neigbhour Model during Training")

# final ROC value
model_knn$results["ROC"]

#validating the K_Nearest Neigbhour model using the test dataset
predict_model_knn <- predict(model_knn, newdata = testdf)

# Confusion Matrix 
cm_knn <-confusionMatrix(predict_model_knn, testdf$diabetes_mellitus, positive="Positive")
cm_knn

# Prediction Probabilities
pred_prob_knn <- predict(model_knn, testdf, type="prob")

# ROC value
roc_knn <- roc(testdf$diabetes_mellitus, pred_prob_knn$Positive)
roc_knn

# Plot the ROC curve with AUC value
plot(roc_knn, 
     colorize=TRUE,
     avg='horizontal',
     spread.estimate='boxplot',
     lwd=3,
     col='orange',
     print.auc = TRUE, auc.polygon = TRUE, 
     legacy.axes = TRUE, main = "K-Nearest Neigbor ROC Curve with AUC Value")


# Decision Tree Model
set.seed(223)
control <- trainControl(method="cv", number=10,  summaryFunction = twoClassSummary, classProbs = TRUE)
metric <- "ROC"
model_dtm <- caret::train(diabetes_mellitus ~., 
                          data = traindf,
                          method = "rpart",
                          metric = metric,
                          tuneLength = 20,
                          trControl = control)
                          #preProcess = c("center","scale","pca"))

model_dtm

#Plot model accuracy vs different values of cp (complexity parameter)
plot((model_dtm), main="ROC for Decision Tree Model during Training")

#Final ROC value
model_dtm$results["ROC"] 

#Best Model Cp (Complexity Parameter)
best <- model_dtm$bestTune

#Structure of final model selected
best2 <- model_dtm$finalModel

#Plot of DTM 
#rpart.plot::rpart.plot(best, extra= 106)


# Visualize the decision tree with rpart.plot
rpart.plot::rpart.plot(best2, type =2, fallen.leaves = F, 
                       extra = 101, 
                       faclen=5,
                       under =F,
                       roundint=F, #don't round to integers in output
                       digits=1, #display 5 decimal places in output
                       cex = 0.12,
                       box.palette = "auto")

##prp(model_dtm$finalModel)
#fancyRpartPlot(model_dtm$finalModel, cex = 0.12)

#Validating the Decision Tree model using the test dataset
predict_dtm <- predict(model_dtm, newdata = testdf)

#Confusion Matrix 
cm_dtm <-confusionMatrix(predict_dtm, testdf$diabetes_mellitus, positive="Positive")
cm_dtm

#Prediction Probabilities
pred_prob_dtm <- predict(model_dtm, testdf, type="prob")

# ROC value
roc_dtm <- roc(testdf$diabetes_mellitus, pred_prob_dtm$Positive)
roc_dtm

# Plot the ROC curve with AUC value
plot(roc_dtm, 
     colorize=TRUE,
     avg='horizontal',
     spread.estimate='boxplot',
     lwd=3,
     col='steelblue',
     print.auc = TRUE, auc.polygon = TRUE, 
     legacy.axes = TRUE, main = "Decission Tree ROC Curve with AUC Value")


#Comparison of ROC values of different models using the Training Data set 
model_list <- list(Support_Vector = model_svm, 
                   Random_Forest = model_rfm,
                   XGBoost = model_xgb, 
                   KNN = model_knn, 
                   Decision_Tree = model_dtm)

resamples <- resamples(model_list)

#### Compare models and summarize/visualize results
if(length(model_list)> 1){
  resamp = resamples(model_list)
  # Summarize the results
  summary(resamp)
  dotplot(resamp, 
          ##main="Comparison of ROC values of models with the Training Data set", #
          metric="ROC")

}
xyplot(resamp, metric="ROC" )

#box plot
bwplot(resamples, 
       main="Comparison of ROC values of models with the Training Data set", 
       metric="ROC")

#dot plot
#dotplot(resamples, metric="ROC")


# Comparision of test results using Test Set

result_rfm <- c(cm_rfm$byClass['Sensitivity'], cm_rfm$byClass['Specificity'], cm_rfm$byClass['Precision'], 
               cm_rfm$byClass['Recall'], cm_rfm$byClass['F1'], roc_rfm$auc)

result_xgb <- c(cm_xgb$byClass['Sensitivity'], cm_xgb$byClass['Specificity'], cm_xgb$byClass['Precision'], 
                cm_xgb$byClass['Recall'], cm_xgb$byClass['F1'], roc_xgb$auc)

result_knn <- c(cm_knn$byClass['Sensitivity'], cm_knn$byClass['Specificity'], cm_knn$byClass['Precision'], 
                cm_knn$byClass['Recall'], cm_knn$byClass['F1'], roc_knn$auc)

result_svm <- c(cm_svm$byClass['Sensitivity'], cm_svm$byClass['Specificity'], cm_svm$byClass['Precision'], 
                cm_svm$byClass['Recall'], cm_svm$byClass['F1'], roc_svm$auc)

result_dtm <- c(cm_dtm$byClass['Sensitivity'], cm_dtm$byClass['Specificity'], cm_dtm$byClass['Precision'], 
                  cm_dtm$byClass['Recall'], cm_dtm$byClass['F1'], roc_dtm$auc)


all_results <- data.frame(rbind(result_rfm, result_xgb, result_knn, result_svm, result_dtm))
names(all_results) <- c("Sensitivity", "Specificity", "Precision", "Recall", "F1", "AUC")
all_results


#Visualization to compare accuracy of models using Four Fold Plot of the developed Models 
col <- c("darkgreen", "#CA225E")

graphics::fourfoldplot(cm_svm$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("Support Vector Machine(",round(cm_svm$overall[1]*100),"%)", sep = ""))

graphics::fourfoldplot(cm_rfm$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("Random Forest Accuracy(",round(cm_rfm$overall[1]*100),"%)", sep = ""))


graphics::fourfoldplot(cm_xgb$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("Xtreme Gradient Boosting Accuracy(",round(cm_xgb$overall[1]*100),"%)", sep = ""))


graphics::fourfoldplot(cm_knn$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("K-Nearest Neighbour Accuracy(",round(cm_knn$overall[1]*100),"%)", sep = ""))

graphics::fourfoldplot(cm_dtm$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("Decission Tree Accuracy(",round(cm_dtm$overall[1]*100),"%)", sep = ""))
