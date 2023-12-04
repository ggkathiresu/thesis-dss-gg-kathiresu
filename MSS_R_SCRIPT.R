install.packages("readxl")
install.packages("dplyr")
install.packages("rpart")
install.packages("randomForest")
library(readxl)
library(dplyr)
library(rpart)
library(randomForest)
library(caret)

data <- read_excel("mijn_data_2.0.xlsx")

data <- data[!is.na(data$BLOOD) & !is.na(data$PLASMA), ]
data[is.na(data)] <- 9

unique(data$BLOOD)
table(data$BLOOD)

data$BLOOD[is.na(data$BLOOD)] <- 9
data$BLOOD <- ifelse(data$BLOOD %in% c(1, 3), 1, ifelse(data$BLOOD %in% c(2, 4), 0, data$BLOOD))
data$PLASMA[is.na(data$PLASMA)] <- 9

data$PLASMA <- ifelse(data$PLASMA %in% c(1, 3), 1, ifelse(data$PLASMA %in% c(2, 4), 0, data$PLASMA))
data$BLOODOPLASMA <- ifelse(data$BLOOD == 1 & data$PLASMA == 1, 1, 0)

data <- subset(data, select = -c(caseid))
data <- subset(data, select = -c(uniqid))

data$BLOOD <- factor(data$BLOOD)
data$PLASMA <- factor(data$PLASMA)
data$BLOODOPLASMA <- factor(data$BLOODOPLASMA)

data <- subset(data, select = -c(BLOOD))
data <- subset(data, select = -c(PLASMA))

dataPlasma <- subset(data, select = -c(BLOOD, BLOODOPLASMA))
dataBlood <- subset(data, select = -c(PLASMA, BLOODOPLASMA))
dataBloodPLASMA <- subset(data, select = -c(PLASMA, BLOOD))

table(data$BLOOD)
table(data$PLASMA)
table(data$BLOODOPLASMA)

str(data)
summary(data)

baseline_accuracy <- numeric(length(folds))
rf_accuracy <- numeric(length(folds))
dt_accuracy <- numeric(length(folds))

rf_confusion_matrices <- list()
dt_confusion_matrices <- list()
f1_scores_rf <- numeric(length(folds))
f1_scores_dt <- numeric(length(folds))
recall_rf <- numeric(length(folds))
recall_dt <- numeric(length(folds))
precision_rf <- numeric(length(folds))
precision_dt <- numeric(length(folds))

set.seed(123)
folds <- createFolds(data$BLOODOPLASMA, k = 10, list = TRUE, returnTrain = FALSE)  

for (i in 1:length(folds)) {
  test_indices <- unlist(folds[i])
  train_indices <- setdiff(1:nrow(data), test_indices)
  
  train_data <- data[train_indices, ]
  test_data <- data[test_indices, ]
  
  baseline_accuracy[i] <- max(table(train_data$BLOODOPLASMA)) / nrow(train_data)
  
  rf_model <- randomForest(BLOODOPLASMA ~ ., data = train_data, ntree = 500, maxdepth = 15, mtry = sqrt(ncol(train_data)), sampsize = 100)
  rf_predictions <- predict(rf_model, newdata = test_data)
  rf_accuracy[i] <- mean(rf_predictions == test_data$BLOODOPLASMA)
  rf_confusion_matrix <- confusionMatrix(rf_predictions, test_data$BLOODOPLASMA)
  rf_confusion_matrices[[i]] <- as.matrix(rf_confusion_matrix$table)
  f1_scores_rf[i] <- rf_confusion_matrix$byClass["F1"]
  recall_rf[i] <- rf_confusion_matrix$byClass["Sensitivity"]
  precision_rf[i] <- rf_confusion_matrix$byClass["Pos Pred Value"]
  
  dt_model <- train(BLOODOPLASMA ~ ., data = train_data, method = "rpart", trControl = ctrl, control = rpart.control(maxdepth = 20))
  dt_predictions <- predict(dt_model, newdata = test_data)
  dt_accuracy[i] <- mean(dt_predictions == test_data$BLOODOPLASMA)
  dt_confusion_matrix <- confusionMatrix(dt_predictions, test_data$BLOODOPLASMA)
  dt_confusion_matrices[[i]] <- as.matrix(dt_confusion_matrix$table)
  f1_scores_dt[i] <- dt_confusion_matrix$byClass["F1"]
  recall_dt[i] <- dt_confusion_matrix$byClass["Sensitivity"]
  precision_dt[i] <- dt_confusion_matrix$byClass["Pos Pred Value"]
  
  print(paste("Fold", i, "Baseline accuracy:", baseline_accuracy[i]))
  print(paste("Fold", i, "Random Forest accuracy:", rf_accuracy[i]))
  print(paste("Fold", i, "Decision Tree accuracy:", dt_accuracy[i]))
}

mean_baseline_accuracy <- mean(baseline_accuracy)
mean_rf_accuracy <- mean(rf_accuracy)
mean_dt_accuracy <- mean(dt_accuracy)

print(paste("Mean Baseline accuracy:", mean_baseline_accuracy))
print(paste("Mean Random Forest accuracy:", mean_rf_accuracy))
print(paste("Mean Decision Tree accuracy:", mean_dt_accuracy))

avg_rf_confusion_matrix <- Reduce(`+`, rf_confusion_matrices) / length(folds)
print("Average Confusion Matrix for Random Forest:")
print(avg_rf_confusion_matrix)

avg_dt_confusion_matrix <- Reduce(`+`, dt_confusion_matrices) / length(folds)
print("Average Confusion Matrix for Decision Tree:")
print(avg_dt_confusion_matrix)

mean_f1_rf <- mean(f1_scores_rf)
mean_recall_rf <- mean(recall_rf)
mean_precision_rf <- mean(precision_rf)
mean_f1_dt <- mean(f1_scores_dt)
mean_recall_dt <- mean(recall_dt)
mean_precision_dt <- mean(precision_dt)

print(paste("Mean F1-score for Random Forest:", mean_f1_rf))
print(paste("Mean Recall for Random Forest:", mean_recall_rf))
print(paste("Mean Precision for Random Forest:", mean_precision_rf))

print(paste("Mean F1-score for Decision Tree:", mean_f1_dt))
print(paste("Mean Recall for Decision Tree:", mean_recall_dt))
print(paste("Mean Precision for Decision Tree:", mean_precision_dt))

library(caret)
set.seed(123)
grid <- expand.grid(cp = seq(0, 0.1, by = 0.01))
ctrl <- trainControl(method = "cv", number = 5)
dt_model_tuned <- train(BLOOD ~ . + PLASMA, data = train_data, method = "rpart", trControl = ctrl, tuneGrid = grid)
print(dt_model_tuned)

best_dt_model <- rpart(BLOOD ~ . + PLASMA, data = train_data, method = "class", cp = dt_model_tuned$bestTune$cp)
dt_predictions_tuned <- predict(best_dt_model, newdata = test_data)
optimized_dt_accuracy <- mean(dt_predictions_tuned == test_data$BLOOD)  
print(paste("Fold", i, "Optimized Decision Tree accuracy:", optimized_dt_accuracy))

dt_model <- rpart(BLOOD ~ . + PLASMA, data = train_data, method = "class", control = rpart.control(maxdepth = 8))
dt_predictions <- predict(dt_model, newdata = test_data)
dt_accuracy[i] <- mean(dt_predictions == test_data$BLOOD)
print(paste("Fold", i, "Adjusted Decision Tree accuracy:", dt_accuracy[i]))

dt_model <- rpart(BLOOD ~ . + PLASMA, data = train_data, method = "class",
                  control = rpart.control(minsplit = 20, minbucket = 10, cp = 0.01))
dt_predictions <- predict(dt_model, newdata = test_data)
dt_accuracy[i] <- mean(dt_predictions == test_data$BLOOD)

library(caret)
ctrl <- trainControl(method = "cv", number = 10)
rf_model <- train(BLOODOPLASMA ~ ., data = data, method = "rf",
                  trControl = ctrl, ntree = 500, maxdepth = 15, mtry = sqrt(ncol(data)), sampsize = 100)
dt_model <- train(BLOODOPLASMA ~ ., data = data, method = "rpart",
                  trControl = ctrl, control = rpart.control(maxdepth = 20))

rf_predictions <- predict(rf_model, newdata = data)
dt_predictions <- predict(dt_model, newdata = data)

confusionMatrix(rf_predictions, data$BLOODOPLASMA)
confusionMatrix(dt_predictions, data$BLOODOPLASMA)

accuracy_data <- data.frame(
  Model = c("Baseline", "Random Forest", "Decision Tree"),
  Accuracy = c(mean_baseline_accuracy, mean_rf_accuracy, mean_dt_accuracy)
)

# Remaining code removed for brevity...
