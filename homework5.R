# Load necessary libraries
library(foreign)
library(psych)
library(car)
library(lattice)

# Load the USArrests dataset
data("USArrests")

# Look at the dataset
head(USArrests)

# We'll use this as the base dataset
us_crime_data <- USArrests

# Check the structure of the data
str(us_crime_data)

# Summary statistics for the dataset
psych::describe(us_crime_data)

# Checking the correlation between variables
cor(us_crime_data)

#3. Correlation Analysis (cor()):
#Murder and Assault: There’s a strong positive correlation (0.80), meaning as assault rates increase, murder rates tend to increase as well.
#Murder and UrbanPop: A weak correlation (0.07), suggesting urban population size has little to no relationship with murder rates.
#Murder and Rape: A moderate correlation (0.56), indicating some connection between murder and rape rates.

# Linear regression model
fit1 <- lm(Murder ~ ., data = us_crime_data)
summary(fit1)
# The model shows that 'Assault' has a significant positive relationship with 'Murder' (p-value < 0.001),
# while 'UrbanPop' has a marginally significant negative relationship (p-value = 0.0559).
# 'Rape' does not show significant impact on 'Murder' (p-value = 0.2764).
# The model explains 67.21% of the variance in 'Murder' (Multiple R-squared = 0.6721).
# The F-statistic (31.42) and its associated p-value (3.322e-11) indicate that the model is statistically significant.

fit2 <- lm(UrbanPop ~ ., data = us_crime_data)
summary(fit2)
# 'Murder' and 'Rape' both have statistically significant effects on 'UrbanPop' (p-values < 0.05),
# while 'Assault' does not show a significant effect (p-value = 0.2186).
# The model explains 23.37% of the variance in 'UrbanPop' (Multiple R-squared = 0.2337).
# The F-statistic (4.676) and its p-value (0.006208) indicate that the model is statistically significant.

# Linear regression model for predicting 'Assault' based on other variables
fit4 <- lm(Assault ~ ., data = us_crime_data)
summary(fit4)
# 'Murder' and 'Rape' are both significant predictors of 'Assault' (p-values < 0.05),
# while 'UrbanPop' does not significantly affect 'Assault' (p-value = 0.2186).
# The model explains 71.92% of the variance in 'Assault' (Multiple R-squared = 0.7192).
# The F-statistic (39.27) and its associated p-value (9.678e-13) indicate that the model is highly statistically significant.

fit3 <- lm(Rape ~ ., data = us_crime_data)
summary(fit3)
# 'Assault' and 'UrbanPop' both have significant positive relationships with 'Rape' (p-values < 0.05),
# whereas 'Murder' does not show a significant effect (p-value = 0.2764).
# The model explains 51.66% of the variance in 'Rape' (Multiple R-squared = 0.5166).
# The F-statistic (16.39) and its p-value (2.21e-07) indicate that the model is statistically significant.

fit4 <- lm(Assault ~ ., data = us_crime_data)
summary(fit4)
# 'Murder' and 'Rape' are both significant predictors of 'Assault' (p-values < 0.05),
# while 'UrbanPop' does not significantly affect 'Assault' (p-value = 0.2186).
# The model explains 71.92% of the variance in 'Assault' (Multiple R-squared = 0.7192).
# The F-statistic (39.27) and its associated p-value (9.678e-13) indicate that the model is highly statistically significant.


# Compute R^2 manually
# Predict for Murder (already done)
crime_pred_murder <- predict(fit1, newdata = us_crime_data)

# Predict for UrbanPop (using the model fit3)
crime_pred_urbanpop <- predict(fit3, newdata = us_crime_data)

# Predict for Assault (using the model fit4)
crime_pred_assault <- predict(fit4, newdata = us_crime_data)

# Predict for Rape (using the model fit2)
crime_pred_rape <- predict(fit2, newdata = us_crime_data)

# Combine all predictions into a data frame to see them together
crime_predictions <- data.frame(
  Murder_Predicted = crime_pred_murder,
  Rape_Predicted = crime_pred_rape,
  UrbanPop_Predicted = crime_pred_urbanpop,
  Assault_Predicted = crime_pred_assault)

# View the combined predictions
head(crime_predictions)

# Total sum of squares (SStotal)
crime_ss <- sum((us_crime_data$Murder - mean(us_crime_data$Murder))^2)
crime_ss 

# R2 manually from predicted and measured data
R2orig <- 1 - sum((crime_pred_murder - us_crime_data$Murder)^2) / crime_ss
R2orig
#R² = 0.672 means that about 67.2% of the variance in the Murder variable is 
#explained by the independent variables in your model (such as Assault, UrbanPop, and Rape).

###PREDICTIONS
# 10-fold cross-validation
k <- 10
n <- nrow(us_crime_data)

# Create partitions for cross-validation
set.seed(2010)
part <- rep(1:k, length.out = n)
part <- sample(part)

# Create a vector to hold predictions
crime_pred_cv <- rep(NA, n)

# Perform cross-validation
for (i in 1:k) {
  # Estimate the model on all subsamples except the selected one
  fit_cv <- lm(Murder ~ ., data = us_crime_data[part != i, ])
  
  # Generate predictions for the selected subsample
  crime_pred_cv[part == i] <- predict(fit_cv, newdata = us_crime_data[part == i, ])
}

# Compute R^2 for cross-validation
R2CV <- 1 - sum((crime_pred_cv - us_crime_data$Murder)^2) / crime_ss
R2orig
R2CV
#An lower R2 indicates that the model may not generalize perfectly, but it’s still doing a 
#reasonably good job of predicting Murder with 60.19% of the variation explained on new data. 

# Jackknife resampling (leave-one-out)
crime_pred_jack <- rep(NA, n)

for (i in 1:n) {
  # Estimate the model on all units except the selected one
  fit_jack <- lm(Murder ~ ., data = us_crime_data[-i, ])
  
  # Generate predictions for the remaining unit
  crime_pred_jack[i] <- predict(fit_jack, newdata = us_crime_data[i, ])
}

# Compute R^2 for jackknife resampling
R2jack <- 1 - sum((crime_pred_jack - us_crime_data$Murder)^2) / crime_ss
R2orig
R2jack
#A lower R2jack can indicate that the model may not generalize perfectly, but it’s still doing a 
#reasonably good job of predicting Murder with 60.19% of the variation explained on new data. 

# 100 repetitions of 10-fold cross-validation
m <- 100
R2CVvec <- rep(NA, m)

for (j in 1:m) {
  # Randomly permute the partition for cross-validation
  part <- sample(part)
  
  crime_pred_cv <- rep(NA, n)
  for (i in 1:k) {
    # Estimate the model on all subsamples except the selected one
    fit_cv <- lm(Murder ~ ., data = us_crime_data[part != i, ])
    
    # Generate predictions for the selected subsample
    crime_pred_cv[part == i] <- predict(fit_cv, newdata = us_crime_data[part == i, ])
  }
  
  # Compute R^2 for this repetition
  R2CV <- 1 - sum((crime_pred_cv - us_crime_data$Murder)^2) / crime_ss
  R2CVvec[j] <- R2CV
}

# Mean R^2 across repetitions
R2CVmean <- mean(R2CVvec)
hist(R2CVvec)
R2orig
R2CVmean
#A lower R²CV indicates that the model may not generalize perfectly, but it’s still doing a 
#reasonably good job of predicting Murder with 60.19% of the variation explained on new data. 

# Bootstrap resampling
m <- 1000
diffErrSsBoot <- numeric(m)

set.seed(2010)
for (i in 1:m) {
  boot_ids <- sample(1:n, size = n, replace = TRUE)  # Resample with replacement
  fit_boot <- lm(Murder ~ ., data = us_crime_data[boot_ids, ])  # Fit model on bootstrap sample
  errSsBoot <- sum(fit_boot$resid^2)  # Calculate sum of squared residuals for the bootstrap model
  predOrg <- predict(fit_boot, newdata = us_crime_data)  # Predict using the bootstrap model on original data
  errSsOrg <- sum((predOrg - us_crime_data$Murder)^2)  # Calculate sum of squared residuals for the original data
  diffErrSsBoot[i] <- errSsOrg - errSsBoot  # Store the difference in error sums
}

# Mean difference in error sums
mean(diffErrSsBoot)

# Calculate R^2 from bootstrap
R2boot <- 1 - (sum(fit1$resid^2) + mean(diffErrSsBoot)) / crime_ss
R2orig
R2boot
#A lower R²CV indicates that the model may not generalize perfectly, but it’s still doing a 
#reasonably good job of predicting Murder with 60.19% of the variation explained on new data. 

# Comparing R2 values
R2s <- c(R2orig = R2orig, R2CV = R2CV, R2CVmean = R2CVmean, R2jack = R2jack, R2boot = R2boot)
R2s
# Summary:
# All these R² values are fairly close to each other, indicating that the model generalizes well across different validation techniques. The slight variations suggest that the model is relatively stable and performs similarly in different testing conditions.

