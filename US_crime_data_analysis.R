# Load necessary libraries
library(ggplot2)
library(corrplot)
library(mice)
library(psych)
library(VIM)
library(norm)
library(missForest)

# Load and prepare dataset
data("diamonds")
df <- diamonds
str(df)

# Introduce artificial missing data (30% MCAR)
tVars <- names(df)  
pMiss <- 0.3  
df_missing_matrix <- as.matrix(df[, tVars])  
nVal <- prod(dim(df_missing_matrix))  
nMiss <- round(nVal * pMiss)  

# Introduce MCAR missingness by randomly sampling positions to set as NA
set.seed(123) 
df_missing_matrix[sample(nVal, size = nMiss, replace = FALSE)] <- NA  

# Replace in original df
df_missing <- df  # Copy of original dataset
df_missing[, tVars] <- as.data.frame(df_missing_matrix)  # Update with missing data
head(df_missing)

write.csv(df_missing, "df_missing.csv", row.names = FALSE)
df_missing <- read.csv("df_missing.csv", stringsAsFactors = FALSE)

### 1. Explore Missing Data ----
# Summary of missing data
missing_summary <- colSums(is.na(df_missing))
print(missing_summary)
colMeans(is.na(df_missing))

### Speculate on the missing data mechanism (MCAR, MAR, NMAR)
# MCAR: Missingness is independent of any variable
# MAR: Missingness is related to other variables (but not the missing ones)
# NMAR: Missingness is related to the missing values themselves

# Since I manually deleted 30% of the values at random (MCAR), it’s important to note that MAR (Missing at Random) wouldn't apply in our case. The missingness is random by design, so the missing data mechanism is MCAR. Otherwise, I would be using hte following method:
# Check if missingness in a variable is related to other variables
missing_data_pattern <- sapply(df_missing, function(x) sum(is.na(x)) / length(x))
print(missing_data_pattern)

# Create missingness indicator for each variable
missing_indicators <- as.data.frame(sapply(df_missing, function(x) ifelse(is.na(x), 1, 0)))
print(head(missing_indicators))

# Check if the missingness indicators have variability (non-zero standard deviation)
missing_variability <- apply(missing_indicators, 2, sd) > 0
print(missing_variability)

#### Analysis of df_missing dataset, to later compare with imputed results
# Convert categorical variables to numeric (ordinal encoding)
df_missing_num <- df_missing
df_missing_num$cut <- as.numeric(factor(df_missing_num$cut, ordered = TRUE))
df_missing_num$color <- as.numeric(factor(df_missing_num$color, ordered = TRUE))
df_missing_num$clarity <- as.numeric(factor(df_missing_num$clarity, ordered = TRUE))
df_missing_num$price <- as.numeric(df_missing_num$price)
df_missing_num$carat <- as.numeric(df_missing_num$carat)
df_missing_num$depth <- as.numeric(df_missing_num$depth)
df_missing_num$table <- as.numeric(df_missing_num$table)
df_missing_num$x <- as.numeric(df_missing_num$x)
df_missing_num$y <- as.numeric(df_missing_num$y)
df_missing_num$z <- as.numeric(df_missing_num$z)
str(df_missing_num)

## Complete df_missing case analysis - correlations and PCA
corDiam<-cor(df_missing_num,use="complete.obs")
corDiam
# Correlation Analysis of Diamond Features (corDiam)
# - Carat is highly correlated with price (0.93), x (0.98), y (0.98), and z (0.98), indicating larger diamonds cost more and have bigger dimensions.
# - Price is also strongly correlated with x, y, and z (~0.89), reinforcing that diamond size is a major price determinant.
# - Clarity has a weak negative correlation with carat (-0.20) and price (-0.06), suggesting that larger/more expensive diamonds may have lower clarity.
# - Cut shows minimal correlation with most variables, except a weak negative correlation with depth (-0.21), implying better cuts tend to have lower depth percentages.
# - Color has a weak positive correlation with carat (0.28) and price (0.18), meaning larger diamonds may tend to have lower color grades.
# - Depth has a moderate negative correlation with table (-0.33), indicating that diamonds with larger tables tend to have lower depth.
# - Table has weak correlations with most features, except a moderate positive correlation with carat (0.15).
# - x, y, and z are highly intercorrelated (~0.99), as expected since they measure physical dimensions of the diamond.

pa_df<-princomp(covmat=corDiam)
summary(pa_df)
loadings(pa_df)
plot(pa_df,type="lines")
# Principal Component (PC) Interpretation:
# - PC1: Dominant factor capturing size-related attributes (carat, price, dimensions x, y, z).
# - PC2: Opposes depth and table size against cut quality.
# - PC3: Primarily represents clarity and color distinctions.
# - PC4 & PC5: Further refine quality aspects, balancing color, clarity, and cut.
# - PC6: Minor influence of cut and table proportions.
# - PC7: Residual variation in price, capturing less significant effects.
# - PCs 8-10: Capture minimal remaining variance, mostly noise.

# Standard deviation of each principal component (PC):
# - PC1 has the highest SD (2.23), meaning it captures the largest spread in the dataset.
# - Subsequent PCs explain progressively smaller amounts of variation, as expected in PCA.

# Proportion of Variance Explained:
# - PC1 accounts for ~49.5% of total variance, making it the dominant component.
# - PC2 contributes ~14.6%, so together, PC1 and PC2 explain ~64% of variance.
# - PC3 adds ~10.7%, bringing the total explained variance to ~75%.
# - By PC4 (~9.0%), cumulative variance reaches ~84%.
# - By PC5 (~8.3%), the first 5 PCs explain ~92%, capturing most of the dataset’s structure.
# - PCs beyond the first 6 contribute diminishing returns, explaining minimal additional variance.

# Missing FA
# Factor Analysis on df_missing dataset
fa.parallel(df_missing_num, fa="fa", n.iter=100, show.legend=TRUE, main="Parallel Analysis Scree Plot for FA")
fa.parallel
#The optimal factors calculation says 5, however, the Scree plot has a cutoff at 4.

# FA with 5 factors, unrotated
fa_res <- fa(df_missing_num, nfactors=4, rotate="none")
print(fa_res$loadings)
fa.diagram(fa_res)

fa_complete <- fa(df_missing_num, nfactors = 5, rotate = "varimax")
print(fa_complete$loadings)
fa.diagram(fa_complete)

fa_complete <- fa(df_missing_num, nfactors = 5, rotate = "oblimin")
print(fa_complete$loadings)
fa.diagram(fa_complete)
# Choosing Rotation for Factor Analysis

# No Rotation (rotate="none"):
# - Most variance is concentrated in MR1 (0.488), making interpretation difficult.
# - Poor factor balance, not recommended.

# Varimax Rotation (rotate="varimax"):
# - Assumes factors are uncorrelated.
# - Provides clearer factor distinctions but still shows some cross-loadings.
# - Ultra-Heywood case detected, indicating possible model misfit.

# Oblimin Rotation (rotate="oblimin"):
# - Allows factors to be correlated, making it more realistic for diamonds data.
# - Balances loadings better than varimax.
# - Still has Ultra-Heywood warnings, so check for multicollinearity.

# Conclusion:
# - I will use oblimin as factors are expected to be correlated.
# - I will avoid no rotation due to poor factor balance.

#Compare to complete dataset:
df$cut <- as.numeric(factor(df_missing_num$cut, ordered = TRUE))
df$color <- as.numeric(factor(df_missing_num$color, ordered = TRUE))
df$clarity <- as.numeric(factor(df_missing_num$clarity, ordered = TRUE))
fa_res <- fa(df, nfactors = 4, rotate="none")
print(fa_res$loadings)
fa.diagram(fa_res)

fa_res <- fa(df, nfactors = 5, rotate="varimax")
print(fa_res$loadings)
fa.diagram(fa_res)

fa_res <- fa(df, nfactors = 4, rotate="oblimin")
print(fa_res$loadings)
fa.diagram(fa_res)
#The FA structure was highly sensitive to missing data.
#After artificially deleting 30% of the data, factor structures changed significantly.
#Variables shifted between factors, and explained variance was lower.
#Different imputation methods are expected to yield further variations.

#Comment how missingness affects FA
#The results of the FA analysis seems to be very sensitive to the type of FA analysis and the data use. We see a big difference to the original dataset's FA in all three methods, compared to the missing dta (artifically deleted 30% of the data). I expect for the imputations to result in very different FA as well.

# Missing LM
lm_model_complete <- lm(price ~ carat + cut + color + clarity + depth + table, data = df_missing)
summary(lm_model_complete)

lm_model_complete <- lm(price ~ carat + cut + color + clarity + depth + table, data = df)
summary(lm_model_complete)


### 2. Compute confidence interval for the mean for all numeric variables using an appropriate method for missing values treatment

# Imputation using MICE ----
# --- Multiple Imputation using Chained Equations (mice) with PMM ---  
# We use 'mice' to handle missing data and impute realistic values.  
# PMM (Predictive Mean Matching) ensures imputed values are plausible by matching to observed data.  
# Although our missing data mechanism is MCAR (Missing Completely At Random),  
# PMM is chosen to preserve the overall distribution and multivariate relationships,  
# which is important for downstream analyses like PCA or regression.  
# Note: PMM may induce artificial correlations if data were originally independent,  
# so we will check post-imputation correlations to assess any such effects.  

# Imputing using MICE with predictive mean matching (pmm)
imp <- mice(df_missing_num, m=5, method='pmm', seed=500)
print(summary(imp))

# Complete the dataset with first imputed set
df_imputed_mice <- complete(imp, 1)
str(df_imputed_mice)



##comparing imputed and observed distributions for numerical variables (several options)
bwplot(df_imputed_mice) 
densityplot(df_imputed_mice)
stripplot(df_imputed_mice)
xyplot(miceImp, G91 ~ age | .imp, pch = 20, cex = 1.4) #scatterplot
xyplot(miceImp, G91 ~ G92 | .imp, pch = 20, cex = 1.4) #scatterplot




# Correlation Matrix & PCA Analysis
corMice<-cor(df_imputed_mice,use="complete.obs")
corMice
pa_mice<-princomp(covmat=corMice)
summary(pa_mice)
loadings(pa_mice)
plot(pa_df,type="lines")

# Scree plot
plot(pca_res$sdev^2, type='b', main='Scree Plot', ylab='Eigenvalues', xlab='Component Number')


## Perform a multivariate method (e.g. factor analysis, multiple regression …) on the MICE imputed data
# Factor analysis using the optimal number of factors 
#optimal_factors <- fa_parallel_results$nfact  # Extract the optimal number of factors
#optimal_factors
fa.parallel(df_imputed_mice, fa="fa", n.iter=100, show.legend=TRUE, main="Parallel Analysis Scree Plot for FA")
#The optimal factors calculation says 5, however, the Scree plot has a cutoff at 4.

# FA with 5 factors, unrotated
fa_res <- fa(df_imputed_mice, nfactors=5, rotate="none")
print(fa_res$loadings)
fa.diagram(fa_res)
#No Rotation: While it's simple, the unrotated solution doesn't provide clear, distinct factors, and the loadings are spread across multiple factors.

# Varimax (orthogonal) rotation
fa_res_varimax <- fa(df_imputed_mice, nfactors=5, rotate="varimax")
print(fa_res_varimax$loadings)
fa.diagram(fa_res_varimax)
#Varimax (Orthogonal): Offers clearer, more interpretable factors by assuming that factors are uncorrelated. It's a good choice if you want easily interpretable, independent factors.

#Based on scree plot and parallel analysis, 6 factors seemed optimal, but solutions with 5, 6, and 7 factors were compared. The 7-factor solution with oblimin rotation was chosen because it showed the best simple structure and model fit, with interpretable, distinct factors and correlated dimensions — which makes sense given the variables (e.g., carat, price, cut). Though varimax (orthogonal) was tested, oblimin (oblique) is more appropriate here as factors are likely related.
# Oblimin (oblique) rotation if factors are expected to correlate, took 6 for better model fit 
fa_res_oblimin <- fa(df_imputed_mice, nfactors=5, rotate="oblimin")
print(fa_res_oblimin$loadings)
fa.diagram(fa_res_oblimin)
# MR1 reflects the physical properties of diamonds that impact price and quality.
# Clarity and color load highly on MR4 and MR5, which aligns with how these characteristics also correlate with price and overall diamond quality.
# The cumulative variance explained by the first few factors is quite high (83.5%), which suggests that the factors captured are meaningful and account for most of the variability in the data.
# The correlations between MR1, price, carat, and other physical properties make sense in the context of diamond pricing—where size, cut, clarity, and color are all closely linked to value.

# Example of Multiple Regression with imputed data
lm_model_mice <- lm(price ~ carat + cut + color + clarity + depth + table, data = df_imputed_mice)
summary(lm_model_mice)

# Observation: PMM might be too "heavy-duty" — because it's a semi-parametric method relying on predictive modeling, which can create correlations that didn't exist.


# 3. Imputation using EM ---
# Comparison of Imputation Methods:
# PMM (Predictive Mean Matching)/MICE is suitable for MCAR data, preserves realistic values, 
# has a moderate risk of introducing artificial correlations, handles mixed data, 
# but does not assume normality.  
# 
# EM (Expectation-Maximization) is also suitable for MCAR data but does not naturally 
# preserve realistic values as well as PMM. It has a low risk of artificial correlations, 
# only works with numeric data, and assumes normality.  
# 
# missForest is a non-parametric method that works well with MCAR data, preserves realistic 
# values, has a low to moderate risk of artificial correlations, handles mixed data, 
# but does not assume normality.

# Prepare the data for EM algorithm
dataPrep <- prelim.norm(as.matrix(df_missing))  # Preliminary normalization
thetahat <- em.norm(dataPrep)  # Compute MLE for the parameters
corTrustEM <- getparam.norm(dataPrep, thetahat, corr = TRUE)$r  # Estimated correlation matrix

# PCA on the estimated correlation matrix
pcEM <- princomp(covmat = corTrustEM)
summary(pcEM)
loadings(pcEM)
plot(pcEM, type = "lines")

# EM algorithm - imputations
rngseed(1234567)  # Set a random seed for reproducibility
impEM <- imp.norm(s = dataPrep, theta = thetahat, x = as.matrix(df_missing))  # Perform imputation

# PCA on the imputed data
pcEMimp <- princomp(covmat = cor(impEM))
summary(pcEMimp)
loadings(pcEMimp)
plot(pcEMimp, type = "lines")


### 4. Imputation using missForest ----
# Alternative imputation method: Since Mice does not have the best results for our MAR dataset, missForest is a non-parametric imputation method that uses a random forest model to predict missing values based on other available values. It works well for datasets where relationships between variables are non-linear and complex. It’s particularly useful when you have a mix of continuous and categorical variables and suspect that the missingness is MAR (Missing at Random).
#missForest handles MCAR well, is good for mixed variables (numeric + ordinal/categorical) — cut, color, clarity. is less likely to force relationships than PMM, doesn’t rely on normality assumption (unlike EM), is easier to apply in practice than EM for complex datasets.
# Re-load and prepare dataset
data("diamonds")
df <- diamonds
str(df)


set.seed(123)
mf_result <- missForest(df_missing)
df_missing$cut <- as.factor(df_missing$cut)
df_missing$color <- as.factor(df_missing$color)
df_missing$clarity <- as.factor(df_missing$clarity)

# See the structure of your dataset
str(df_missing)


# Extract imputed data
df_imputed <- mf_result$ximp

# Optional: Check out-of-bag error (OOB estimate of imputation error)
print(mf_result$OOBerror)

# Optional: Compare correlations before/after
cor_original <- cor(df_complete, use = "complete.obs")  # before missingness
cor_imputed <- cor(df_imputed, use = "complete.obs")   # after imputation

# Compare visually
corrplot(cor_original, title = "Original Correlations")
corrplot(cor_imputed, title = "Imputed Correlations")

# Impute missing data using missForest 
imputed_data_mf <- missForest(df_missing)$ximp

# Compute correlation matrix for original (complete case) data
corTrustComp_complete <- cor(df_missing, use = "complete.obs")

# Perform PCA for the complete case data
pcComp_complete <- princomp(covmat = corTrustComp_complete)
summary(pcComp_complete) # PCA summary for complete case
loadings(pcComp_complete) # Loadings for complete case

# Compute correlation matrix for the imputed data
corTrustComp_imputed <- cor(imputed_data_mf, use = "complete.obs")

# Perform PCA for the imputed data
pcComp_imputed <- princomp(covmat = corTrustComp_imputed)
summary(pcComp_imputed) # PCA summary for imputed data
loadings(pcComp_imputed) # Loadings for imputed data

# Confidence intervals for mean of numeric variables
# For complete data
apply(df_missing, 2, function(x) t.test(x, conf.level = 0.95)$conf.int)

# For imputed data
apply(imputed_data_mf, 2, function(x) t.test(x, conf.level = 0.95)$conf.int)

# Factor Analysis on Complete Data
fa_res_complete <- fa(df_missing, nfactors = 6, rotate = "oblimin")
print(fa_res_complete$loadings)
fa.diagram(fa_res_complete)

# Factor Analysis on Imputed Data
fa_res_imputed <- fa(imputed_data_mf, nfactors = 6, rotate = "oblimin")
print(fa_res_imputed$loadings)
fa.diagram(fa_res_imputed)


### 5. Comparison across Mice, EM and MissForest
# Compare these results to the results obtained using complete/available (choose one) case scenario.

# Means & CI	Means, width of CI, impact of missingness on estimates
cbind

# Correlations	Strength/structure of variable relationships

# PCA/FA Loadings	Which variables load on what (are components/factors similar?)

#Comparison to df_missing and original dataset
fa_res_oblimin <- fa(df_missing, nfactors=6, rotate="oblimin")
print(fa_res_oblimin$loadings)
fa.diagram(fa_res_oblimin)
fa_res_oblimin <- fa(df, nfactors=6, rotate="oblimin")
print(fa_res_oblimin$loadings)
fa.diagram(fa_res_oblimin)
# This inconsistency shows imputation can be tricky when dealing with MCAR because while I was filling in random values, it could distorted the factor loadings because I was assuming the imputed values are representative of the real distribution.
# This highlights the imputation process can lead to distortions in results, especially when the missingness is MCAR but the imputation method is not perfectly aligned with the real underlying patterns

# Compare PCA and loadings 
loadings(df_imputed) # Loadings for complete case
print(loadings)

#compare loadings for first component
(loadComp<-cbind(comp=pcComp$loadings[,1], avail=pcAvail$loadings[,1], EM=pcEM$loadings[,1], EMimp=pcEMimp$loadings[,1]))

# compare eigenvalues
eigenVal<-cbind(comp=pcComp$sdev, avail=pcAvail$sdev, EM=pcEM$sdev, EMimp=pcEMimp$sdev)^2
matplot(eigenVal,type="o")
eigenVal

#FA

# Regression coefficients	Stability and strength of predictors, R²
# Create linear models for the imputed data and complete case data (example: linear regression on 'price')
lmLRcomp <- lm(price ~ carat + cut + color + clarity + depth + table + x + y + z, data = df_missing_num)
summary(lmLRcomp)

# Perform regression on imputed data
lmLRmice0 <- with(data = miceImp, exp = lm(price ~ 1))  # Model without any predictors
lmLRmice <- with(data = miceImp, exp = lm(price ~ carat + cut + color + clarity + depth + table + x + y + z))  # Full model
lmLRmiceNoF5 <- with(data = miceImp, exp = lm(price ~ carat + cut + color + clarity + depth + table + x + y + z))  # Example without 'F5'

# Pool results from the imputed datasets
library(mitools)
pooled_mice <- pool(lmLRmice)

# Compare models using the pooled results
D1(lmLRmice, lmLRmice0)  # Comparison between full model and intercept-only model
round(summary(pooled_mice), 3)  # Summary of pooled regression results

# Compare models using Wald test
pool.compare(lmLRmice, lmLRmice0, method = "Wald")
pool.compare(lmLRmice, lmLRmiceNoF5, method = "Wald")

# Compute R-squared for the pooled models
pool.r.squared(lmLRmice, adjusted = FALSE)  # Unadjusted R-squared
pool.r.squared(lmLRmice, adjusted = TRUE)   # Adjusted R-squared

### 6. Interpret the results

# We addressed missing data using (method, e.g., missForest, EM). For comparison, we also analyzed complete cases. Confidence intervals for means and multivariate methods (PCA/FA/regression) were performed on both datasets to assess the impact of missing data treatment on statistical conclusions."

# Comparison of Imputation methods
# Method	Suitable for MCAR?	Preserves Realistic Values?	Risk of Artificial Correlations	Handles Mixed Data	Assumes Normality?
# PMM	✅ Yes	✅ Yes	⚠️ Moderate	✅ Yes	❌ No
# EM	✅ Yes	⚠️ Not as naturally	✅ Low	❌ No (numeric only)	✅ Yes
# missForest	✅ Yes	✅ Yes (non-parametric)	⚠️ Low/Moderate	✅ Yes	❌ No

# Results:  "Means and confidence intervals were (similar/different) between complete and imputed data. PCA/FA revealed (similar/different) factor/component structures, with eigenvalues/loadings (stable/unstable) across methods. Correlations and regression coefficients (changed/remained consistent), suggesting (impact/no impact) of missing data on analysis."
# How to interpret and report
Means + CI:
  
  "Using complete case analysis, the mean of 'price' was X with 95% CI [a, b], based on N=..."
"Using imputation via missForest, the mean was Y with CI [c, d], based on N=total rows."
Interpretation: See if mean shifts, CIs narrow — what does that say about missing data?
  PCA/FA:
  
  "We identified 3 components/factors in both cases. The first component (e.g., 'Size') explained X% of variance in complete case and Y% in imputed data."
Loadings comparison: Are the same variables still strongly related to the same factors/components?
  Interpretation: Did imputation stabilize or change structure? What does that suggest about data and missingness?
  Correlations:
  
  "Correlations between 'carat' and 'price' were r = .8 in complete case, and r = .85 after imputation."
Interpretation: Did imputation reveal hidden relationships that were lost due to missingness?
  Regression:
  
  "Regression predicting 'price' from 'carat', 'cut', etc., had R² = X in complete case and R² = Y after imputation."
Interpretation: Does using all available information improve predictive power?