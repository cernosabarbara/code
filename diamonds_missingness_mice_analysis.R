library(ggplot2)
library(mice)
library(psych)
library(norm)
library(mitools)

data("diamonds")
df <- diamonds[, c("price", "carat", "depth", "cut", "color")]  # 5 vars: 3 numeric, 2 categorical
df$cut <- as.numeric(factor(df$cut, ordered = TRUE))
df$color <- as.numeric(factor(df$color, ordered = TRUE))

# Introduce 30% MCAR missing data
set.seed(123)
df_missing <- df
total_elements <- prod(dim(df_missing))  
nMiss <- round(total_elements * 0.3)    
if (nMiss > total_elements) nMiss <- total_elements  

# Convert to vector, introduce missingness
df_vec <- as.vector(as.matrix(df_missing))  
df_vec[sample(total_elements, nMiss)] <- NA 
df_missing <- as.data.frame(matrix(df_vec, nrow = nrow(df_missing), ncol = ncol(df_missing)))  
colnames(df_missing) <- colnames(df) 

missing_rate <- sum(is.na(df_missing)) / total_elements 
print(paste("Missing rate:", round(missing_rate, 2)))

##############################################################################
# Speculate on Missing Data Mechanism
# MCAR: We randomly introduced 30% missingness, independent of any variable.

##############################################################################
# Task 1: Confidence Intervals for Numeric Variables Means
# Complete case analysis
num_vars <- c("price", "carat", "depth")
ci_complete <- sapply(df[num_vars], function(x) {
  x <- na.omit(x)
  mean_x <- mean(x)
  se_x <- sd(x) / sqrt(length(x))
  c(mean_x - 1.96 * se_x, mean_x + 1.96 * se_x)  
})
colnames(ci_complete) <- num_vars
print(round(t(ci_complete), 2))

# Missing data complete case analysis 
ci_missing <- sapply(df_missing[num_vars], function(x) {
  x <- na.omit(x)
  mean_x <- mean(x)
  se_x <- sd(x) / sqrt(length(x))
  c(mean_x - 1.96 * se_x, mean_x + 1.96 * se_x)  # 95% CI
})
colnames(ci_missing) <- num_vars
print(round(t(ci_missing), 2))

# MICE imputation (PMM) for CI
imp <- mice(df_missing, m = 5, method = "pmm", seed = 500, printFlag = FALSE)
mi_means <- sapply(num_vars, function(var) {
  means <- sapply(1:5, function(i) mean(complete(imp, i)[[var]]))
  vars <- sapply(1:5, function(i) var(complete(imp, i)[[var]]) / nrow(df_missing))
  mi_result <- MIcombine(results = as.list(means), variances = as.list(vars))
  c(mi_result$coefficients - 1.96 * sqrt(mi_result$variance), 
    mi_result$coefficients + 1.96 * sqrt(mi_result$variance))
})
colnames(mi_means) <- num_vars
print(round(t(mi_means), 2))
# Overall: Missing data widens CIs (especially for price), while MICE restores them closer to Complete, validating imputation’s benefit under MCAR.
# Price: Complete is slightly narrower than Missing, showing reduced precision with 30% missing data. MICE aligns closely with Complete, suggesting effective recovery of the mean.
# Carat: All three are nearly identical.
# Depth: Complete has the narrowest range, Missing slightly wider, and MICE matches Missing’s width but centers like Complete, showing imputation preserves precision.

##############################################################################
# Task 2: EM Imputation
dataPrep <- prelim.norm(as.matrix(df_missing))  
thetahat <- em.norm(dataPrep)  
rngseed(1234567) 
impEM <- imp.norm(s = dataPrep, theta = thetahat, x = as.matrix(df_missing))
imputed_data_em <- as.data.frame(impEM)
colnames(imputed_data_em) <- colnames(df_missing)  

##############################################################################
# Task 3: Linear Regression Comparison
# Define formula
formula <- price ~ carat + depth + cut + color

# Linear models for each dataset
lm_complete <- lm(formula, data = df)
lm_missing <- lm(formula, data = df_missing)  
df_imputed_mice <- complete(imp, 1) 
lm_mice <- lm(formula, data = df_imputed_mice) 
lm_em <- lm(formula, data = imputed_data_em)
lm_mice_pooled <- with(imp, lm(price ~ carat + depth + cut + color))  
pooled_mice <- pool(lm_mice_pooled)
# Comment: lm_missing shifts slightly with fewer cases; lm_mice exaggerates some effects (e.g., intercept, depth) from one imputation; lm_mice_pooled balances these across 5 imputations, aligning closer to complete case for key predictors like carat.

# Extract coefficients
coef_comp <- cbind(Complete = coef(lm_complete), Missing = coef(lm_missing), 
                   MICE_Single = coef(lm_mice), MICE_Pooled = pooled_mice$pooled$estimate, 
                   EM = coef(lm_em))
print(round(coef_comp, 3))

# Extract R-squared values
rsq_comp <- c(Complete = summary(lm_complete)$r.squared, Missing = summary(lm_missing)$r.squared, 
              MICE_Single = summary(lm_mice)$r.squared, MICE_Pooled = pool.r.squared(lm_mice_pooled)[1], 
              EM = summary(lm_em)$r.squared)
print(round(rsq_comp, 3))
# Intercept: Very wide range; EM lowest, MICE_Single highest, MICE_Pooled moderates between imputations, Complete as baseline.
# Carat: Stable, highest in Missing, showing robust predictive power across methods. Depth: More negative in MICE vs. Complete and EM, suggesting imputation amplifies it. Cut & Color: Consistent.
# R-squared: Tight range, EM highest, MICE_Pooled lowest, indicating model fit is robust to missingness and imputation.
# Coefficients show more variability (especially intercept, depth) than R-squared, reflecting multivariate sensitivity to missing data treatment, while high R-squared consistency suggests overall explanatory power holds steady.

# Extended Mice model comparisons
lm_mice0 <- with(imp, lm(price ~ 1))  # Intercept-only model
lm_mice_reduced <- with(imp, lm(price ~ carat + depth + color)) 
print(D1(lm_mice_pooled, lm_mice0))
print(D1(lm_mice_pooled, lm_mice_reduced))
print(round(pool.r.squared(lm_mice_pooled, adjusted = FALSE)[1], 3))
print(round(pool.r.squared(lm_mice_pooled, adjusted = TRUE)[1], 3))
# Intercept: reduced is much higher.
# R-squared: reduced lower, reflecting cut’s contribution.
# Reduced model shifts coefficients (larger intercept, depth) due to cut’s absence, but full model’s stability and D1 significance highlight cut’s importance.


##############################################################################
# Task 4: PCA Comparison
# PCA on each dataset
pca_complete <- principal(df, nfactors = 5, rotate = "oblimin")
pca_missing <- principal(df_missing, nfactors = 5, rotate = "oblimin")
pca_mice <- principal(df_imputed_mice, nfactors = 5, rotate = "oblimin")
pca_em <- principal(imputed_data_em, nfactors = 5, rotate = "oblimin")
# PCA structure is robust across complete, missing, and imputed datasets, with price and carat driving TC1; minor variations reflect imputation effects, but overall consistency validates MCAR handling.
# Correlations: TC1-TC3 steady, TC1-TC4 slightly more negative in MICE, others negligible differences.
# Variance: TC1 loadings nearly identical, minor increase in Missing and EM; other components stable.

# Compare loadings for first component
loadings_comp <- cbind(Original = pca_complete$loadings[,1], Missing = pca_missing$loadings[,1], 
                       MICE = pca_mice$loadings[,1], EM = pca_em$loadings[,1])
print(round(loadings_comp, 3))
# TC1 dominated by price and carat across all, with MICE slightly lower; depth, cut, color near 0, showing consistent structure.

# Compare eigenvalues
eigenvalues <- cbind(Complete = pca_complete$values, Missing = pca_missing$values, 
                     MICE = pca_mice$values, EM = pca_em$values)
colnames(eigenvalues) <- c("Complete Case", "Missing Case", "MICE Imputed", "EM Imputed")
print(round(eigenvalues, 3))
# TC1 slightly higher in MICE vs. others, indicating imputation may inflate primary variance slightly; others consistent.

# Plot eigenvalue comparison
matplot(eigenvalues, type = "b", pch = 1, col = c("red", "blue", "green", "black"), lwd = 2, lty = 1,
        xlab = "Factor Number", ylab = "Eigenvalue", main = "Eigenvalue Comparison")
legend("topright", legend = colnames(eigenvalues), col = c("red", "blue", "green", "black"), lty = 1, pch = 1)


##############################################################################
# 5: Interpretation of result

# We addressed missing data using Predictive Mean Matching (PMM) via MICE and Expectation-Maximization (EM). For comparison, we also analyzed the complete cases. Confidence intervals for means and multivariate methods (PCA/regression) were performed on both datasets to assess the impact of missing data treatment on statistical conclusions.
# Recap: Under 30% MCAR missingness, CIs showed minor widening in the missing dataset, with MICE and EM effectively recovering precision close to the complete case. 
# Regression revealed a larger coefficient variability (e.g., intercept, depth) than R-squared, highlighting multivariate sensitivity, but imputation kept core relationships intact with the original. 
# PCA: Structure held steady, validating imputation under MCAR, with price and carat dominating variance and minimal imputation effects. 
# Overall, both PMM and EM handled MCAR well, preserving statistical integrity across univariate and multivariate analyses, with PMM’s pooled approach offering balanced estimates.





