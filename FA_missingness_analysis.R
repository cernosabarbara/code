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

#Perform FA on the original dataset:
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
#Comment how missingness affects FA
#The FA structure was highly sensitive to missing data.
#After artificially deleting 30% of the data, factor structures changed significantly.
#Variables shifted between factors, and explained variance was lower.
#Different imputation methods are expected to yield further variations.

#The results of the FA analysis seems to be very sensitive to the type of FA analysis and the level of data missing. We see a big difference to the original dataset's FA in all three methods, compared to the missing dta (artifically deleted 30% of the data). I expect for the imputations to result in very different FA as well.
