
library(knitr)
library(pander)
library(foreign)
library(psych)
library(car)
library(lattice)
library(ggplot2)

data <- read.table("(your_path)dataBIG5.csv", header = TRUE, sep = "\t")  # Data from the BIG 5 personality test
summary(data)

data <- data[, 28:47]  # Focus on agreeableness and conscientiousness
rownames(data) <- 1:nrow(data)

# Recoding Likert scale data
recod <- function(x) {
  x <- factor(x)
  levels(x) <- c("5", "4", "3", "2", "1", "0")
  as.numeric(as.character(x))
}

# Recode selected columns
data$A1 <- recod(data$A1)
data$A3 <- recod(data$A3)
data$A5 <- recod(data$A5)
data$A7 <- recod(data$A7)
data$C2 <- recod(data$C2)
data$C4 <- recod(data$C4)
data$C6 <- recod(data$C6)
data$C8 <- recod(data$C8)

# PCA analysis on recoded data
data.pca <- prcomp(data[1:200, ], center = TRUE, scale. = TRUE)
biplot(data.pca)

# Reload data
data <- read.table("C:/Users/Bibi Sophia/Documents/Informatics/data/dataBIG5.csv", header = TRUE, sep = "\t")
data <- data[, c(2, 4, 7, 28:47)]  # Selecting columns for age, gender, country, agreeableness, and conscientiousness
data <- data[data$age %in% 5:70, ]
data <- data[data$country %in% c("NL", "NO", "SI"), ]
data <- data[data$gender %in% c(1, 2), ]

# Convert categorical variables
data$spol <- factor(data$gender)
data$drzava <- factor(data$country)
data$starost <- as.numeric(data$age)

# Summary statistics
pander(describe(data))
pander(summary(data))

# Interesting graphs
ggplot(data) +
  geom_boxplot(aes(x = drzava, y = starost, fill = drzava)) +
  labs(x = 'Age') +
  ggtitle("Age by Country") +
  theme_bw() +
  theme(axis.text.x = element_text(face = 'bold', size = 6, angle = 45, hjust = 1))

ggplot(data) +
  geom_boxplot(aes(x = drzava, y = starost, fill = spol)) +
  labs(x = 'Age') +
  ggtitle("Age by Country and Gender") +
  theme_bw() +
  theme(axis.text.x = element_text(face = 'bold', size = 6, angle = 45, hjust = 1))

# Summary and distribution of age
summary(data$starost)
table(is.na(data$starost))

ggplot(data, aes(x = starost)) + 
  geom_histogram(binwidth = 5, colour = "black", fill = "white") +
  geom_vline(aes(xintercept = mean(starost, na.rm = TRUE)), color = "red", linetype = "dashed", size = 1)

# Test for normality of numeric variables
qqPlot(data$starost)  # Visual check for normality
data$starost <- jitter(data$starost)
ks.test(data$starost, y = 'pnorm')  # Kolmogorov-Smirnov test for normality

# Test associations between variables

# Age vs Gender: Mann-Whitney-Wilcoxon test
tbl1 <- table(data$starost, data$spol)
ggplot(data) +
  geom_boxplot(aes(x = spol, y = starost, fill = spol)) +
  labs(x = 'Age') +
  ggtitle("Age by Gender") +
  theme_bw() +
  theme(axis.text.x = element_text(face = 'bold', size = 6, angle = 45, hjust = 1))

wilcox.test(tbl1)

# Age vs Country: Mann-Whitney-Wilcoxon test
tbl2 <- table(data$starost, data$drzava)
wilcox.test(tbl2)

# Age vs Age: Pearson correlation test
cor(data$starost, data$starost, method = "pearson", use = "complete.obs")

# Categorical association: Chi-square test between Gender and Country
tblc <- table(data$drzava, data$spol)
assocplot(tblc, col = c("black", "red"), space = 0.3)
chisq.test(data$drzava, data$spol)

# Test assumptions for ANOVA

# Levene's test for homogeneity of variances
leveneTest(starost ~ drzava, data = data, center = mean)

# Welch ANOVA for unequal variances
oneway.test(starost ~ drzava, data = data)

# Post-hoc test for unequal variances
pairwise.t.test(x = data$starost, g = data$drzava, p.adjust.method = "holm", pool.sd = FALSE)

# If assuming equal variances, perform standard ANOVA
fit <- aov(starost ~ drzava, data = data)
summary(fit)
