library(lmerTest)
library(haven)
library(tidyverse)
library(RColorBrewer)
library(lme4)
library(lmtest)
library(flexplot)
library(DHARMa)

data <- read.csv("lme_data_tableau_new_actions.csv")

# Assuming you have already fitted the full model as:
fullModel <- lmer(Keep ~ Phase + (1 | Users) + (1 | Datasets) + (1|task), data = data)

# Now, fit the reduced model without the fixed effect of Phase
reducedModel <- lmer(Keep ~ 1 + (1 | Users) + (1 | Datasets) + (1|task), data = data)

# Perform the likelihood ratio test
# lrtResult <- anova(reducedModel, fullModel)
lrtResult <- lrtest(fullModel, reducedModel)

# Print the results
print(lrtResult)

shapiro.test(resid(fullModel))