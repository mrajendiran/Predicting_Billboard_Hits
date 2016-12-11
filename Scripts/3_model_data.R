setwd('/Users/bradyfowler/Documents/Fall Semester/Modeling_6021/predicting_billboard_success')
suppressMessages({
  library(caret); library(DAAG); library(car); library(PerformanceAnalytics); library(dplyr); library(readr); 
  library(glmnet); library(htmlTable); library(ggthemes); library(boot); library(lubridate); library(randomForest); 
  library(party); library(ggplot2); library(scales); library(ROCR); library(lubridate)
})

set.seed(12345)

#######################################################################
#
# Load Manually Created Functions
#
#######################################################################

# check CV accuracy function
cv.glm.accuracy <- function(model, df, N=10) {
  stopifnot(nrow(df) %% N == 0)
  df    <- df[order(runif(nrow(df))), ]
  bins  <- rep(1:N, nrow(df) / N)
  folds <- split(df, bins)
  all.accuracy <- vector()
  all_fold = list(1:N)[[1]]
  for (i in 1:length(folds)) {
    current.fold = all_fold[-which(all_fold==7)]
    train <- dplyr::bind_rows(folds[current.fold])
    test  <- folds[-current.fold][[1]]
    current.model <- glm(model, family = binomial(link = "logit"), data = train)
    curr.predict  <-ifelse(predict(current.model, newdata = test, type="response") >.5,1,0)
    curr.accuracy <- sum(curr.predict == test$pred.hit.top.ten) / nrow(test)
    all.accuracy <- c(all.accuracy, curr.accuracy[[1]]) 
  }
  return(mean(all.accuracy))
}

# ROC curve function
roc_curve <- function(model, test.set, test.set.response) {
  # output predictions and FPR/TPR rate
  predictions        = predict(model, newdata = test.set, type="response")
  pred.fpr.tpr       = prediction(predictions, test.set.response)
  compare.perf       = performance(pred.fpr.tpr, "tpr", "fpr")
  auc.perf           = performance(pred.fpr.tpr, measure="auc")
  roc.vals           = data.frame(cbind(compare.perf@x.values[[1]], compare.perf@y.values[[1]]))
  colnames(roc.vals) = c("fp", "tp")
  
  # plot curve
  ggplot(roc.vals, aes(x=fp, y=tp, colour="#2980b9")) + 
    labs(x=compare.perf@x.name, y=compare.perf@y.name) +
    scale_x_continuous(labels = percent, limits = c(0,1)) + scale_y_continuous(labels = percent, limits = c(0,1)) + 
    geom_abline(aes(intercept=0, slope=1, colour="#34495e")) + guides(colour = FALSE) + coord_equal() +
    ggtitle(bquote(atop(.("ROC Curve"), atop(italic(.(paste("AUC: ", round(auc.perf@y.values[[1]],digits=2)))), "")))) + geom_line(size=1.2) +
    theme_fivethirtyeight()
}

auc_values <- function(model, test.set, test.set.response) {
  # output predictions and FPR/TPR rate
  predictions        = predict(model, newdata = test.set, type="response")
  pred.fpr.tpr       = prediction(predictions, test.set.response)
  auc.values         = round(performance(pred.fpr.tpr, "auc")@y.values[[1]], digits=3)
  return(auc.values)
}

roc_values <- function(model, test.set, test.set.response) {
  # output predictions and FPR/TPR rate
  predictions        = predict(model, newdata = test.set, type="response")
  pred.fpr.tpr       = prediction(predictions, test.set.response)
  compare.perf       = performance(pred.fpr.tpr, "tpr", "fpr")
  auc.perf           = performance(pred.fpr.tpr, measure="auc")
  roc.vals           = data.frame(cbind(compare.perf@x.values[[1]], compare.perf@y.values[[1]]))
  colnames(roc.vals) = c("fp", "tp")
  return(roc.vals)
}

#######################################################################
#
# load raw data
#
#######################################################################
raw <- readRDS('Raw Data/final.analysis.RDS')

#######################################################################
#
# test and training
#
#######################################################################
train <- raw %>% 
  filter(year(birthday)<2014) %>% 
  select(-title, -art_0, -artist, -artist_id, -song_id, -spotifyid, -birthday, -pred.top.rank, -weeks.at.top.ten, -total.weeks) %>% 
  na.omit()

test <- raw %>% 
  filter(year(birthday)>=2014 & total.weeks > 12) %>% 
  filter(!(song_id %in% c('5955', '7538', '4635', '8040', '2000', '6489', 
                          '5642', '3407', '3010', '2501', '6271', '4014', '6020', '1994'))) %>%  # manually remove tracks identified to be re-entries in test data
  select(-title, -art_0, -artist, -artist_id, -song_id, -spotifyid, -birthday, -pred.top.rank, -weeks.at.top.ten, -total.weeks) %>% 
  na.omit()

# cast response as factors
train$pred.hit.top.ten <- as.factor(train$pred.hit.top.ten)
test$pred.hit.top.ten  <- as.factor(test$pred.hit.top.ten)

# remove NA's 
train <- na.omit(train)
test  <- na.omit(test)

# print summary
train %>% summary()

#######################################################################
#
# Logistic Regression Model - All Variables
# 
#######################################################################

# Model Selection #####################################################

# use hybrid stepwise selection
s.null <- glm(pred.hit.top.ten~1, data=train, family = binomial(link = "logit"))
s.full <- glm(pred.hit.top.ten~., data=train, family = binomial(link = "logit"))

both <- step(s.null, scope=list(lower=s.null, upper=s.full), direction="both")
summary(both)
#                           Estimate Std. Error z value Pr(>|z|)    
# (Intercept)            -2.442e+00  8.973e-01  -2.721  0.00650 ** 
# danceability            1.893e+00  3.428e-01   5.523 3.33e-08 ***
# prior.not.top.ten.hits -6.034e-02  8.611e-03  -7.007 2.43e-12 ***
# mode                   -3.535e-01  8.533e-02  -4.143 3.43e-05 ***
# prior.top.ten.hits      1.626e-01  1.438e-02  11.306  < 2e-16 ***
# acousticness           -7.915e-01  2.702e-01  -2.930  0.00339 ** 
# energy                 -1.623e+00  4.021e-01  -4.037 5.41e-05 ***
# loudness                7.516e-02  2.764e-02   2.719  0.00656 ** 
# duration_ms             2.113e-06  1.002e-06   2.109  0.03493 *  
# valence                 4.788e-01  2.357e-01   2.031  0.04222 *  
# avg.age.not.top.ten    -3.543e-02  2.317e-02  -1.529  0.12624    
# time_signature          2.637e-01  1.806e-01   1.460  0.14428    
# (Dispersion parameter for binomial family taken to be 1)
# Null deviance: 4223.9  on 4866  degrees of freedom
# Residual deviance: 3875.4  on 4855  degrees of freedom
# AIC: 3899.4
# Number of Fisher Scoring iterations: 7

# Remove avg.age.not.top.ten and time_signature due to insignificance
both <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + 
              mode + prior.top.ten.hits + acousticness + energy + loudness + 
              duration_ms + valence, family = binomial(link = "logit"), data = train)

plot(both)
res <- residuals(both, type = "deviance")
plot(predict(both), res, xlab="Fitted values", ylab = "Residuals", ylim = max(abs(res)) * c(-1,1))
abline(h = 0, lty = 2)

# Testing ###################################################################

cv.glm.accuracy(formula(both), train, N=31)
# 0.8598726 CV Accuracy Score using Training Set

cv.glm(train, both, K=10)$delta
# 0.1233321 0.1232959 prediction error rate and adjusted prediction error rate

# check for raw prediction accuracy (pick .2 as the threshold for our response)
both.pred <- ifelse(predict(both, newdata = test, type="response") >.2,1,0)
sum(test$pred.hit.top.ten == both.pred) / length(both.pred)
# 0.7378641 is raw prediction accuracy

roc_curve(both, test, test$pred.hit.top.ten)

#################################################################################
# Looking at Interaction Terms for the Logistic Regression Model - All Variables
#################################################################################

# look at possible interactions for the both model

add1(both, ~.^2,test="Chisq")
#Single term additions

#Model:
#  pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
#  prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
#  valence + avg.age.not.top.ten + time_signature
#Df Deviance    AIC    LRT Pr(>Chi)   
#<none>                                          3875.4 3899.4                   
#danceability:prior.not.top.ten.hits         1   3872.7 3898.7 2.6950 0.100662   
#danceability:mode                           1   3875.0 3901.0 0.3764 0.539523   
#danceability:prior.top.ten.hits             1   3875.3 3901.3 0.0268 0.870010   
#danceability:acousticness                   1   3875.3 3901.3 0.0973 0.755043   
#danceability:energy                         1   3874.5 3900.5 0.8809 0.347951   
#danceability:loudness                       1   3875.0 3901.0 0.3897 0.532444   
#danceability:duration_ms                    1   3875.4 3901.4 0.0000 0.998422   
#danceability:valence                        1   3874.3 3900.3 1.1137 0.291289   
#danceability:avg.age.not.top.ten            1   3874.6 3900.6 0.7283 0.393437   
#danceability:time_signature                 1   3875.1 3901.1 0.2336 0.628867   
#prior.not.top.ten.hits:mode                 1   3873.9 3899.9 1.4485 0.228770   
#prior.not.top.ten.hits:prior.top.ten.hits   1   3873.4 3899.4 2.0039 0.156898   
#prior.not.top.ten.hits:acousticness         1   3871.0 3897.0 4.4172 0.035579 * 
#prior.not.top.ten.hits:energy               1   3872.9 3898.9 2.4780 0.115447   
#prior.not.top.ten.hits:loudness             1   3868.6 3894.6 6.7947 0.009143 **
#prior.not.top.ten.hits:duration_ms          1   3872.9 3898.9 2.4971 0.114056   
#prior.not.top.ten.hits:valence              1   3871.0 3897.0 4.3205 0.037655 * 
#prior.not.top.ten.hits:avg.age.not.top.ten  1   3874.4 3900.4 0.9773 0.322873   
#prior.not.top.ten.hits:time_signature       1   3874.7 3900.7 0.6953 0.404381   
#mode:prior.top.ten.hits                     1   3875.1 3901.1 0.2724 0.601704   
#mode:acousticness                           1   3873.6 3899.6 1.7381 0.187381   
#mode:energy                                 1   3875.2 3901.2 0.1199 0.729129   
#mode:loudness                               1   3875.3 3901.3 0.0489 0.824914   
#mode:duration_ms                            1   3874.0 3900.0 1.3275 0.249258   
#mode:valence                                1   3874.1 3900.1 1.3132 0.251820   
#mode:avg.age.not.top.ten                    1   3872.4 3898.4 2.9720 0.084717 . 
#mode:time_signature                         1   3875.3 3901.3 0.0418 0.837933   
#prior.top.ten.hits:acousticness             1   3874.6 3900.6 0.8114 0.367714   
#prior.top.ten.hits:energy                   1   3874.7 3900.7 0.6543 0.418570   
#prior.top.ten.hits:loudness                 1   3874.4 3900.4 0.9689 0.324948   
#prior.top.ten.hits:duration_ms              1   3875.0 3901.0 0.4063 0.523878   
#prior.top.ten.hits:valence                  1   3875.0 3901.0 0.3745 0.540556   
#prior.top.ten.hits:avg.age.not.top.ten      1   3875.2 3901.2 0.1191 0.729961   
#prior.top.ten.hits:time_signature           1   3875.3 3901.3 0.0444 0.833023   
#acousticness:energy                         1   3875.4 3901.4 0.0038 0.950670   
#acousticness:loudness                       1   3874.5 3900.5 0.8732 0.350084   
#acousticness:duration_ms                    1   3875.4 3901.4 0.0001 0.994339   
#acousticness:valence                        1   3875.3 3901.3 0.0631 0.801684   
#acousticness:avg.age.not.top.ten            1   3875.2 3901.2 0.1913 0.661817   
#acousticness:time_signature                 1   3875.0 3901.0 0.3265 0.567756   
#energy:loudness                             1   3874.5 3900.5 0.8638 0.352673   
#energy:duration_ms                          1   3871.2 3897.2 4.2022 0.040371 * 
#energy:valence                              1   3874.6 3900.6 0.8176 0.365872   
#energy:avg.age.not.top.ten                  1   3872.6 3898.6 2.8075 0.093823 . 
#energy:time_signature                       1   3874.7 3900.7 0.6751 0.411277   
#loudness:duration_ms                        1   3871.9 3897.9 3.4249 0.064219 . 
#loudness:valence                            1   3874.8 3900.8 0.5946 0.440627   
#loudness:avg.age.not.top.ten                1   3875.0 3901.0 0.3510 0.553531   
#loudness:time_signature                     1   3874.9 3900.9 0.4307 0.511635   
#duration_ms:valence                         1   3874.3 3900.3 1.0853 0.297519   
#duration_ms:avg.age.not.top.ten             1   3875.3 3901.3 0.0211 0.884457   
#duration_ms:time_signature                  1   3874.2 3900.2 1.1331 0.287118   
#valence:avg.age.not.top.ten                 1   3872.6 3898.6 2.7986 0.094349 . 
#valence:time_signature                      1   3875.4 3901.4 0.0072 0.932602   
#avg.age.not.top.ten:time_signature          1   3874.8 3900.8 0.5714 0.449689   

# checking for influential the interaction term - energy:duration_ms 
lr_i1 <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
               prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
               valence + avg.age.not.top.ten + time_signature + energy:duration_ms,
             data=train, family = binomial(link = "logit"))

roc_curve(lr_i1, test, test$pred.hit.top.ten)
cv.glm(train, lr_i1, K=10)$delta
#[1] 0.1233128 0.1232699 -> Performs worse than the both model

# checking for influential the interaction term - prior.not.top.ten.hits:valence 
lr_i2 <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
               prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
               valence + avg.age.not.top.ten + time_signature + prior.not.top.ten.hits:valence,
             data=train, family = binomial(link = "logit"))

roc_curve(lr_i2, test, test$pred.hit.top.ten)
cv.glm(train, lr_i2, K=10)$delta
#[1] 0.1234024 0.1233613 -> Performs worse than the both model

# checking for influential the interaction term - prior.not.top.ten.hits:valence 
lr_i2 <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
               prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
               valence + avg.age.not.top.ten + time_signature + prior.not.top.ten.hits:valence,
             data=train, family = binomial(link = "logit"))

roc_curve(lr_i2, test, test$pred.hit.top.ten)
cv.glm(train, lr_i2, K=10)$delta
#[1] 0.1234024 0.1233613 -> Performs worse than the both model

# checking for influential the interaction term - prior.not.top.ten.hits:loudness 
lr_i3 <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
               prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
               valence + avg.age.not.top.ten + time_signature + prior.not.top.ten.hits:loudness,
             data=train, family = binomial(link = "logit"))

roc_curve(lr_i3, test, test$pred.hit.top.ten)
cv.glm(train, lr_i3, K=10)$delta
#[1] 0.1232172 0.1231778 -> Performs worse than the both model

# checking for influential the interaction term - prior.not.top.ten.hits:acousticness 
lr_i4 <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
               prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
               valence + avg.age.not.top.ten + time_signature + prior.not.top.ten.hits:acousticness,
             data=train, family = binomial(link = "logit"))

roc_curve(lr_i4, test, test$pred.hit.top.ten)
#cv.glm(train, lr_i4, K=10)$delta
#[1] 0.1230844 0.1230420 -> Performs worse than the both model


# adding to the train data frame two new columns: total.hits and probability of top.ten.hits
train %>% 
  mutate(total.hits = prior.top.ten.hits + prior.not.top.ten.hits,
         prob.top.ten.hits = ifelse(total.hits != 0,
                                    prior.top.ten.hits/(prior.top.ten.hits + prior.not.top.ten.hits), 
                                    0)) -> train

test %>% 
  mutate(total.hits = prior.top.ten.hits + prior.not.top.ten.hits,
         prob.top.ten.hits = ifelse(total.hits != 0,
                                    prior.top.ten.hits/(prior.top.ten.hits + prior.not.top.ten.hits), 
                                    0)) -> test

# Now lets look at substitute prior.top.ten.hits and prior.not.top.ten.hits variables for the two new ones created
lr_new <- glm(pred.hit.top.ten ~ danceability + total.hits + mode + 
                prob.top.ten.hits + acousticness + energy + loudness + duration_ms + 
                valence + avg.age.not.top.ten + time_signature + prior.not.top.ten.hits:loudness,
              data=train, family = binomial(link = "logit"))

roc_curve(lr_new, test, test$pred.hit.top.ten)
cv.glm(train, lr_new, K=10)$delta
#[1] 0.1218580 0.1218197 -> Performs worse than the both model

########### look for possible interactions for the lr_new model 

add1(lr_new,~.^2,test="Chisq")

#Single term additions

#Model:
#  pred.hit.top.ten ~ danceability + total.hits + mode + prob.top.ten.hits + 
#  acousticness + energy + loudness + duration_ms + valence + 
#  avg.age.not.top.ten + time_signature + prior.not.top.ten.hits:loudness
#Df Deviance    AIC    LRT Pr(>Chi)   
#<none>                                     3830.0 3856.0                   
#danceability:total.hits                1   3828.4 3856.4 1.5779 0.209063   
#danceability:mode                      1   3829.7 3857.7 0.3182 0.572665   
#danceability:prob.top.ten.hits         1   3829.8 3857.8 0.2405 0.623843   
#danceability:acousticness              1   3829.9 3857.9 0.0815 0.775276   
#danceability:energy                    1   3829.1 3857.1 0.9194 0.337643   
#danceability:loudness                  1   3829.5 3857.5 0.4943 0.482020   
#danceability:duration_ms               1   3830.0 3858.0 0.0094 0.922632   
#danceability:valence                   1   3829.0 3857.0 1.0469 0.306234   
#danceability:avg.age.not.top.ten       1   3829.6 3857.6 0.4348 0.509640   
#danceability:time_signature            1   3829.8 3857.8 0.2093 0.647315   
#total.hits:mode                        1   3828.1 3856.1 1.8603 0.172589   
#total.hits:prob.top.ten.hits           1   3820.9 3848.9 9.1092 0.002543 **
#total.hits:acousticness                1   3829.6 3857.6 0.4165 0.518687   
#total.hits:energy                      1   3825.2 3853.2 4.7608 0.029116 * 
#total.hits:loudness                    1   3821.8 3849.8 8.1976 0.004195 **
#total.hits:duration_ms                 1   3830.0 3858.0 0.0449 0.832116   
#total.hits:valence                     1   3829.8 3857.8 0.1985 0.655946   
#total.hits:avg.age.not.top.ten         1   3830.0 3858.0 0.0283 0.866385   
#total.hits:time_signature              1   3829.8 3857.8 0.2433 0.621830   
#mode:prob.top.ten.hits                 1   3828.3 3856.3 1.7273 0.188753   
#mode:acousticness                      1   3828.8 3856.8 1.1985 0.273622   
#mode:energy                            1   3829.9 3857.9 0.0580 0.809614   
#mode:loudness                          1   3830.0 3858.0 0.0062 0.937160   
#mode:duration_ms                       1   3829.1 3857.1 0.8913 0.345130   
#mode:valence                           1   3828.6 3856.6 1.3506 0.245171   
#mode:avg.age.not.top.ten               1   3826.4 3854.4 3.6358 0.056549 . 
#mode:time_signature                    1   3830.0 3858.0 0.0000 0.995390   
#prob.top.ten.hits:acousticness         1   3829.5 3857.5 0.4551 0.499910   
#prob.top.ten.hits:energy               1   3829.8 3857.8 0.2426 0.622301   
#prob.top.ten.hits:loudness             1   3830.0 3858.0 0.0030 0.956367   
#prob.top.ten.hits:duration_ms          1   3829.7 3857.7 0.2854 0.593215   
#prob.top.ten.hits:valence              1   3830.0 3858.0 0.0070 0.933432   
#prob.top.ten.hits:avg.age.not.top.ten  1   3829.5 3857.5 0.5168 0.472222   
#prob.top.ten.hits:time_signature       1   3826.3 3854.3 3.7432 0.053022 . 
#acousticness:energy                    1   3830.0 3858.0 0.0157 0.900282   
#acousticness:loudness                  1   3829.0 3857.0 0.9691 0.324904   
#acousticness:duration_ms               1   3830.0 3858.0 0.0255 0.873009   
#acousticness:valence                   1   3830.0 3858.0 0.0483 0.825972   
#acousticness:avg.age.not.top.ten       1   3829.7 3857.7 0.3032 0.581895   
#acousticness:time_signature            1   3829.6 3857.6 0.4314 0.511283   
#energy:loudness                        1   3828.4 3856.4 1.6125 0.204143   
#energy:duration_ms                     1   3826.2 3854.2 3.7648 0.052341 . 
#energy:valence                         1   3829.2 3857.2 0.7696 0.380338   
#energy:avg.age.not.top.ten             1   3827.9 3855.9 2.0707 0.150149   
#energy:time_signature                  1   3829.4 3857.4 0.6213 0.430558   
#loudness:duration_ms                   1   3826.7 3854.7 3.3136 0.068710 . 
#loudness:valence                       1   3829.4 3857.4 0.5705 0.450067   
#loudness:avg.age.not.top.ten           1   3829.7 3857.7 0.2933 0.588118   
#loudness:time_signature                1   3829.6 3857.6 0.3688 0.543682   
#duration_ms:valence                    1   3828.9 3856.9 1.1354 0.286618   
#duration_ms:avg.age.not.top.ten        1   3829.5 3857.5 0.4637 0.495913   
#duration_ms:time_signature             1   3828.5 3856.5 1.4933 0.221708   
#valence:avg.age.not.top.ten            1   3827.1 3855.1 2.8502 0.091363 . 
#valence:time_signature                 1   3830.0 3858.0 0.0000 0.997715   
#avg.age.not.top.ten:time_signature     1   3829.4 3857.4 0.5507 0.458013   

# adding influential the interaction term - total.hits:energy 
lr_new_i1 <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
                   prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
                   valence + avg.age.not.top.ten + time_signature + total.hits:energy,
                 data=train, family = binomial(link = "logit"))

roc_curve(lr_new_i1, test, test$pred.hit.top.ten)
cv.glm(train, lr_new_i1, K=10)$delta
#[1] 0.1233594 0.1233133 -> Performs worse than the both model

# adding influential the interaction term - total.hits:loudness 
lr_new_i2 <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
                   prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
                   valence + avg.age.not.top.ten + time_signature + total.hits:loudness,
                 data=train, family = binomial(link = "logit"))

roc_curve(lr_new_i2, test, test$pred.hit.top.ten)
cv.glm(train, lr_new_i2, K=10)$delta
#[1] 0.1231410 0.1231055 -> Performs worse than the both model

# adding influential the interaction term - total.hits:prob.top.ten.hits 
lr_new_i3 <- glm(pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + mode + 
                   prior.top.ten.hits + acousticness + energy + loudness + duration_ms + 
                   valence + avg.age.not.top.ten + time_signature + total.hits:prob.top.ten.hits,
                 data=train, family = binomial(link = "logit"))

roc_curve(lr_new_i3, test, test$pred.hit.top.ten)
#cv.glm(train, lr_new_i3, K=10)$delta
#[1] 0.1232288 0.1231900 -> Performs worse than the both model


# Now let's look at what what model the step fuction returns adding interactions  
both2 <- step(both, ~.^2, test = 'Chisq', direction = 'both')
summary(both2)

#Output
#Call:
#  glm(formula = pred.hit.top.ten ~ danceability + prior.not.top.ten.hits + 
#        mode + prior.top.ten.hits + acousticness + energy + loudness + 
#        duration_ms + valence + avg.age.not.top.ten + time_signature + 
#        prior.not.top.ten.hits:loudness + energy:duration_ms + danceability:prior.not.top.ten.hits + 
#        mode:avg.age.not.top.ten + energy:avg.age.not.top.ten + prior.not.top.ten.hits:mode + 
#        prior.not.top.ten.hits:prior.top.ten.hits + mode:acousticness, 
#      family = binomial(link = "logit"), data = train)

#Deviance Residuals: 
#  Min       1Q   Median       3Q      Max  
#-1.6325  -0.6145  -0.4849  -0.3279   3.1818  

#Coefficients:
#Estimate Std. Error z value Pr(>|z|)    
#(Intercept)                                8.076e-01  1.695e+00   0.477 0.633662    
#danceability                               1.459e+00  3.946e-01   3.698 0.000217 ***
#prior.not.top.ten.hits                    -7.671e-02  4.436e-02  -1.729 0.083785 .  
#mode                                       3.303e-01  4.770e-01   0.692 0.488693    
#prior.top.ten.hits                         1.844e-01  2.150e-02   8.577  < 2e-16 ***
#acousticness                              -3.596e-01  4.137e-01  -0.869 0.384775    
#energy                                    -6.717e+00  2.037e+00  -3.297 0.000976 ***
#loudness                                   3.459e-02  3.128e-02   1.106 0.268818    
#duration_ms                               -5.866e-06  4.369e-06  -1.343 0.179372    
#valence                                    4.723e-01  2.366e-01   1.996 0.045900 *  
#avg.age.not.top.ten                       -1.647e-01  1.139e-01  -1.446 0.148214    
#time_signature                             2.675e-01  1.794e-01   1.491 0.136043    
#prior.not.top.ten.hits:loudness            1.085e-02  3.972e-03   2.733 0.006285 ** 
#energy:duration_ms                         1.128e-05  6.057e-06   1.863 0.062472 .  
#danceability:prior.not.top.ten.hits        1.085e-01  5.380e-02   2.017 0.043668 *  
#mode:avg.age.not.top.ten                  -7.033e-02  4.649e-02  -1.513 0.130369    
#energy:avg.age.not.top.ten                 2.379e-01  1.516e-01   1.569 0.116720    
#prior.not.top.ten.hits:mode                2.424e-02  1.613e-02   1.503 0.132814    
#prior.not.top.ten.hits:prior.top.ten.hits -3.136e-03  2.157e-03  -1.454 0.146014    
#mode:acousticness                         -7.215e-01  5.012e-01  -1.440 0.150006   

# model with only influential terms:

# pred.hit.top.ten ~ energy + valence + danceability + prior.top.ten.hits + 
#  danceability:prior.not.top.ten.hits + prior.not.top.ten.hits:loudness

# checking the performance of the new model 
lr_new2 <- glm(pred.hit.top.ten ~ energy + valence + danceability + prior.top.ten.hits + 
                 danceability:prior.not.top.ten.hits + prior.not.top.ten.hits:loudness,
               data=train, family = binomial(link = "logit"))

roc_curve(lr_new2, test, test$pred.hit.top.ten)
cv.glm(train, lr_new2, K=10)$delta
#[1] 0.1239037 0.1238855 -> Performs worse than the both model

# In Summary, since all the analyzed interaction terms and models including interactions perform 
# worse or roughly the same as the both model found at the top, we have decided not to add any 
# interaction term in order to not complicate the interpretability of the model and keep it as 
# simple as possible. 


#######################################################################
#
# Lasso Model - All Variables
# 
#######################################################################

# create lasso model
s.lasso <- glmnet(as.matrix(train[, names(train) != "pred.hit.top.ten"]), as.matrix(train$pred.hit.top.ten), alpha=1, family = "binomial")

# minimum lamdba
lasso.cv <- cv.glmnet(data.matrix(train[, names(train) != "pred.hit.top.ten"]), data.matrix(train$pred.hit.top.ten), alpha=1, family = "binomial")
best.lambda <- lasso.cv$lambda.min
# 0.0005438486

# Testing ###################################################################

best.lambda.pred <- predict(s.lasso, s=best.lambda ,newx = data.matrix(test[, names(test) != "pred.hit.top.ten"]))

# show performance of model
grid <- 10^seq(1,-3,length=100)
final.lasso       <- glmnet(data.matrix(test[, names(test) != "pred.hit.top.ten"]), data.matrix(test$pred.hit.top.ten), alpha=1, lambda=grid, family = "binomial")
final.lasso.coef  <- predict(final.lasso, type="coefficients", s = best.lambda)

# Lasso does not eliminate any variables so we will stop our testing here and instead go with the 
# model that we have created using all the variables (section above)


#######################################################################
#
# Logistic Regression Model - Audio Traits Subset
# 
#######################################################################

audio.train <- train %>% 
  select(-prior.top.ten.hits, -prior.not.top.ten.hits, -prior.weeks.at.top.ten, -avg.age.top.ten, -avg.age.not.top.ten) %>% 
  na.omit()

audio.test <- test %>%
  select(-prior.top.ten.hits, -prior.not.top.ten.hits, -prior.weeks.at.top.ten, -avg.age.top.ten, -avg.age.not.top.ten) %>% 
  na.omit()

# cast response as factors
audio.train$pred.hit.top.ten <- as.factor(audio.train$pred.hit.top.ten)
audio.test$pred.hit.top.ten  <- as.factor(audio.test$pred.hit.top.ten)

# remove NA's 
audio.train <- na.omit(audio.train)
audio.test  <- na.omit(audio.test)

# Model Selection #####################################################

# use hybrid stepwise selection
a.null <- glm(pred.hit.top.ten~1, data=audio.train, family = binomial(link = "logit"))
a.full <- glm(pred.hit.top.ten~., data=audio.train, family = binomial(link = "logit"))

audio <- step(a.null, scope=list(lower=a.null, upper=a.full), direction="both")
summary(audio)
# Coefficients:
#                    Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)    -3.056e+00  8.758e-01  -3.489 0.000484 ***
#   danceability    2.204e+00  3.373e-01   6.533 6.46e-11 ***
#   mode           -3.849e-01  8.459e-02  -4.551 5.35e-06 ***
#   duration_ms     3.060e-06  9.729e-07   3.145 0.001662 ** 
#   acousticness   -9.726e-01  2.671e-01  -3.641 0.000271 ***
#   energy         -1.712e+00  3.952e-01  -4.333 1.47e-05 ***
#   loudness        9.709e-02  2.717e-02   3.573 0.000352 ***
#   valence         4.139e-01  2.319e-01   1.785 0.074315 .  
#   time_signature  3.142e-01  1.832e-01   1.715 0.086312 .  
#   key            -1.665e-02  1.136e-02  -1.466 0.142704    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
#   (Dispersion parameter for binomial family taken to be 1)
# 
#   Null deviance: 4223.9  on 4866  degrees of freedom
#   Residual deviance: 4050.8  on 4857  degrees of freedom
#   AIC: 4070.8
# 
#   Number of Fisher Scoring iterations: 5

# Testing ###################################################################

cv.glm.accuracy(formula(audio), audio.train, N=31)
# 0.8471338 CV Accuracy Score using Training Set

cv.glm(audio.train, audio, K=10)$delta
# 0.1228875 0.1228582 prediction error rate and adjusted prediction error rate

# check for raw prediction accuracy (pick .2 as the threshold for our response)
audio.pred <- ifelse(predict(audio, newdata = audio.test, type="response") >.2,1,0)
sum(audio.test$pred.hit.top.ten == audio.pred) / length(audio.pred)
# 0.7083333 is raw prediction accuracy

roc_curve(audio, audio.test, audio.test$pred.hit.top.ten)


#######################################################################
#
# Logistic Regression Model - Chart Info Subset
# 
#######################################################################

chart.train <- train %>% 
  select(-acousticness, -danceability, -duration_ms, -energy, -instrumentalness, -key, -liveness, -loudness, -mode, -speechiness, -tempo, -time_signature, -valence) %>% 
  na.omit()

chart.test <- test %>%
  select(-acousticness, -danceability, -duration_ms, -energy, -instrumentalness, -key, -liveness, -loudness, -mode, -speechiness, -tempo, -time_signature, -valence) %>% 
  na.omit()

# cast response as factors
chart.train$pred.hit.top.ten <- as.factor(chart.train$pred.hit.top.ten)
chart.test$pred.hit.top.ten  <- as.factor(chart.test$pred.hit.top.ten)

# remove NA's 
chart.train <- na.omit(chart.train)
chart.test  <- na.omit(chart.test)

# Model Selection #####################################################

# use hybrid stepwise selection
c.null <- glm(pred.hit.top.ten~1, data=chart.train, family = binomial(link = "logit"))
c.full <- glm(pred.hit.top.ten~., data=chart.train, family = binomial(link = "logit"))

chart <- step(c.null, scope=list(lower=c.null, upper=c.full), direction="both")
summary(chart)
# Coefficients:
#                       Estimate Std. Error z value Pr(>|z|)    
# (Intercept)            -1.90485    0.06312 -30.178  < 2e-16 ***
# prob.top.ten.hits       1.19882    0.16122   7.436 1.04e-13 ***
# prior.not.top.ten.hits -0.04488    0.00828  -5.421 5.94e-08 ***
# prior.top.ten.hits      0.09750    0.01730   5.637 1.73e-08 ***
# (Dispersion parameter for binomial family taken to be 1)
# Null deviance: 4223.9  on 4866  degrees of freedom
# Residual deviance: 3950.1  on 4863  degrees of freedom
# AIC: 3958.1
# Number of Fisher Scoring iterations: 7

# Testing ###################################################################

cv.glm.accuracy(formula(chart), chart.train, N=31)
# 0.866242 CV Accuracy Score using Training Set

cv.glm(chart.train, chart, K=10)$delta
#  0.1247449 0.1247305 prediction error rate and adjusted prediction error rate

# check for raw prediction accuracy (pick .2 as the threshold for our response)
chart.pred <- ifelse(predict(chart, newdata = chart.test, type="response") >.2,1,0)
sum(chart.test$pred.hit.top.ten == chart.pred) / length(chart.pred)
# 0.75 is raw prediction accuracy

roc_curve(chart, chart.test, chart.test$pred.hit.top.ten)


##############################################################################
#
# LOGISTIC REGRESSION ROC CURVES
# 
##############################################################################

# auc values
both.auc  <- auc_values(both, test, test$pred.hit.top.ten)
audio.auc <- auc_values(audio, audio.test, audio.test$pred.hit.top.ten)
chart.auc <- auc_values(chart, chart.test, chart.test$pred.hit.top.ten)

# roc values
both.roc  <- roc_values(both, test, test$pred.hit.top.ten)
audio.roc <- roc_values(audio, audio.test, audio.test$pred.hit.top.ten)
chart.roc <- roc_values(chart, chart.test, chart.test$pred.hit.top.ten)

all.roc <- rbind(        
  cbind(ord = 1, mod = 'logi', type = paste("Full Model AUC :: ",   both.auc),  both.roc), 
  cbind(ord = 2, mod = 'logi', type = paste("Audio Traits AUC :: ", audio.auc), audio.roc), 
  cbind(ord = 3, mod = 'logi', type = paste("Chart Info AUC :: ",   chart.auc), chart.roc)) %>% 
  arrange(ord)

logistic <- ggplot(all.roc %>% filter(mod=="logi"), aes(x=fp, y=tp, group=as.factor(ord), colour=type)) + 
  labs(x="False Positive Rate", y="True Positive Rate") +
  scale_x_continuous(labels = percent, limits = c(0,1)) +
  scale_y_continuous(labels = percent, limits = c(0,1)) + 
  geom_abline(aes(intercept=0, slope=1)) +
  ggtitle(bquote(atop(.("ROC Curve - Logistic")))) + 
  coord_equal() + geom_line(size=1.2) + theme(legend.position="bottom") +
  theme_fivethirtyeight()

logistic

##############################################################################
#
# RANDOM FOREST
# 
##############################################################################
# function for rand forest
rand_forest <- function(train, test) {
  rf.fit <- randomForest(pred.hit.top.ten ~., data = train)
  print(rf.fit)
  
  #prepare model for ROC Curve https://www.r-bloggers.com/part-3-random-forests-and-model-selection-considerations/
  test.forest = predict(rf.fit, type = "prob", newdata = test)
  forestpred = prediction(test.forest[,2], test$pred.hit.top.ten)
  forestperf = performance(forestpred, "tpr", "fpr")
  auc.perf = performance(forestpred, measure="auc")
  
  roc.vals = data.frame(cbind(forestperf@x.values[[1]], forestperf@y.values[[1]]))
  colnames(roc.vals) <- c("fp", "tp")
  
  ggplot(roc.vals, aes(x=fp, y=tp, colour="#2980b9")) + 
    labs(x=forestperf@x.name, y=forestperf@y.name) +
    scale_x_continuous(labels = percent, limits = c(0,1)) + scale_y_continuous(labels = percent, limits = c(0,1)) + 
    geom_abline(aes(intercept=0, slope=1, colour="#34495e")) + guides(colour = FALSE) + coord_equal() +
    ggtitle(bquote(atop(.("ROC Curve"), atop(italic(.(paste("AUC: ", round(auc.perf@y.values[[1]],digits=2)))), "")))) + geom_line(size=1.2) +
    theme_fivethirtyeight()
}  

# run forest
rand_forest(train, test)

rf.fit <- randomForest(pred.hit.top.ten ~., data = train)
rf.imp <- data.frame(importance(rf.fit))
rf.imp$variable <- rownames(rf.imp)
rf.imp %>% arrange(desc(MeanDecreaseGini))

##############################################################################
#
# MAKE AN ED SHEERHAN SONG FAMOUS
# 
##############################################################################

ed <- raw %>%
  filter(year(birthday)>=2014 & 
           total.weeks > 12   &
           pred.hit.top.ten == 0) %>% 
  filter(song_id == 5737)
ed %>% glimpse
knitr::kable(ed)

predict(both, newdata = ed, type="response")
# 0.2600005

# drop the acousticness
ed$acousticness = 0
predict(both, newdata = ed, type="response")
# 0.3088853

# boost the danciness
ed$danceability = 1
predict(both, newdata = ed, type="response")
# 0.3868141

# boost the energy --> measure of drive
ed$energy = .3
predict(both, newdata = ed, type="response")
# 0.534926

# loudness wars
ed$loudness = -3.451
predict(both, newdata = ed, type="response")
# 0.5535656