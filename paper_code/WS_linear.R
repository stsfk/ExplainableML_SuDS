if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  lubridate,
  zeallot,
  RcppRoll,
  caret,
  rsample,
  recipes,
  ParBayesianOptimization
)

# data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")

SEED = 439759

outer_repeats = 1
inner_repeats = 1
outer_v = 5
inner_v = 5

# cv_folds
set.seed(SEED)
cv_folds <- nested_cv(
  trainable_df,
  outside = vfold_cv(
    v = outer_v,
    repeats = outer_repeats,
    strata = c("peak_flow")
  ),
  inside = vfold_cv(
    v = inner_v,
    repeats = inner_repeats,
    strata = c("peak_flow")
  )
)

# functions ---------------------------------------------------------------

gen_s_e <- function(m = 1440, l = 6, n = 6,
                    option = 1, a1 = 2) {
  # Function to gen s_e tibble
  
  # a1 is the first term
  
  # Detailed rain ts to be included
  s_e_l <- tibble(s = c(0:l),
                  e = c(0:l))
  
  # Aggregated rain TS
  d <- (m - l - n * a1) * 2 / n / (n - 1)
  #d <- round(d)
  
  e <- (a1 + 0:(n - 1) * d) %>%
    cumsum() %>%
    round()
  
  # avoid later interval < current interval
  e <- (c(2, diff(e)) %>% sort() %>% cumsum()) + l
  
  # identify s
  s <- c(l + 1, e[-length(e)] + 1)
  
  s_e <- tibble(s = s,
                e = e)
  
  if (option == 1) {
    s_e <- rbind(s_e_l, s_e)
    return(s_e)
  }
  
  if (option == 2) {
    s_e <- rbind(s_e_l, s_e)
    s_e[, 1] <- 0 # key line
    return(s_e)
  }
  
  if (option == 3) {
    s_e[, 1] <- 0 # key line
    
    s_e <- rbind(s_e_l, s_e)
    return(s_e)
  }
  
  if (option == 4) {
    s_e[, 1] <- s_e[[1, 1]] # key line
    
    s_e <- rbind(s_e_l, s_e)
    return(s_e)
  }
}


feature_vector <- function(x, s, e) {
  # This is a function to return vector of input feature,
  # s is the start location
  # e is the end location
  
  e <- e + 1
  
  if (s == 0) {
    out <- roll_sum(x, e, align = "right", fill = NA)
  } else {
    out <-
      roll_sum(x, e, align = "right", fill = NA) - roll_sum(x, s, align = "right", fill = NA)
  }
  
  out
}


gen_feature <- function(m, l, n, account_cum_rain, account_season, 
                        data_process, option = 1) {
  # This is a function for creating features
  #
  #   y: the output variable
  #   x: the original predictor
  #   s_e: the index of new predictors
  
  s_e <- gen_s_e(m, l, n, option)
  
  # create a vector of the original predictor
  org_pred <- data_process$X
  
  # create out tibble
  out <- data_process %>%
    select(Y)
  
  # create features and name the columns based on the "s" and "e" index
  for (i in 1:nrow(s_e)) {
    c(s, e) %<-% s_e[i,]
    
    var_name <- paste0("X", s, "_", e)
    out[var_name] <- feature_vector(org_pred, s, e)
  }
  
  # add cum rainfall var
  if (account_cum_rain) {
    out["cum_rain"] <- data_process$cum_rain
  }
  
  # add season vars
  if (account_season) {
    # the month10 is linear combination of month4-9, thus excluded
    out <- out %>%
      bind_cols(data_process[str_detect(names(data_process), "month4|5|6|7|8|9")])
  }
  
  out
}


prepare_train_test_data <- function(m, l, n, account_cum_rain, account_season,
                                    outer_i, data_process, cv_folds, option) {
  # extract folds from data_feature for training, validation, and testing
  training_outer <- analysis(cv_folds$splits[[outer_i]])
  test <- assessment(cv_folds$splits[[outer_i]])
  
  training_outer_record_id <- training_outer %>%
    pull(record_id) %>%
    unlist()
  
  test_record_id <- test %>%
    pull(record_id) %>%
    unlist()
  
  # get relative location of validation in training_outer data sets; analysis is used here for caret
  val <-
    lapply(cv_folds$inner_resamples[[outer_i]]$splits, analysis)
  val_record_id <- val %>%
    lapply(function(x)
      x %>% pull(record_id) %>% unlist())
  
  # get training_inner
  val_record_id <-
    lapply(val_record_id, function(x)
      which(training_outer_record_id %in% x)) # get relative location
  
  # derive data_feature
  data_feature <- gen_feature(m, l, n, account_cum_rain, account_season, data_process, option)
  
  # subset
  dtrain <- data_feature[training_outer_record_id,]
  dtest <- data_feature[test_record_id,]
  
  # return
  list(
    dtrain = dtrain,
    dtest = dtest,
    val = val_record_id,
    train_index = training_outer_record_id,
    test_index = test_record_id
  )
}


# outer_i, cv_folds, data_process, and option are defined outside
# the scoringFunction, so bayesOpt do not need other parameters
scoringFunction <- function(m, l, n, account_cum_rain, account_season) {
  # This function calls 'prepare_train_test_data', which calls 'gen_feature',
  # which calls 'feature_vector' and 'gen_s_e'.
  
  # prepare model training
  c(dtrain, dtest, val, train_index, test_index) %<-%
    prepare_train_test_data(m, l, n, account_cum_rain, account_season,
                            outer_i, data_process, cv_folds, option)
  
  recipe <- recipe(Y ~ ., data = dtrain) 
  
  ctrl <- trainControl(
    method = "cv",
    number = length(val),
    index = val,
    savePredictions = "none",
    returnData = FALSE
  )
  
  # training
  ModelFit <- caret::train(
    recipe,
    dtrain,
    method = "lm",
    trControl = ctrl
  )
  
  # prepare output
  coefficients <- ModelFit$finalModel$coefficients
  
  val_rmse <- ModelFit$resample$RMSE %>% mean()
  test_pred <- tibble(
    datetime = data_process$datetime[test_index],
    ob = dtest$Y,
    pred = predict(ModelFit, dtest))
  
  
  test_rmse <- hydroGOF::gof(sim = test_pred$pred, obs = test_pred$ob, digits = 10)[4]

  # output
  out <- list(
    Score = -val_rmse,
    test_rmse = test_rmse,
    coefficients = list(coefficients),
    test_pred = list(test_pred)
  )
  
  gc()
  
  return(out)
}


optimization_wrapper <- function(option, outer_i, cv_folds, data_process){

  # optimization hyperparas
  initPoints <- 50 # experiment
  max_iter <- 100 # experiment
  patience <- 50 # experiment
  plotProgress <- T
  
  bounds <- list(
    m = c(144L, 1440L),
    l = c(1L, 36L),
    n = c(2L, 36L),
    account_cum_rain = c(0L, 1L),
    account_season = c(0L, 1L)
  )
  
  gsPoints = 100
  
  # optimization
  optObj <- bayesOpt(
    FUN = scoringFunction, 
    bounds = bounds, 
    initPoints = initPoints,
    iters.n = 1, 
    iters.k = 1,
    convThresh = 1e7,
    gsPoints = gsPoints,
    plotProgress = plotProgress)
  
  for (iter in (initPoints + 2):max_iter){
    optObj <- updateGP(optObj)
    optObj <- addIterations(optObj, iters.n = 1, iters.k = 1)
    if ((iter - which.max(optObj$scoreSummary$Score))>patience && iter > (patience + initPoints)){
      # early stop if no improvement > patience, and iter > patience + initPoints
      break
    }
  }
  
  optObj
}


# Optimization ------------------------------------------------------------


eval_grid <- expand.grid(option = 1:4,
                         outer_i = c(1:5)) %>%
  as_tibble()

optObjs <- vector("list", nrow(eval_grid))

for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]
  outer_i <- eval_grid$outer_i[i]
  
  optObj <- optimization_wrapper(option, outer_i, cv_folds, data_process)
  
  fname <- paste0("./data/WS/lm_fits/lm_opt_", option, "_iter_", outer_i, ".Rda")
  save(optObj, file = fname)
  
  optObjs[[i]] <- optObj
}

save(optObjs, file = "./data/WS/lm_fits/lm_optObjs.Rda")
