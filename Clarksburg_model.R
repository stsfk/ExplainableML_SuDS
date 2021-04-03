if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,
  tidymodels,
  lubridate,
  RcppRoll,
  zeallot,
  xgboost,
  ParBayesianOptimization,
  hydroGOF
)

# Prepare -----------------------------------------------------------------

SEED <- 88192
set.seed(SEED)
start_time <- Sys.time()

ntread <- parallel::detectCores() - 2

tree_method = "gpu_hist"

# Data --------------------------------------------------------------------

load("./data/Clarksburg/rainfall_runoff.Rda")
load("./data/Clarksburg/trainable_index.Rda")

data_process <- data_process %>%
  select(datetime, X=rain, Y=flow)

# create CV folds
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
  
  # Detailed rain ts to be included, s_e_l, more recent rainfalls 
  s_e_l <- tibble(s = c(0:l),
                  e = c(0:l))
  
  # Aggregated rain TS
  d <- (m - l - n * a1) * 2 / n / (n - 1) # compute the common difference
  
  e <- (a1 + 0:(n - 1) * d) %>%
    cumsum() %>%
    round()
  
  # sort to avoid later interval < the previous interval
  e <- (c(a1, diff(e)) %>% sort() %>% cumsum()) + l
  
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


gen_feature <- function(m, l, n, account_cum_rain, option = 1) {
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
  
  out
}


prepare_train_test_data <- function(m, l, n,                                     
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
  
  # get relative location of validation in training_outer data sets
  val <-
    lapply(cv_folds$inner_resamples[[outer_i]]$splits, assessment)
  val_record_id <- val %>%
    lapply(function(x)
      x %>% pull(record_id) %>% unlist())
  
  # get training_inner
  val_record_id <-
    lapply(val_record_id, function(x)
      which(training_outer_record_id %in% x)) # get relative location
  
  # derive data_feature
  data_feature <- gen_feature(m, l, n, data_process, option)
  
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


# outer_i, cv_folds, data_process, option, and tree_method are defined outside
# the scoringFunction, so bayesOpt do not need other parameters
scoringFunction <- function(eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma,
                            m, l, n,
                            save_model = T,
                            best_iteration_correction = F) {
  # This function calls 'prepare_train_test_data', which calls 'gen_feature',
  # which calls 'feature_vector' and 'gen_s_e'.
  # inner_v is from global
  
  dir_path <- paste0("./data/Clarksburg/model_fits/xgb_opt_",option,"_iter_",outer_i)
  
  c(dtrain, dtest, val, train_index, test_index) %<-%
    prepare_train_test_data(m, l, n,
                            outer_i, data_process, cv_folds, option)
  
  n_X_vars <- names(dtrain) %>% str_detect("^X") %>% sum() # for setting monotone_constraints
  preprocess_cols <- c(rep(1, n_X_vars), rep(0, ncol(dtrain) - n_X_vars - 1))
  
  dtrain <- xgb.DMatrix(data = data.matrix(dtrain %>% dplyr::select(-Y)),
                        label = dtrain$Y)
  
  dtest <- xgb.DMatrix(data = data.matrix(dtest %>% dplyr::select(-Y)),
                       label = dtest$Y)
  
  # xgboost hyperparameters and CV training
  Pars <- list(
    booster = "gbtree",
    eta = eta,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    colsample_bytree = colsample_bytree,
    subsample = subsample,
    gamma = gamma
  )
  
  xgbcv <- xgb.cv(
    objective = "reg:squarederror",
    data = dtrain,
    folds = val, # experiment nfold = 5
    tree_method = tree_method,
    max_bin = 256,
    nround = 5000,
    early_stopping_rounds = 20,
    verbose = 0,
    params = Pars,
    monotone_constraints = preprocess_cols
  )
  
  # train model on all training_outer fold
  pred_df <- list()
  
  if (best_iteration_correction){
    best_iteration <- round(xgbcv$best_iteration/(1-1/inner_v))
  } else {
    best_iteration <- xgbcv$best_iteration
  }
  
  watchlist <- NULL
  xgbFit <- xgb.train(
    data = dtrain,
    objective = "reg:squarederror",
    tree_method = tree_method,
    max_bin = 256,
    nround = best_iteration,
    verbose = 0,
    params = Pars,
    monotone_constraints = preprocess_cols
  )
  
  # predictions
  train_pred <- tibble(
    datetime = data_process$datetime[train_index],
    ob = getinfo(dtrain, "label"),
    pred = predict(xgbFit, dtrain))
  
  test_pred <- tibble(
    datetime = data_process$datetime[test_index],
    ob = getinfo(dtest, "label"),
    pred = predict(xgbFit, dtest))

  # save model
  if (save_model){
    
    model_id <- (list.files(dir_path) %>% length()) + 1
    file_path <- paste0(dir_path, "/model_",model_id,".model")
    
    xgb.save(model = xgbFit, fname = file_path)
  }
  
  # output
  out <- list(
    Score = -min(xgbcv$evaluation_log$test_rmse_mean),
    nrounds = xgbcv$best_iteration,
    train_pred = list(train_pred),
    test_pred = list(test_pred)
  )
  
  gc()
  
  return(out)
}



optimization_wrapper <- function(option, outer_i, cv_folds, data_process){
  
  # build (if not exist) and clean the folder containing the simulation result
  
  dir_path_main <- paste0("./data/Clarksburg/model_fits/")
  ifelse(!dir.exists(file.path(dir_path_main)), dir.create(file.path(dir_path_main)), FALSE)
  
  dir_path <- paste0("./data/Clarksburg/model_fits/xgb_opt_",option,"_iter_",outer_i)
  ifelse(!dir.exists(file.path(dir_path)), dir.create(file.path(dir_path)), FALSE)
  
  unlink(paste0(dir_path, "/*"))
  
  # optimization hyperparas
  initPoints <- 50 # 12 experiment
  max_iter <- 100 # 100 experiment
  patience <- 50
  plotProgress <- T
  
  bounds <- list(
    eta = c(0.005, 0.1),
    max_depth = c(2L, 10L),
    min_child_weight = c(1L, 10L),
    subsample = c(0.25, 1),
    colsample_bytree = c(0.25, 1),
    gamma = c(0, 10),
    m = c(144L, 1440L),
    l = c(1L, 36L),
    n = c(2L, 36L)
  )
  
  gsPoints = 200
  
  # optimization
  optObj <- bayesOpt(
    FUN = scoringFunction, 
    bounds = bounds, 
    initPoints = initPoints,
    iters.n = 1, 
    iters.k = 1,
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

eval_grid <- expand.grid(option = 1:1,
                         outer_i = c(1:5)) %>%
  as_tibble()

optObjs <- vector("list", nrow(eval_grid))

for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]
  outer_i <- eval_grid$outer_i[i]
  
  optObj <- optimization_wrapper(option, outer_i, cv_folds, data_process)
  
  fname <- paste0("./data/Clarksburg/model_fits/xgb_opt_", option, "_iter_", outer_i, ".Rda")
  save(optObj, file = fname)
  
  optObjs[[i]] <- optObj
}

save(optObjs, file = "./data/Clarksburg/model_fits/optObjs.Rda")


