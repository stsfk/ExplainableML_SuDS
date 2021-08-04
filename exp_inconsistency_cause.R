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
  xgboost,
  ParBayesianOptimization
)

# data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")

# CONSTANT
tree_method = "gpu_hist"
#tree_method = "hist"
SEED = 439759

# initial split
initial_fold <- initial_split(trainable_df, prop = 60/100, strata = c("peak_flow"))

trainable_df_splitted <- analysis(initial_fold)
test_df_splitted <- testing(initial_fold)

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

divide_train_test_data <- function(df, prop, strata){
  # random select prop events as training and validation sets
  
  first_split <- initial_split(df, prop = prop, strata = strata)
  
  train_df <- analysis(first_split)
  
  if (prop/(1-prop) == 1) {
    val_df <- assessment(first_split)
  } else {
    val_df <- initial_split(assessment(first_split), prop = prop/(1-prop),  strata = strata) %>%
      analysis()
  }
  
  list(train_df = train_df, val_df = val_df)
}

extract_DMatrix <- function(row_index, data_feature){
  # This function create xgb.DMatrix for row_index subset of data_process
  
  out <- data_feature[row_index,]
  out <- xgb.DMatrix(data = data.matrix(out %>% dplyr::select(-Y)),
                     label = out$Y)
  
  out
}

scoringFunction <- function(eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma,
                            m, l, n,
                            save_model = T, 
                            option = 1) {
  
  #  "train_df", "val_df", "test_df", and "temp_path" is from the environment

  data_feature <- gen_feature(m, l, n, account_cum_rain, account_season, 
                              data_process, option)
  
  c(train_index, val_index, test_index) %<-% {
    list(train_df, val_df, test_df) %>%
      lapply(function(x)
        unlist(x$record_id))
  }
  
  c(dtrain, dval, dtest, dtrain_val) %<-% {
    list(train_index,
         val_index,
         test_index,
         c(train_index, val_index)) %>%
      lapply(extract_DMatrix, data_feature = data_feature)
  }
  
  ## experiment
  
  n_X_vars <- names(data_feature) %>% str_detect("^X") %>% sum() # for setting monotone_constraints
  preprocess_cols <- c(rep(1, n_X_vars), rep(0, ncol(dtrain) - n_X_vars - 1))
  
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
  
  # identify best_iteration
  watchlist <- list(validation=dval)
  xgbFit <- xgb.train(
    data = dtrain,
    objective = "reg:squarederror",
    tree_method = tree_method,
    max_bin = 256,
    verbose = 0,
    params = Pars,
    watchlist = watchlist,
    nrounds = 5000,
    early_stopping_rounds = 20,
    monotone_constraints = preprocess_cols
  )
  
  best_iteration <- xgbFit$best_iteration
  
  score <- hydroGOF::gof(
    sim = predict(xgbFit, dval),
    obs = getinfo(dval, "label"),
    digits = 8
  )[4]
  
  # train model on dtrain_val
  pred_df <- list()
  
  watchlist <- NULL
  xgbFit <- xgb.train(
    data = dtrain_val,
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
    datetime = data_process$datetime[c(train_index, val_index)],
    ob = getinfo(dtrain_val, "label"),
    pred = predict(xgbFit, dtrain_val)) %>%
    arrange(datetime)
  
  test_pred <- tibble(
    datetime = data_process$datetime[test_index],
    ob = getinfo(dtest, "label"),
    pred = predict(xgbFit, dtest)) %>%
    arrange(datetime)
  
  # save model
  if (save_model){
    
    model_id <- (list.files(temp_path) %>% length()) + 1
    file_path <- paste0(temp_path, "/model_",model_id,".model")
    
    xgb.save(model = xgbFit, fname = file_path)
  }
  
  # output
  out <- list(
    Score = -score,
    best_iteration = best_iteration,
    train_pred = list(train_pred),
    test_pred = list(test_pred)
  )
  
  gc()
  
  return(out)
}

# Experiment --------------------------------------------------------------

optimization_wrapper <- function(final_model_path){
  # train_df, val_df, test_df are from the environment
  
  # optimization hyperparas
  initPoints <- 12 # 12 experiment
  max_iter <- 13 # 20 experiment
  patience <- 10
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
  
  for (iter in (initPoints + 1):max_iter) {
    optObj <- updateGP(optObj)
    optObj <- addIterations(optObj, iters.n = 1, iters.k = 1)
    if ((iter - which.max(optObj$scoreSummary$Score)) > patience &&
        iter > (patience + initPoints)) {
      # early stop if no improvement > patience, and iter > patience + initPoints
      break
    }
  }
  
  # post-process opt results
  best_model_id <- optObj$scoreSummary$Score %>% which.max()
  
  file.copy(from = paste0(temp_path, "/model_", best_model_id, ".model"),
            to = final_model_path)
  
  optObj
}

temp_path <- "./data/WS/inconsist_exp/temp"
for (prop in c(5, 10)){
  for (split in c(1:2)){
    c(train_df, val_df) %<-% divide_train_test_data(trainable_df_splitted, prop = prop/60, strata = "peak_flow")
    
    for (repeat_id in c(1:2)){
      
      # clean temp path of Bayesian opt results
      unlink(paste0(temp_path, "/*"))
      
      final_model_path <- paste0("./data/WS/inconsist_exp/",
                         "prop=",prop,
                         "split=",split,
                         "repeat=",repeat_id,
                         ".model")
      optObj <- optimization_wrapper(final_model_path)
      
      
      optObj_path <- paste0("./data/WS/inconsist_exp/",
                         "prop=",prop,
                         "split=",split,
                         "repeat=",repeat_id,
                         ".Rda")
      save(optObj, file = optObj_path)
      
    }
  }
}





# Optimization ------------------------------------------------------------


if (!dir.exists(dir_path)){
  dir.create(dir_path)
}

dir_path <- paste0("./data/WS/inconsist_exp/",
                   "prop=",prop,
                   "split=",split,
                   "repeat=",repeat_id,
                   ".model")


eval_grid <- expand.grid(
  prop = c(5, 10, 20, 30),
  split = c(1:10),
  repeat_id = c(1:10)
)

df <- trainable_df_splitted
prop <- 5/60
strata <- "peak_flow"

c(train_df, val_df)  %<-% divide_train_test_data(df, prop, strata)
test_df <- test_df_splitted










scoringFunction(eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma,
                m, l, n, dir_path)








# recycle -----------------------------------------------------------------


dir_path <- paste0("./data/WS/model_fits")
dir_path <- paste0("./data/WS/model_fits/xgb_opt_")

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



optimization_wrapper <- function(option, data_process){
  
  # build (if not exist) and clean the folder containing the simulation result
  
  dir_path <- paste0("./data/SHC/model_fits/xgb_opt_",option)
  
  if (!dir.exists(dir_path)){
    dir.create(dir_path)
  }
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



