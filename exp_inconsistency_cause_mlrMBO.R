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
  mlrMBO
)

# data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")

# CONSTANT
tree_method = "gpu_hist"
#tree_method = "hist"
SEED = 439759

# remove zero runoff event
trainable_df <- trainable_df %>%
  filter(peak_flow != 0)

# initial split, 40% data -> test; 60% data training
# "prop: The proportion of data to be retained for modeling/analysis."
initial_fold <- initial_split(trainable_df, prop = 60/100, strata = c("peak_flow"))

trainable_df_splitted <- analysis(initial_fold)
test_df <- testing(initial_fold)

save(trainable_df_splitted, test_df, file = "./data/WS/inconsist_exp/init_split.Rda")

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
  # for the particular study, the data is chosen from the remaining 60% of samples
  # "prop: The proportion of data to be retained for modeling/analysis."
  
  # create smaller training dataset
  first_split <- initial_split(df, prop = prop, strata = strata)
  train_df <- analysis(first_split)
  
  # the assessment dataset is (1 - prop) of the original dataset in term of size
  # from the assessment dataset select prop/(1 - prop) as validation dataset
  if (prop/(1-prop) == 1) {
    val_df <- assessment(first_split)
  } else {
    val_df <- initial_split(assessment(first_split), prop = prop/(1-prop),  strata = strata) %>%
      analysis()
  }
  
  list(train_df = train_df, val_df = val_df)
}

extract_DMatrix <- function(row_index, data_feature){
  # This function create xgb.DMatrix for row_index subset of data_feature
  
  out <- data_feature[row_index,]
  out <- xgb.DMatrix(data = data.matrix(out %>% dplyr::select(-Y)),
                     label = out$Y)
  
  out
}

scoringFunction <- function(x, save_model = T) {
  
  # "train_df", "val_df", "test_df", and "temp_path" is from the environment
  # "option" is from global
  
  eta <- x["eta"] %>% unlist()
  max_depth <- x["max_depth"] %>% unlist()
  min_child_weight <- x["min_child_weight"] %>% unlist()
  subsample <- x["subsample"] %>% unlist()
  colsample_bytree <- x["colsample_bytree"] %>% unlist()
  gamma <- x["gamma"] %>% unlist()
  
  m <- x["m"] %>% unlist()
  l <- x["l"] %>% unlist()
  n <- x["n"] %>% unlist()
  account_cum_rain <- x["account_cum_rain"] %>% unlist()
  account_season <- x["account_season"] %>% unlist()
  
  # create data_feature
  data_feature <- gen_feature(m, l, n, account_cum_rain, account_season, 
                              data_process, option)
  
  # get the record id of the samples in training, validation, and test folds
  c(train_index, val_index, test_index) %<-% {
    list(train_df, val_df, test_df) %>%
      lapply(function(x)
        unlist(x$record_id))
  }
  
  # create dtrain, dval, dtest, and dtrain_val DMatrix for training
  c(dtrain, dval, dtest, dtrain_val) %<-% {
    list(train_index,
         val_index,
         test_index,
         c(train_index, val_index)) %>%
      lapply(extract_DMatrix, data_feature = data_feature)
  }
  
  ## experiment
  
  # create input for monotone_constraints
  n_X_vars <- names(data_feature) %>% str_detect("^X") %>% sum() # for setting monotone_constraints
  preprocess_cols <- c(rep(1, n_X_vars), rep(0, ncol(data_feature) - n_X_vars - 1))
  
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
  
  # compute the prediction loss on dval, which is the score to be optimized
  score <- hydroGOF::gof(
    sim = predict(xgbFit, dval),
    obs = getinfo(dval, "label"),
    digits = 8
  )[4]
  
  # train model on dtrain_val
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
  
  # predictions for dtrain_val and dtest
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
  
  # summary
  out <- list(
    Score = score,
    best_iteration = best_iteration,
    train_pred = list(train_pred),
    test_pred = list(test_pred)
  )
  
  # save model
  if (save_model){
    # save model
    model_id <- (list.files(temp_path, pattern = "\\.model$") %>% length()) + 1
    file_path <- paste0(temp_path, "/model_",model_id,".model")
    xgb.save(model = xgbFit, fname = file_path)
    
    # save predictions and scores
    file_path <- paste0(temp_path, "/gof_",model_id,".Rda")
    save(out, file = file_path)
  }
  
  # output
  gc()
  
  return(score)
}

save_dMatrix <- function(x,
                         final_data_path,
                         train_df,
                         val_df,
                         test_df){
  # save dataset for the optimal model
  
  m <- x["m"] %>% unlist()
  l <- x["l"] %>% unlist()
  n <- x["n"] %>% unlist()
  account_cum_rain <- x["account_cum_rain"] %>% unlist()
  account_season <- x["account_season"] %>% unlist()
  
  # create data_feature
  data_feature <- gen_feature(m, l, n, account_cum_rain, account_season, 
                              data_process, option)
  
  # get the record id of the samples in training, validation, and test folds
  c(train_index, val_index, test_index) %<-% {
    list(train_df, val_df, test_df) %>%
      lapply(function(x)
        unlist(x$record_id))
  }
  
  # create dtrain, dval, dtest, and dtrain_val DMatrix for training
  c(dtrain, dval, dtest, dtrain_val) %<-% {
    list(train_index,
         val_index,
         test_index,
         c(train_index, val_index)) %>%
      lapply(extract_DMatrix, data_feature = data_feature)
  }
  
  # save data
  xgboost::xgb.DMatrix.save(dtrain, fname = paste0(final_data_path, "dtrain.data"))
  xgboost::xgb.DMatrix.save(dval, fname = paste0(final_data_path, "dval.data"))
  xgboost::xgb.DMatrix.save(dtest, fname = paste0(final_data_path, "dtest.data"))
  xgboost::xgb.DMatrix.save(dtrain_val, fname = paste0(final_data_path, "dtrain_val.data"))
  
  # save feature names for distributing SHAP
  write.table(
    names(data_feature[-1]),
    file = paste0(final_data_path, "feature_name.csv"),
    quote = F,
    row.names = F,
    col.names = F
  )
}

obj_fun <- makeSingleObjectiveFunction(
  fn = scoringFunction,
  par.set = makeParamSet(
    makeNumericParam("eta",                    lower = 0.005, upper = 0.1),
    makeIntegerParam("max_depth",              lower= 2,      upper = 10),
    makeIntegerParam("min_child_weight",       lower= 1,    upper = 10),
    makeNumericParam("subsample",              lower = 0.20,  upper = 1),
    makeNumericParam("colsample_bytree",       lower = 0.20,  upper = 1),
    makeNumericParam("gamma",                  lower = 0,     upper = 10),
    makeIntegerParam("m",                      lower= 144,  upper = 1440),
    makeIntegerParam("l",                      lower= 1,    upper = 36),
    makeIntegerParam("n",                      lower= 2,    upper = 36),
    makeIntegerParam("account_cum_rain",       lower= 0,    upper = 1),
    makeIntegerParam("account_season",         lower= 0,    upper = 1)
  ),
  has.simple.signature = FALSE,
  minimize = TRUE
)


opt_wrapper <- function(save_results = T,
                        final_model_path = NULL,
                        final_gof_path = NULL,
                        run_path = NULL,
                        n_iter = 100) {
  
  # design matrix
  des = generateDesign(
    n = 4 * getNumberOfParameters(obj_fun),
    par.set = getParamSet(obj_fun),
    fun = lhs::randomLHS
  )
  
  des$y = apply(des, 1, obj_fun)
  
  # optimization control
  control <- makeMBOControl() %>%
    setMBOControlTermination(., iters = n_iter - 4 * getNumberOfParameters(obj_fun))
  
  # run optimization
  run <- mbo(
    fun = obj_fun,
    design = des,
    control = control,
    show.info = TRUE
  )
  
  # save results
  if (save_results) {
    file.copy(from = paste0(temp_path, "/model_", run$best.ind, ".model"),
              to = final_model_path, overwrite = T)
    
    file.copy(from = paste0(temp_path, "/gof_", run$best.ind, ".Rda"),
              to = final_gof_path, overwrite = T)
    
    save(run, file = run_path)
  }
  
  run
}



# Experiment --------------------------------------------------------------

option = 1
temp_path <- "./data/WS/inconsist_exp/temp"

for (prop in c(5, 10, 20, 30)) {
  for (split in c(1:10)) {
    
    # split
    c(train_df, val_df) %<-% divide_train_test_data(trainable_df_splitted,
                                                    prop = prop / 60,
                                                    strata = "peak_flow")

    save(
      train_df,
      val_df,
      test_df,
      file = paste0(
        "./data/WS/inconsist_exp/split_",
        "prop=",
        prop,
        "split=",
        split,
        ".Rda"
      )
    )
    
    for (repeat_id in c(1:10)) {
      
      # prepare temporal path for experiments
      unlink(paste0(temp_path, "/*"))
      
      # prepare run
      final_model_path <- paste0(
        "./data/WS/inconsist_exp/",
        "prop=",
        prop,
        "split=",
        split,
        "repeat=",
        repeat_id,
        ".model"
      )
      
      final_gof_path <- paste0(
        "./data/WS/inconsist_exp/gof_",
        "prop=",
        prop,
        "split=",
        split,
        "repeat=",
        repeat_id,
        ".Rda"
      )
      
      final_data_path <- paste0(
        "./data/WS/inconsist_exp/data_",
        "prop=",
        prop,
        "split=",
        split,
        "repeat=",
        repeat_id
      )
      
      run_path <- paste0(
        "./data/WS/inconsist_exp/run_",
        "prop=",
        prop,
        "split=",
        split,
        "repeat=",
        repeat_id,
        ".Rda"
      )
      
      # run optimization
      run <- opt_wrapper(
        save_results = T,
        final_model_path,
        final_gof_path,
        run_path,
        n_iter = 100
      )
      
      # save datasets used in training and assessment
      save_dMatrix(run$x,
                   final_data_path,
                   train_df,
                   val_df,
                   test_df)
    }
  }
}
