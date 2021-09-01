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
  mlrMBO,
  recipes
)

# data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")

tree_method = "gpu_hist"
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


# outer_i, cv_folds, data_process, option, and tree_method are defined outside
# the scoringFunction, so bayesOpt do not need other parameters
scoringFunction <- function(x,
                            save_model = T) {
  # This function calls 'prepare_train_test_data', which calls 'gen_feature',
  # which calls 'feature_vector' and 'gen_s_e'.
  # inner_v is from global
  
  m <- x["m"] %>% unlist()
  l <- x["l"] %>% unlist()
  n <- x["n"] %>% unlist()
  account_cum_rain <- x["account_cum_rain"] %>% unlist()
  account_season <- x["account_season"] %>% unlist()
  
  # get dataset for training
  c(dtrain, dtest, val, train_index, test_index) %<-%
    prepare_train_test_data(m, l, n, account_cum_rain, account_season,
                            outer_i, data_process, cv_folds, option)
  
  
  # train linear regression model
  recipe <- recipe(Y ~ ., data = dtrain)
  
  # convert index of val samples to index of train samples
  val <- lapply(val, function(x) setdiff(1:nrow(dtrain), x))
  
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
  
  test_pred <- tibble(
    datetime = data_process$datetime[test_index],
    ob = dtest$Y,
    pred = predict(ModelFit, dtest))
  
  train_pred <- tibble(
    datetime = data_process$datetime[train_index],
    ob = dtrain$Y,
    pred = predict(ModelFit, dtrain))
  
  val_rmse <- ModelFit$resample$RMSE %>% mean()
  test_rmse <- hydroGOF::gof(sim = test_pred$pred, obs = test_pred$ob, digits = 10)[4]
  
  # summary
  out <- list(
    Score = val_rmse,
    train_pred = list(train_pred),
    test_pred = list(test_pred),
    coefficients = list(coefficients)
  )
  
  # save model
  if (save_model){
    dir_path <- paste0("./data/WS/lm_fits/lm_opt_",option,"_iter_",outer_i)
    
    model_id <- (list.files(dir_path, pattern = "\\.Rda$") %>% length()) + 1
    file_path <- paste0(dir_path, "/gof_",model_id,".Rda")
    save(out, file = file_path) 
  }
  
  # output

  gc()
  
  return(val_rmse)
}



save_data <- function(x, final_data_path) {
  # outer_i, data_process, cv_folds, option are from global
  
  m <- x["m"] %>% unlist()
  l <- x["l"] %>% unlist()
  n <- x["n"] %>% unlist()
  account_cum_rain <- x["account_cum_rain"] %>% unlist()
  account_season <- x["account_season"] %>% unlist()
  
  # get dataset for training
  c(dtrain, dtest, val, train_index, test_index) %<-%
    prepare_train_test_data(m,
                            l,
                            n,
                            account_cum_rain,
                            account_season,
                            outer_i,
                            data_process,
                            cv_folds,
                            option)
  
  feature_names <- names(dtrain)[-1]
  
  save(dtrain, file = paste0(final_data_path, "dtrain.data"))
  
  save(dtest, file = paste0(final_data_path, "dtest.data"))
  
  write.table(
    feature_names,
    file = paste0(final_data_path, "feature_name.csv"),
    quote = F,
    row.names = F,
    col.names = F
  )
}


obj_fun <- makeSingleObjectiveFunction(
  fn = scoringFunction,
  par.set = makeParamSet(
    makeIntegerParam("m",                      lower= 144,  upper = 1440),
    makeIntegerParam("l",                      lower= 1,    upper = 36),
    makeIntegerParam("n",                      lower= 2,    upper = 36),
    makeIntegerParam("account_cum_rain",       lower= 0,    upper = 1),
    makeIntegerParam("account_season",         lower= 0,    upper = 1)
  ),
  has.simple.signature = FALSE,
  minimize = TRUE
)

opt_wrapper <- function(option,
                        outer_i,
                        cv_folds,
                        data_process,
                        n_iter = 100) {
  # build (if not exist) and clean the folder containing the simulation result
  
  des = generateDesign(
    n = 4 * getNumberOfParameters(obj_fun),
    par.set = getParamSet(obj_fun),
    fun = lhs::randomLHS
  )
  
  des$y = apply(des, 1, obj_fun)
  
  control <- makeMBOControl() %>%
    setMBOControlTermination(., iters = n_iter - 4 * getNumberOfParameters(obj_fun))
  
  run <- mbo(
    fun = obj_fun,
    design = des,
    control = control,
    show.info = TRUE
  )
  
  run
}


# Optimization ------------------------------------------------------------


for (option in 1:1){
  for (outer_i in 1:5){
    
    # prepare running
    temp_path <- paste0("./data/WS/lm_fits/lm_opt_", option, "_iter_", outer_i)
    
    if (!dir.exists(temp_path)) {
      dir.create(temp_path)
    }
    unlink(paste0(temp_path, "/*"))
    
    final_gof_path <- paste0(
      "./data/WS/lm_fits/gof_",
      "iter=",
      outer_i,
      "opt=",
      option,
      ".Rda"
    )
    
    run_path <- paste0(
      "./data/WS/lm_fits/run_",
      "iter=",
      outer_i,
      "opt=",
      option,
      ".run"
    )
    
    final_data_path <- paste0(
      "./data/WS/lm_fits/iter=",
      outer_i,
      "opt=",
      option
    )
    
    # run simulation
    run <- opt_wrapper(
      option,
      outer_i,
      cv_folds,
      data_process
    )
    
    # save results
    file.copy(from = paste0(temp_path, "/gof_", run$best.ind, ".Rda"),
              to = final_gof_path)
    
    save(run, file = run_path)
    
    save_data(run$x, final_data_path)
  }
}




