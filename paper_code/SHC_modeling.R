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
  ParBayesianOptimization,
  hydroGOF
)

tree_method <- "gpu_hist"

# data --------------------------------------------------------------------
# read rain
rain <- read_tsv("./data/SHC/SHC_amd_10min_intensity.dat", comment = ";",
                 col_names = c("station", "year", 	"month",	"day", 	"hour",	"minute",	"rain"),
                 col_types = cols(
                   station = col_character(),
                   year = col_double(),
                   month = col_double(),
                   day = col_double(),
                   hour = col_double(),
                   minute = col_double(),
                   rain = col_double()
                 )) %>%
  mutate(datetime = ymd_hm(paste(year, month, day, hour, minute))) %>%
  select(datetime, rain)

# fill missing value with 0
rain <- tibble(datetime = seq(from = ymd_hm("2009-01-10 05:50"), to = ymd_hm("2010-12-30 12:40"), by = 600)) %>%
  left_join(rain, by = "datetime") %>%
  mutate(rain = replace(rain, is.na(rain), 0)) %>%
  arrange(datetime) # inch/hour

# read flow
flow <- read_tsv("./data/SHC/C417_Measured_Flow.dat",
                 col_names = c("datetime", "flow"),
                 col_types = cols(
                   datetime = col_character(),
                   flow = col_double()
                 )) %>%
  mutate(datetime = ymd_hm(datetime),
         flow = flow/35.314666212661) %>% # CFS to CMS
  arrange(datetime)

# join rainfall and flow
data_process <- rain %>%
  left_join(flow, by = "datetime") %>%
  arrange(datetime) %>%
  rename(Y = flow, 
         X = rain)

# data split
val_index <- data_process %>% 
  mutate(ind = 1:n()) %>%
  dplyr::filter(datetime >= ymd_hm("2009-07-20 00:00") & datetime < ymd_hm("2009-07-29 00:00")) %>%
  dplyr::filter(!is.na(Y)) %>%
  pull(ind) # 1294 records

test_index <- data_process %>% 
  mutate(ind = 1:n()) %>%
  dplyr::filter(datetime >= ymd_hm("2009-08-01 00:00")) %>%
  dplyr::filter(!is.na(Y)) %>%
  pull(ind) #  4461 records

train_index <- data_process %>% 
  mutate(ind = 1:n()) %>%
  filter(!is.na(Y)) %>%
  pull(ind) %>%
  setdiff(c(test_index, val_index)) # 3166 records

# Functions ---------------------------------------------------------------

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

gen_feature <- function(m, l, n, data_process, option = 1) {
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

extract_DMatrix <- function(row_index, data_feature){
  # This function create xgb.DMatrix for row_index subset of data_process
  
  out <- data_feature[row_index,]
  out <- xgb.DMatrix(data = data.matrix(out %>% dplyr::select(-Y)),
              label = out$Y)
  
  out
}

scoringFunction <- function(eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma,
                            m, l, n,
                            save_model = T) {
  
  # `option`, `train_index`, `val_index`, `test_index`, is from global env
  # generate train, val, and test dataset
  
  data_feature <- gen_feature(m, l, n , data_process, option)
  
  c(dtrain, dval, dtest, dtrain_val) %<-% {
    list(train_index,
         val_index,
         test_index,
         c(train_index, val_index)) %>%
      lapply(extract_DMatrix, data_feature = data_feature)
  }
  
  ## experiment
  
  preprocess_cols <- rep(1, ncol(dtest) - 1)
  
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
  
  # train model on all training_outer fold
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
    dir_path <- paste0("./data/SHC/model_fits/xgb_opt_",option)
    
    model_id <- (list.files(dir_path) %>% length()) + 1
    file_path <- paste0(dir_path, "/model_",model_id,".model")
    
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


# Optimization ------------------------------------------------------------

eval_grid <- expand.grid(option = 1:4) %>%
  as_tibble()

optObjs <- vector("list", nrow(eval_grid))

for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]

  optObj <- optimization_wrapper(option, data_process)
  
  fname <- paste0("./data/SHC/model_fits/xgb_opt_", option, ".Rda")
  save(optObj, file = fname)
  
  optObjs[[i]] <- optObj
}

save(optObjs, file = "./data/SHC/xgb_fits.Rda")
