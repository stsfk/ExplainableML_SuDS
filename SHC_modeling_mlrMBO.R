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

tree_method <- "gpu_hist"

set.seed(12345)

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

scoringFunction <- function(x, save_model = T) {
  
  # `option`, `train_index`, `val_index`, `test_index`, is from global env
  # generate train, val, and test dataset
  
  eta <- x["eta"] %>% unlist()
  max_depth <- x["max_depth"] %>% unlist()
  min_child_weight <- x["min_child_weight"] %>% unlist()
  subsample <- x["subsample"] %>% unlist()
  colsample_bytree <- x["colsample_bytree"] %>% unlist()
  gamma <- x["gamma"] %>% unlist()
  
  m <- x["m"] %>% unlist()
  l <- x["l"] %>% unlist()
  n <- x["n"] %>% unlist()
  
  data_feature <- gen_feature(m, l, n , data_process, option)
  
  c(dtrain, dval, dtest, dtrain_val) %<-% {
    list(train_index,
         val_index,
         test_index,
         c(train_index, val_index)) %>%
      lapply(extract_DMatrix, data_feature = data_feature)
  }
  
  ## experiment
  
  preprocess_cols <- rep(1, ncol(data_feature) - 1)
  
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
  
  # summary
  out <- list(
    Score = -score,
    best_iteration = best_iteration,
    train_pred = list(train_pred),
    test_pred = list(test_pred)
  )
  
  # save model
  if (save_model){
    dir_path <- paste0("./data/SHC/model_fits/xgb_opt_",option)
    
    model_id <- (list.files(dir_path, pattern = "\\.model$") %>% length()) + 1
    file_path <- paste0(dir_path, "/model_",model_id,".model")
    xgb.save(model = xgbFit, fname = file_path)
    
    file_path <- paste0(dir_path, "/gof_",model_id,".Rda")
    save(out, file = file_path) 
  }

  gc()
  
  return(score)
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
    makeIntegerParam("n",                      lower= 2,    upper = 36)
  ),
  has.simple.signature = FALSE,
  minimize = TRUE
)


opt_wrapper <- function(option,
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

save_dMatrix <- function(x, final_data_path) {
  # outer_i, data_process, cv_folds, option are from global
  
  m <- x["m"] %>% unlist()
  l <- x["l"] %>% unlist()
  n <- x["n"] %>% unlist()
  account_cum_rain <- x["account_cum_rain"] %>% unlist()
  account_season <- x["account_season"] %>% unlist()
  
  data_feature <- gen_feature(m, l, n , data_process, option)
  feature_names <- names(data_feature)[-1]
  
  # get dataset for training
  c(dtrain, dval, dtest, dtrain_val) %<-% {
    list(train_index,
         val_index,
         test_index,
         c(train_index, val_index)) %>%
      lapply(extract_DMatrix, data_feature = data_feature)
  }

  xgboost::xgb.DMatrix.save(dtrain_val, fname = paste0(final_data_path, "dtrain_val.data"))
  
  xgboost::xgb.DMatrix.save(dtest, fname = paste0(final_data_path, "dtest.data"))
  
  write.table(
    feature_names,
    file = paste0(final_data_path, "feature_name.csv"),
    quote = F,
    row.names = F,
    col.names = F
  )
}


# Optimization ------------------------------------------------------------

for (option in 1:1){
  # prepare running
  temp_path <- paste0("./data/SHC/model_fits/xgb_opt_", option)
  
  if (!dir.exists(temp_path)) {
    dir.create(temp_path)
  }
  unlink(paste0(temp_path, "/*"))
  
  final_gof_path <- paste0(
    "./data/SHC/model_fits/gof_",
    "opt=",
    option,
    ".Rda"
  )
  
  final_model_path <- paste0(
    "./data/SHC/model_fits/gof_",
    "opt=",
    option,
    ".model"
  )
  
  run_path <- paste0(
    "./data/SHC/model_fits/run_",
    "opt=",
    option,
    ".run"
  )
  
  final_data_path <- paste0(
    "./data/SHC/model_fits/opt=",
    option
  )
  
  # run simulation
  run <- opt_wrapper(
    option,
    data_process,
    n_iter = 100
  )
  
  # save results
  file.copy(from = paste0(temp_path, "/model_", run$best.ind, ".model"),
            to = final_model_path, overwrite = T)
  
  file.copy(from = paste0(temp_path, "/gof_", run$best.ind, ".Rda"),
            to = final_gof_path, overwrite = T)
  
  save(run, file = run_path)
  
  save_dMatrix(run$x, final_data_path)
}
