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
  ParBayesianOptimization,
  hydroGOF
)


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

extract_data_matrix <- function(row_index, data_feature){
  # This function create data.matrix for row_index subset of data_process
  
  data_feature[row_index,] 
}

scoringFunction <- function(m, l, n) {
  
  # `option`, `train_index`, `val_index`, `test_index`, is from global env
  # generate train, val, and test dataset
  
  data_feature <- gen_feature(m, l, n , data_process, option)
  
  c(dtrain, dval, dtest, dtrain_val) %<-% {
    list(train_index,
         val_index,
         test_index,
         c(train_index, val_index)) %>%
      lapply(extract_data_matrix, data_feature = data_feature)
  }
  
  ## experiment
  
  # Fit to dtrain only, and validate on dval
  lmFit <- lm(Y~., data = dtrain)
  
  score <- hydroGOF::gof(
    sim = predict(lmFit, dval),
    obs = dval$Y,
    digits = 8
  )[4]
  
  # Fit to dtrain and dval, and test on dtest
  lmFit <- lm(Y~., data = dtrain_val)
  
  
  # predictions
  train_pred <- tibble(
    datetime = data_process$datetime[c(train_index, val_index)],
    ob = dtrain_val$Y,
    pred = predict(lmFit, dtrain_val)) %>%
    arrange(datetime)
  
  test_pred <- tibble(
    datetime = data_process$datetime[test_index],
    ob = dtest$Y,
    pred = predict(lmFit, dtest)) %>%
    arrange(datetime)
  
  # output
  out <- list(
    Score = -score,
    train_pred = list(train_pred),
    test_pred = list(test_pred),
    coefficients = list(lmFit$coefficients)
  )
  
  gc()
  
  return(out)
}

optimization_wrapper <- function(option, data_process){
  
  # optimization hyperparas
  initPoints <- 50 # 12 experiment
  max_iter <- 100 # 100 experiment
  patience <- 50
  plotProgress <- T
  
  bounds <- list(
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
  
  fname <- paste0("./data/SHC/lm_fits/lm_", option, ".Rda")
  save(optObj, file = fname)
  
  optObjs[[i]] <- optObj
}

save(optObjs, file = "./data/SHC/lm_fits/lm_fits.Rda")
