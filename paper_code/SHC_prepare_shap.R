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


load_optObj <- function(option){
  load(paste0("./data/SHC/model_fits/xgb_opt_", option, ".Rda"))
  
  optObj
}

prepare_shap_input <- function(option, best_only = T){
  optObj <- load_optObj(option)
  
  feature_engineering_hyperparas <- optObj$scoreSummary %>%
    as_tibble() %>%
    mutate(id = 1:n())
  
  if (best_only){
    feature_engineering_hyperparas <- feature_engineering_hyperparas %>%
      dplyr::arrange(desc(Score)) %>%
      dplyr::select(m:n, id) %>%
      dplyr::slice(1)
  } else {
    feature_engineering_hyperparas <- feature_engineering_hyperparas %>%
      dplyr::select(m:account_season)
  }
  
  for (j in 1:nrow(feature_engineering_hyperparas)){
    
    c(m, l, n, id) %<-% (feature_engineering_hyperparas %>% 
                                                             dplyr::slice(j))
    
    shap_input <- gen_feature(m, l, n, data_process, option) %>%
      dplyr::select(-Y)
    
    # subset by train and test index
    dtrain <- shap_input[c(train_index, val_index),]
    dtest <- shap_input[test_index,]
    
    write_csv(dtrain, file = paste0("./data/SHC/model_fits/xgb_opt_", option, "/train_", id, ".csv"), col_names = T)
    write_csv(dtest, file = paste0("./data/SHC/model_fits/xgb_opt_", option, "/test_", id, ".csv"), col_names = T)  
  }
}

# Modeling ----------------------------------------------------------------

eval_grid <- expand.grid(option = 1:4) %>%
  as_tibble()

for (i in 1:nrow(eval_grid)){
  
  option <- eval_grid$option[i]
  
  prepare_shap_input(option)
}

# Id of best candidate ----------------------------------------------------

eval_grid <- expand.grid(option = 1:4) %>%
  as_tibble()

outs <- vector("list", 4)

for (i in 1:nrow(eval_grid)){
  
  option <- eval_grid$option[i]
  optObj <- load_optObj(option)
  
  feature_engineering_hyperparas <- optObj$scoreSummary %>%
    as_tibble() %>%
    mutate(id = 1:n())
  
  outs[[i]] <- feature_engineering_hyperparas %>%
    dplyr::arrange(desc(Score)) %>%
    dplyr::select(id) %>%
    dplyr::slice(1) %>%
    mutate(option = option) %>%
    dplyr::select(option, id)
}


optimal_candidate_id <- outs %>%
  bind_rows()

write_csv(optimal_candidate_id, file = "./data/SHC/model_fits/optimal_candidate_id.csv")


# Cleanup dtrain and dtest csv --------------------------------------------

eval_grid <- expand.grid(option = 1:4,
                         outer_i = 1:5) %>%
  as_tibble()

cleanup_shap_input <- function(option, outer_i){
  fpath <- paste0("./data/WS/model_fits/xgb_opt_", option, "_iter_", outer_i)
  fpattern <- "*.csv"
  
  file.path(fpath, dir(fpath, fpattern)) %>%
    file.remove() %>%
    any()
}


