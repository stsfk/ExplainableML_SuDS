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
  hydroGOF
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

get_feature_names <- function(x, option = 1){
  
  m <- x["m"] %>% unlist()
  l <- x["l"] %>% unlist()
  n <- x["n"] %>% unlist()
  account_cum_rain <- x["account_cum_rain"] %>% unlist()
  account_season <- x["account_season"] %>% unlist()
  
  
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
  
  xgboost::xgb.DMatrix.save(dtrain, fname = paste0(final_data_path, "dtrain.data"))
  xgboost::xgb.DMatrix.save(dval, fname = paste0(final_data_path, "dval.data"))
  xgboost::xgb.DMatrix.save(dtest, fname = paste0(final_data_path, "dtest.data"))
  xgboost::xgb.DMatrix.save(dtrain_val, fname = paste0(final_data_path, "dtrain_val.data"))
}

assign_shap_time_step <- function(SHAP_m, SHAP_m_names){
  
  mean_shap <- colMeans(SHAP_m)
  
  rain_depth_feature_ind <- SHAP_m_names %>% str_detect("^X") %>% which()
  rain_features <- SHAP_m_names[rain_depth_feature_ind]
  other_features <- SHAP_m_names[-rain_depth_feature_ind]
  
  out <- tibble(
    time_step = 0:1440,
    shap = 0
  )
  
  for (j in seq_along(rain_features)) {
    rain_feature <- rain_features[j]
    s_e <- rain_feature %>% str_split("_", simplify = T) %>%
      str_extract("[0-9]+") %>%
      as.numeric()
    c(s, e) %<-% s_e
    
    out[(s + 1):(e + 1), 2] <-
      out[(s + 1):(e + 1), 2] + mean_shap[j] / length(s:e) # SHAP to be distributed
  }
  
  out
}


# data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")

prop <- 5
split <- 1
repeat_id <- 2

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


# load model
model <- xgboost::xgb.load(final_model_path)

# load run results
load(run_path)
run$x

# load data
dtest <- xgboost::xgb.DMatrix(paste0(final_data_path, "dtest.data"))
ob <- xgboost::getinfo(dtest, "label")

pred <- predict(model, dtest)
SHAP_m <- predict(model, dtest, predcontrib=T)

feature_name <-
  read.csv(paste0(final_data_path, "feature_name.csv")) %>% unlist() %>% unname()
SHAP_m_names <- c(feature_name, "bias")


out <- assign_shap_time_step(abs(SHAP_m), SHAP_m_names)


ggplot(out, aes(time_step, shap)) +
  geom_line() +
  scale_x_log10()
