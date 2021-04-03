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

tree_method <- "hist"

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


# Function ----------------------------------------------------------------
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

shap_observational <- function(option, optObjs){
  
  # `train_index`, `val_index`, `test_index`, is from global env
  # generate train, val, and test dataset
  
  # prepare data
  optObj <- optObjs[[option]]
  c(m, l, n) %<-% getBestPars(optObj)[c("m", "l", "n")]
  data_feature <- gen_feature(m, l, n , data_process, option)
  
  dall <- data_feature %>%
    .[complete.cases(.),]
  
  dall <- xgb.DMatrix(data = data.matrix(dall %>% dplyr::select(-Y)),
                      label = dall$Y)

  # load model
  model <- xgboost::xgb.load(
    paste0(
      "./data/SHC/model_fits/xgb_opt_",
      option,
      "/model_",
      optObj$scoreSummary$Score %>% which.max(),
      ".model"
    )
  )

  out <- predict(model, dall, predcontrib = T) %>%
    as_tibble()
  
  out
  
}


load("./data/SHC/xgb_fits.Rda")

assign_shap_time_step <- function(shap_matrix){
  
  mean_shap <- colMeans((shap_matrix))
  
  rain_depth_feature_ind <- names(shap_matrix) %>% str_detect("^X") %>% which()
  rain_features <- names(shap_matrix)[rain_depth_feature_ind]
  other_features <- names(shap_matrix)[-rain_depth_feature_ind]
  
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


shap_matrix <- shap_observational(4,optObjs)
data_plot <- assign_shap_time_step(abs(shap_matrix))

ggplot(data_plot, aes(time_step, shap)) +
  geom_line() +
  scale_x_log10()


