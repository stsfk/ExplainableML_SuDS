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
  hydroGOF,
  cowplot,
  ggthemes,
  viridis,
  hrbrthemes
)

# Data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")
load("./data/WS/cv_folds.Rda")
load("./data/WS/model_fits/ob_SHAPs.Rda")

# Function ----------------------------------------------------------------

get_predction <- function(option, outer_i, id){
  
  dtrain <- read_csv(
    paste0(
      "./data/WS/model_fits/xgb_opt_",
      option,
      "_iter_",
      outer_i,
      "/train_",
      id,
      ".csv"
    ),
    col_types = cols(.default = col_double())
  )
  
  dtest <- read_csv(
    paste0(
      "./data/WS/model_fits/xgb_opt_",
      option,
      "_iter_",
      outer_i,
      "/test_",
      id,
      ".csv"
    ),
    col_types = cols(.default = col_double())
  )
  
  
  model <- xgboost::xgb.load(
    paste0(
      "./data/WS/model_fits/xgb_opt_",
      option,
      "_iter_",
      outer_i,
      "/model_",
      id,
      ".model"
    )
  )
  
  # new_data is the data of both training and the test set
  new_data <- dtrain %>% 
    bind_rows(dtest) %>%
    data.matrix() %>%
    xgboost::xgb.DMatrix()
  
  predict(model, new_data)
}

distribute_shap <- function(ids, shap_matrix){
  
  out <- matrix(0, nrow = length(ids), ncol = 1441)
  
  for (i in seq_along(ids)) {
    
    # Get the corresponding rainfall time series
    id <- ids[[i]]
    record_id <- shap_matrix$record_id[id]
    
    p_series <- data_process$X[(record_id-1440):record_id] %>% rev()
    
    # Distribute SHAP of rainfall depth features to rainfall of each time step
    rain_depth_feature_ind <- names(shap_matrix) %>% str_detect("^X") %>% which()
    rainfall_depth_feature_shap <- shap_matrix[id,rain_depth_feature_ind] %>% unlist()
    
    s_e <- names(rainfall_depth_feature_shap) %>% 
      str_split("_", simplify = T) %>%
      str_extract("[0-9]+") %>%
      as.numeric() %>%
      matrix(ncol = 2)
    
    for (j in seq_along(rainfall_depth_feature_shap)){
      s <- s_e[j, 1] + 1
      e <- s_e[j, 2] + 1
      
      p_segment <- p_series[s:e]
      
      if (sum(p_segment)!= 0){
        weights <- p_segment/sum(p_segment)
      } else {
        weights <- rep(1/length(p_segment), length(p_segment))
      }
      
      out[i,s:e] <- out[i,s:e] + rainfall_depth_feature_shap[j] * weights
    }
  }
  
  out
}


# Ob SHAP -----------------------------------------------------------------

WS_SHAP_time_steps <- vector("list", 5)

for (iter in 1:5){
  # outputing relative importance
  shap_matrixs <- ob_SHAPs[[iter]]
  
  scaler <- shap_matrixs %>%
    select(-record_id, -BIAS, -iter) %>%
    data.matrix() %>% 
    abs() %>% 
    colMeans() %>% 
    sum()
    
  # Distribute SHAP
  shap_ob_dist <- distribute_shap(1:nrow(shap_matrixs), shap_matrixs) %>%
    as_tibble()
  
  # compute hourly SHAP
  temp <- shap_ob_dist %>%
    dplyr::select(V1:V1441) %>%
    data.matrix() %>%
    abs() %>%
    colMeans() %>%
    unname()
  
  temp <- temp/scaler
  
  WS_SHAP_time_steps[[iter]] <- tibble(
    importance = temp,
    iter = iter,
    method = "OB_SHAP",
    time_step = (1:length(temp)) - 1
  )
}


# Int SHAP ----------------------------------------------------------------

WS_SHAP_time_steps_int <- vector("list", 5)

for (iter in 1:5){
  # outputing relative importance
  feature_names <- ob_SHAPs[[iter]] %>%
    select(-record_id, -BIAS, -iter) %>%
    names()
  
  shap_matrixs <- read.csv(paste0("./data/WS/model_fits/int_SHAP_iter", iter, ".csv"), header = F) %>%
    as_tibble() %>%
    set_names(feature_names)
  
  shap_matrixs <- ob_SHAPs[[iter]] %>%
    select(record_id, BIAS, iter) %>%
    bind_cols(shap_matrixs)
  
  # Distribute SHAP
  shap_ob_dist <- distribute_shap(1:nrow(shap_matrixs), shap_matrixs) %>%
    as_tibble()
  
  # compute hourly SHAP
  temp <- shap_ob_dist %>%
    dplyr::select(V1:V1441) %>%
    data.matrix() %>%
    abs() %>%
    colMeans() %>%
    unname()
  
  # scale to relative importance
  scaler <- read.csv(paste0("./data/WS/model_fits/int_SHAP_iter", iter, ".csv"), header = F) %>% 
    abs() %>% 
    colMeans() %>% 
    sum()
  temp <- temp/scaler
  
  WS_SHAP_time_steps_int[[iter]] <- tibble(
    importance = temp,
    iter = iter,
    method = "INT_SHAP",
    time_step = (1:length(temp)) - 1
  )
}

# Other importance --------------------------------------------------------
trainable_record_id <- trainable_df$record_id %>%
  unlist()
average_rain <- rep(0,1441)
for (i in trainable_record_id){
  s <- i - 1440
  e <- i
  
  average_rain <- average_rain + rev(data_process$X[s:e])
}

average_rain <- average_rain/max(average_rain)

other_importance <- function(iter){
  
  model <- xgboost::xgb.load(paste0("./data/WS/model_fits/gof_iter=", iter, "opt=1.model"))
  
  feature_names <- ob_SHAPs[[iter]] %>%
    select(-record_id, -BIAS, -iter) %>%
    names()
  
  out <- xgboost::xgb.importance(feature_names =  feature_names, model = model)
  
  out[,-1] %>%
    t() %>%
    as_tibble() %>%
    set_names(out[,1] %>% unlist())
}

distribute_other_importance <- function(importance_matrix){
  
  out <- matrix(0, nrow = nrow(importance_matrix), ncol = 1441)
  
  for (i in 1:nrow(importance_matrix)) {
    
    # Get the corresponding rainfall time series
    p_series <- average_rain
    
    # Distribute SHAP of rainfall depth features to rainfall of each time step
    rain_depth_feature_ind <- names(importance_matrix) %>% str_detect("^X") %>% which()
    rainfall_depth_feature_impor <- importance_matrix[i,rain_depth_feature_ind] %>% unlist()
    
    s_e <- names(rainfall_depth_feature_impor) %>% 
      str_split("_", simplify = T) %>%
      str_extract("[0-9]+") %>%
      as.numeric() %>%
      matrix(ncol = 2)
    
    for (j in seq_along(rainfall_depth_feature_impor)){
      s <- s_e[j, 1] + 1
      e <- s_e[j, 2] + 1
      
      p_segment <- p_series[s:e]
      
      if (sum(p_segment)!= 0){
        weights <- p_segment/sum(p_segment)
      } else {
        weights <- rep(1/length(p_segment), length(p_segment))
      }
      
      out[i,s:e] <- out[i,s:e] + rainfall_depth_feature_impor[j] * weights
    }
  }
  
  out
}

# gain, cover, frequency
importance_matrixs <- vector("list", 5)
for (iter in 1:5){
  importance_matrixs[[iter]] <- other_importance(iter) %>%
    distribute_other_importance() %>%
    t() %>%
    as_tibble() %>%
    set_names(c("Gain", "Cover", "Frequency")) %>%
    mutate(time_step = c(0:1440),
           iter = iter)
}


# Plot --------------------------------------------------------------------



SHAP <- WS_SHAP_time_steps %>%
  bind_rows() %>%
  select(iter, time_step, SHAP=importance)

SHAP_int <- WS_SHAP_time_steps_int %>%
  bind_rows() %>%
  select(iter, time_step, SHAP_int=importance)

other_importance <- importance_matrixs %>%
  bind_rows()


data_plot <- other_importance %>%
  left_join(SHAP, by = c("time_step", "iter")) %>%
  left_join(SHAP_int, by = c("time_step", "iter")) %>%
  select(iter, time_step, everything()) %>%
  gather(item, value, Gain:SHAP_int)


data_plot <- data_plot  %>%
  mutate(
    item = factor(
      item,
      levels = c("Gain", "Cover", "Frequency", "SHAP", "SHAP_int"),
      labels = c("Gain", "Cover", "Frequency", "Observational SHAP (relative)", "Interventional SHAP (relative)")
    )
  )

ggplot(data_plot, aes(time_step, value, color = factor(iter)))+
  geom_line(size = 0.25) +
  facet_grid(~item) +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 1, 5, 10, 50, 100, 500, 1000),
    labels = c(0, 1, " ", 10, " ", 100, " ", 1000)
  ) +
  scale_color_manual(values = c("#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2")) +
  labs(x = "Time step in the past",
       y = "Average importance of rainfall for discharge prediction",
       color = "Outer CV iteration") +
  theme_bw(base_size = 8) +
  theme(legend.position = "top",
        legend.key.height = unit(0.2, "line"))


ggsave(filename = "./data/figures/figure8.png", width = 7, height = 3.5, units = "in", dpi = 600)
ggsave(filename = "./data/figures/figure8.pdf", width = 7, height = 3.5, units = "in")


