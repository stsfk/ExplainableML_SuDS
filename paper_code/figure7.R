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
  ggthemes
)

# Data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")
load("./data/WS/cv_folds.Rda")

load("./data/WS/shap_matrixs.Rda")


trainable_record_id <- trainable_df$record_id %>%
  unlist()

average_rain <- rep(0,1441)
for (i in trainable_record_id){
  s <- i - 1440
  e <- i
  
  average_rain <- average_rain + rev(data_process$X[s:e])
}

average_rain <- average_rain/max(average_rain)

# Function ----------------------------------------------------------------

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

other_importance <- function(option, outer_i, id){
  
  feature_names <- read_csv(
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
  ) %>%
    names()
  
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
  
  out <- xgboost::xgb.importance(feature_names =  feature_names, model = model)
  
  out %>%
    as_tibble() %>%
    mutate(option = option,
           outer_i = outer_i)
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


shap_ob_sums <- sapply(shap_matrixs$shap_ob, function(x)
  x %>%
    dplyr::select(-record_id, -BIAS) %>%
    abs() %>%
    colMeans() %>%
    sum())

shap_int_sums <- sapply(shap_matrixs$shap_int, function(x)
  x %>%
    dplyr::select(-record_id) %>%
    abs() %>%
    colMeans() %>%
    sum())


# Prepare distributed SHAP ------------------------------------------------


# SHAP importance

shap_distributed <- shap_matrixs %>%
  dplyr::select(option, outer_i, id) %>%
  mutate(
    ob_impor = vector("list",1),
    int_impor = vector("list",1)
  )

for (i in 1:nrow(shap_distributed)){
  
  shap_ob <- shap_matrixs$shap_ob[[i]]
  shap_int <- shap_matrixs$shap_int[[i]]
  
  shap_ob_dist <- distribute_shap(1:nrow(shap_ob), shap_ob)
  shap_int_dist <- distribute_shap(1:nrow(shap_int), shap_int)
  
  shap_distributed$ob_impor[[i]] <- abs(shap_ob_dist) %>%
    colMeans()/shap_ob_sums[i]
  
  shap_distributed$int_impor[[i]] <- abs(shap_int_dist) %>%
    colMeans()/shap_int_sums[i]
}

add_time_step_dist_shap <- function(col){
  col <- dplyr::enquo(col)
  shap_distributed %>%
    dplyr::select(option, outer_i, !!col) %>%
    unnest(cols = !!col) %>%
    mutate(time_step = rep(0:1440, nrow(shap_distributed)))
}

ob_impor <- add_time_step_dist_shap(ob_impor)
int_impor <- add_time_step_dist_shap(int_impor)



# Other importance measures

optimal_candidate_id <-
  read_csv("./data/WS/model_fits/optimal_candidate_id.csv",
           col_types = cols("i", "i", "i"))

data_other_importance <- vector("list", nrow(optimal_candidate_id))
for (i in 1:nrow(optimal_candidate_id)){
  
  c(option, outer_i, id) %<-% optimal_candidate_id[i, ]
  
  importance_matrix <- other_importance(option, outer_i, id)
  
  feature_names <- importance_matrix$Feature
  importance_matrix <- importance_matrix %>%
    dplyr::select(Gain, Cover, Frequency) %>%
    t() 
  
  rownames(importance_matrix) <- NULL
  importance_matrix <- importance_matrix %>% 
    as_tibble()
  colnames(importance_matrix) <- feature_names
  
  # data_plot_observational_relative
  data_other_importance[[i]] <- distribute_other_importance(importance_matrix) %>% 
    t() %>%
    as_tibble() %>%
    set_names(c("Gain", "Cover", "Frequency")) %>%
    mutate(time_step = c(0:1440),
           option = option,
           outer_i = outer_i)
}


data_other_importance <- data_other_importance %>%
  bind_rows()

# Plot --------------------------------------------------------------------

data_plot <- data_other_importance %>%
  left_join(ob_impor, by = c("option", "outer_i", "time_step")) %>%
  left_join(int_impor, by = c("option", "outer_i", "time_step")) %>%
  select(option, outer_i, time_step, everything()) %>%
  gather(item, value, Gain:int_impor)


data_plot_mean <- data_plot %>%
  group_by(time_step, option, item) %>%
  dplyr::summarise(value = mean(value)) %>%
  mutate(outer_i = 0) %>% # for binding rows
  ungroup()


data_plot <- data_plot_mean %>%
  bind_rows(data_plot) %>%
  mutate(case = map_chr(outer_i, function(x)
    ifelse(x == 0, "Mean of all outer CV iterations", "Each outer CV iteration"))) %>%
  mutate(option = factor(
    option,
    levels = c(1:4),
    labels = str_c("Aggregation option ", 1:4)
  )) %>%
  mutate(
    item = factor(
      item,
      levels = c("Gain", "Cover", "Frequency", "int_impor", "ob_impor"),
      labels = c("Gain", "Cover", "Frequency", "Interventional SHAP (relative)", "Observational SHAP (relative)")
    )
  )

ggplot(data_plot, aes(time_step, value, group = outer_i, color = case, linetype = case))+
  geom_line(size = 0.25) +
  scale_linetype_manual(values = c("dashed", "solid")) +
  scale_color_manual(values = c("grey40", "red"))   + 
  facet_grid(option~item)+
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 1, 5, 10, 50, 100, 500, 1000),
    labels = c(0, 1, " ", 10, " ", 100, " ", 1000)
  ) +
  labs(x = "Time step in the past",
       y = "Average importance of rainfall for discharge prediction") +
  theme_bw(base_size = 8) +
  theme(legend.position = "top",
        legend.title = element_blank(),
        legend.key.height = unit(0.2, "line"))

ggsave(filename = "./data/WS/plot/figure7.png", width = 7, height = 5, units = "in", dpi = 600)
ggsave(filename = "./data/WS/plot/figure7.pdf", width = 7, height = 5, units = "in")
