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

optimal_candidate_id <- read_csv("./data/WS/model_fits/optimal_candidate_id.csv",
                                 col_types = cols("i", "i", "i"))


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


# Process -----------------------------------------------------------------

# Distribute SHAP and add 
shap_distributed <- shap_matrixs %>%
  dplyr::select(option, outer_i, id) %>%
  mutate(
    shap_ob_dist = vector("list",1)
  )

for (i in 1:nrow(shap_distributed)){
  
  c(option, outer_i, id) %<-% optimal_candidate_id[i,]
  
  shap_ob <- shap_matrixs$shap_ob[[i]]
  
  shap_ob_dist <- distribute_shap(1:nrow(shap_ob), shap_ob) %>%
    as_tibble() %>%
    mutate(record_id = shap_ob$record_id,
           pred = get_predction(option, outer_i, id),
           ob = data_process$Y[shap_ob$record_id])
  
  shap_distributed$shap_ob_dist[[i]] <- shap_ob_dist 
}

# compute the rainfall age
rainfall_age <- function(event_id_chosen, model_i = 1, output_hour = T){
  # shap_distributed from global
  distributed_impor_m <- shap_distributed$shap_ob_dist[[model_i]] %>%
    dplyr::select(V1:V1441) %>%
    data.matrix() %>%
    abs()
  record_id_lookup <- shap_distributed$shap_ob_dist[[model_i]]$record_id
  
  row_index <- which(record_id_lookup == event_id_chosen)
  importance_series <- distributed_impor_m[row_index, ]
  
  out <- sum(importance_series * (0:1440))/sum(importance_series)
  
  if (output_hour){
    out/6
  } else {
    out
  }
}

# discharge prediction
model_i = 1
pred <- shap_distributed$shap_ob_dist[[model_i]]$pred
record_id_lookup <- shap_distributed$shap_ob_dist[[model_i]]$record_id


event_id_large <- trainable_df[trainable_df$peak_flow %>% which.max(),] %>%
  dplyr::pull(record_id) %>%
  unlist()

event_id_mid <- trainable_df[17,] %>%
  dplyr::pull(record_id) %>%
  unlist()

event_id_small <- trainable_df[29,] %>%
  dplyr::pull(record_id) %>%
  unlist()


data_plot_l <- tibble(
  time_step = seq_along(event_id_large),
  rain = data_process$X[event_id_large],
  discharge = pred[which(record_id_lookup %in% event_id_large)],
  rainfall_age = sapply(event_id_large, rainfall_age)
) %>%
  dplyr::filter(time_step <=100) %>%
  mutate(case = "Large runoff event")

data_plot_m <- tibble(
  time_step = seq_along(event_id_mid),
  rain = data_process$X[event_id_mid],
  discharge = pred[which(record_id_lookup %in% event_id_mid)],
  rainfall_age = sapply(event_id_mid, rainfall_age)
) %>%
  dplyr::filter(time_step <=300) %>%
  mutate(case = "Medium runoff event")

data_plot_s <- tibble(
  time_step = seq_along(event_id_small),
  rain = data_process$X[event_id_small],
  discharge = pred[which(record_id_lookup %in% event_id_small)],
  rainfall_age = sapply(event_id_small, rainfall_age)
) %>%
  dplyr::filter(time_step <=250) %>%
  mutate(case = "Small runoff event")


data_plot <- data_plot_l %>%
  bind_rows(data_plot_m) %>%
  bind_rows(data_plot_s) %>%
  gather(item, value, -time_step, -case) %>%
  mutate(item = factor(
    item,
    levels = c("rain", "discharge", "rainfall_age"),
    labels = c("Rainfall intensity\n[mm/h]", "Predicted discharge [L/s]", "Average age of rainfall\naffecting prediction [h]")
  ))

ggplot(data_plot, aes(time_step/6, value))+
  geom_line(size = 0.35)+
  facet_grid(item~case, scales = "free",  switch="y") +
  labs(x = "Time since the beginning of runoff event [h]") +
  theme_bw(base_size = 8) +
  theme(axis.title.y = element_blank(),
        strip.placement = "outside",
        strip.background.y = element_blank())

ggsave(filename = "./data/WS/plot/figure10.png", width = 7, height = 4.5, units = "in", dpi = 600)
ggsave(filename = "./data/WS/plot/figure10.pdf", width = 7, height = 4.5, units = "in")




# Analyze hyperparameter selection ----------------------------------------

# feature selected times
eval_grid <- expand.grid(
  option = 1:4,
  outer_i = 1:5
) %>%
  as_tibble()

outs <- vector("list", nrow(eval_grid))
for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]
  outer_i <- eval_grid$outer_i[i]
  load(paste0("./data/WS/model_fits/xgb_opt_", option, "_iter_", outer_i, ".Rda"))
  
  outs[[i]] <- optObj$scoreSummary %>%
    as_tibble() %>%
    dplyr::select(m:account_season, Score) %>%
    dplyr::slice(which.max(Score)) 
  
}
outs <- outs %>%
  bind_rows()


# maximum contributions
load("./data/WS/shap_matrixs.Rda")

out1 <- rep(0,20)
out2 <- rep(0,20)
for (i in 1:20){
  
  shap_ob <- shap_matrixs[i,]$shap_ob[[1]] %>%
    dplyr::select(contains("X")|contains("cum")|contains("month"))
  
  shap_ob_sub<-shap_matrixs[i,]$shap_ob[[1]] %>%
    dplyr::select(contains("X"))
  
  shap_int <- shap_matrixs[i,]$shap_int[[1]] %>%
    dplyr::select(contains("X")|contains("cum")|contains("month"))
  
  shap_int_sub <- shap_matrixs[i,]$shap_int[[1]] %>%
    dplyr::select(contains("X"))
  
  out1[[i]] <- 1-sum(abs(shap_ob_sub))/sum(abs(shap_ob))
  out2[[i]] <- 1-sum(abs(shap_int_sub))/sum(abs(shap_int))  
}

sum(out1!=0)
sum(out2!=0)
out1 %>% max()
out2 %>% max()
