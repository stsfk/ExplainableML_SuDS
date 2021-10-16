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

# data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")

load("./data/WS/inconsist_exp/split_prop=5split=1.Rda")

# compute the average rain recorded for the test dataset
test_record_id <- test_df$record_id %>% unlist()

# functions ---------------------------------------------------------------

distribute_shap <- function(shap_matrix, ids){
  # Distribute SHAP of rainfall depth features to rainfall of each time step
  
  # shap_maxtrix: absolute value of the SHAP metrics for dtest
  # ids: the record_id of samples in dtest
  
  # out: 
  out <- matrix(0, nrow = length(ids), ncol = 1441)
  
  # get the s and e index (t-a, t-b in the paper) from the feature names
  rain_depth_feature_ind <- colnames(shap_matrix) %>% str_detect("^X") %>% which()
  
  s_e <- colnames(shap_matrix)[rain_depth_feature_ind] %>% 
    str_split("_", simplify = T) %>%
    str_extract("[0-9]+") %>%
    as.numeric() %>%
    matrix(ncol = 2)
  
  for (i in seq_along(ids)) {
    rainfall_depth_feature_shap <- shap_matrix[i,rain_depth_feature_ind] %>% unlist()
    
    # Get the corresponding rainfall time series, from t-0 to t-1440
    record_id <- ids[[i]]
    p_series <- data_process$X[(record_id-1440):record_id] %>% rev()
    
    # Distribute SHAP value of each rainfall depth feature
    for (j in seq_along(rainfall_depth_feature_shap)){
      
      # get the p_segment based on s and e index
      s <- s_e[j, 1] + 1
      e <- s_e[j, 2] + 1
      
      p_segment <- p_series[s:e]
      
      # distribute proportional to rainfall depth
      if (sum(p_segment)!= 0){
        weights <- p_segment/sum(p_segment)
      } else {
        weights <- rep(1/length(p_segment), length(p_segment))
      }
      
      # fill the i th row of out
      out[i,s:e] <- out[i,s:e] + rainfall_depth_feature_shap[j] * weights
    }
  }
  
  out
}

compute_importance_consistency <- function(x, thd=5e-5){
  # after the peak, the impact on prediction should decrease
  x_diff <- x[which.max(x):length(x)] %>%
    diff()
  
  if (max(x_diff) <= thd){
    out1 <- T
  } else {
    out1 <- F
  }
  
  # before the peak, the impact on prediction should increase
  if (which.max(x) > 1){
    x_diff2 <- x[1:which.max(x)] %>%
      diff()
  } else {
    x_diff2 <- x[1]
  }
  
  if (min(x_diff2) >= -thd){
    out2 <- T
  } else {
    out2 <- F
  }
  
  # if consistent, out1 and out2 should both be true
  all(out1, out2)
}

consistency_gof_wrapper <- function(prop,split,repeat_id){
  load("./data/WS/ready_for_training.Rda")
  
  # paths of the modeling results
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
  
  # load data and extract observation
  dtest <- xgboost::xgb.DMatrix(paste0(final_data_path, "dtest.data"))
  ob <- xgboost::getinfo(dtest, "label")
  
  # get predcition
  pred <- predict(model, dtest)
  
  # compute shap_matrix, the absolute values are used
  shap_matrix <- predict(model, dtest, predcontrib=T) %>%
    abs()
  
  # assign names to shap_matrix
  feature_name <-
    read.csv(paste0(final_data_path, "feature_name.csv"), header = F) %>% unlist() %>% unname()
  SHAP_m_names <- c(feature_name, "bias")
  colnames(shap_matrix) <- SHAP_m_names
  
  # distribute SHAP values to rainfall of each time step
  distributed_SHAP <- distribute_shap(shap_matrix, test_record_id)
  
  # return results
  tibble(
    consistency = compute_importance_consistency(colMeans(distributed_SHAP)),
    nse = hydroGOF::NSE(pred, ob)
  )
}

# data --------------------------------------------------------------------

eval_grid <- expand.grid(
  prop = c(5,10,20,30),
  split = 1:10,
  repeat_id = 1:10
)

asses_results <- vector("list", nrow(eval_grid))

for (i in 1:nrow(eval_grid)){
  prop <- eval_grid$prop[i]
  split <- eval_grid$split[i]
  repeat_id <- eval_grid$repeat_id[i]
  
  asses_results[[i]] <- consistency_gof_wrapper(prop,split,repeat_id)
  
  gc()
}

data_plot <- asses_results %>%
  bind_rows() %>%
  bind_cols(eval_grid)

data_plot %>%
  ggplot(aes(nse, consistency)) +
  geom_boxplot() +
  geom_point()+
  facet_grid(prop ~split)+
  coord_flip()

data_plot %>%
  ggplot(aes(nse, consistency)) +
  geom_boxplot()+
  geom_point()+
  facet_grid(~prop)+
  coord_flip()

data_plot %>%
  group_by(prop) %>%
  summarise(consistency = sum(consistency))

save(data_plot, file = "./data/WS/inconsist_exp/data_plot.Rda")


# recycle -----------------------------------------------------------------

