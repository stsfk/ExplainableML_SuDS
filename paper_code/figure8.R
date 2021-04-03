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

# Distribute SHAP and add prediction
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



# Analysis ----------------------------------------------------------------

N=5
option_chosen <- 1

data_analysiss <- shap_distributed %>%
  dplyr::filter(option == option_chosen)

data_plots <- vector("list", nrow(data_analysiss))

for (i in seq_along(data_plots)) {
  data_analysis <- data_analysiss$shap_ob_dist[[i]]
  
  data_plot1 <- data_analysis %>%
    mutate(interval = cut_number(pred, n = N)) %>%
    group_by(interval) %>%
    summarise_at(vars(contains("V")), function(x)
      mean(abs(x))) %>%
    t()
  
  data_plot1 <- as_tibble(data_plot1[-1,]) %>%
    mutate_all(as.numeric) %>%
    mutate(time_step = 0:1440) %>%
    gather(item, value, -time_step) %>%
    mutate(item = factor(
      item,
      levels = paste0("V", 1:N),
      labels = c("[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80,100]")
    ),
    importance_type = "impor")
  
  
  data_plot2 <- data_analysis %>%
    mutate(interval = cut_number(pred, n = N)) %>%
    group_by(interval) %>%
    summarise_at(vars(contains("V")), function(x)
      mean(x)) %>%
    t()
  
  data_plot2 <- as_tibble(data_plot2[-1, ]) %>%
    mutate_all(as.numeric) %>%
    mutate(time_step = 0:1440) %>%
    gather(item, value,-time_step) %>%
    mutate(item = factor(
      item,
      levels = paste0("V", 1:N),
      labels = c("[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80,100]")
    ),
    importance_type = "contr")
  
  data_plot <- data_plot1 %>%
    bind_rows(data_plot2) %>%
    mutate(
      importance_type = factor(
        importance_type,
        levels = c("impor", "contr"),
        labels = c("Average importance", "Average contribution")
      ),
      outer_i = data_analysiss$outer_i[[i]]
    )
  
  data_plots[[i]] <- data_plot
}

data_plot <- data_plots %>% 
  bind_rows()

# compute the mean value
data_plot_mean <- data_plot %>%
  group_by(time_step, importance_type, item) %>%
  dplyr::summarise(value = mean(value)) %>%
  mutate(outer_i = 0) # for binding rows

data_plot <- data_plot_mean %>%
  bind_rows(data_plot) %>%
  mutate(case = map_chr(outer_i, function(x)
    ifelse(x == 0, "Mean of all outer CV iterations", "Each outer CV iteration")))


ggplot(data_plot, aes(time_step, value, color = case, linetype = case, group = outer_i)) +
  facet_grid(importance_type~item) +
  geom_line(size = 0.25) +
  scale_linetype_manual(values = c("dashed", "solid")) +
  scale_color_manual(values = c("grey40", "red")) +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 1, 5, 10, 50, 100, 500, 1000),
    labels = c(0, 1, " ", 10, " ", 100, " ", 1000)
  ) +
  labs(x = "Time step in the past",
       y = "Rainfall's impact on discharge prediction [L/s]") +
  theme_bw(base_size = 8) +
  theme(legend.position = "top",
        legend.title = element_blank(),
        legend.key.height = unit(0.2, "line"))

# Output ------------------------------------------------------------------

ggsave(filename = "./data/WS/plot/figure8.png", width = 7, height = 3.6, units = "in", dpi = 600)
ggsave(filename = "./data/WS/plot/figure8.pdf", width = 7, height = 3.6, units = "in")


