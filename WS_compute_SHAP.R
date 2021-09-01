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
  hydroGOF,
  cowplot,
  ggthemes
)

# Data --------------------------------------------------------------------

load("./data/WS/ready_for_training.Rda")
load("./data/WS/cv_folds.Rda")


# observational SHAP ------------------------------------------------------


compute_ob_SHAP <- function(iter, option){
  data_all <-
    read_csv(paste0(
      "./data/WS/model_fits/iter=",
      iter,
      "opt=",
      option,
      "_train_test.csv"
    ))
  
  # load model
  model <- xgboost::xgb.load(
    paste0(
      "./data/WS/model_fits/gof_iter=",
      iter,
      "opt=",
      option,
      ".model"
    )
  )
  
  # predict SHAP
  SHAP <- predict(model, data_all[-1] %>% data.matrix(), predcontrib = T) %>%
    as_tibble()
  
  # join record index
  train_event_index <- analysis(cv_folds$splits[[iter]])$record_id %>% unlist()
  test_event_index <- assessment(cv_folds$splits[[iter]])$record_id %>% unlist()
  
  SHAP %>%
    mutate(record_id = c(train_event_index, test_event_index)) %>%
    select(record_id, everything())
}

ob_SHAPs <- vector("list", 5)

for (iter in 1:5){
  ob_SHAPs[[iter]] <- compute_ob_SHAP(iter, option = 1) %>%
    mutate(iter = iter)
}

save(ob_SHAPs, file = "./data/WS/model_fits/ob_SHAPs.Rda")

















shap_observational <- function(option, outer_i){
  
  load(paste0("./data/WS/model_fits/run_iter=", outer_i, "opt=", option, ".run"))
  
  
  # predict contri for both training and the test sets

  
  # adding record id index
  train_event_index <- analysis(cv_folds$splits[[outer_i]])$record_id %>% unlist()
  test_event_index <- assessment(cv_folds$splits[[outer_i]])$record_id %>% unlist()
  
  out %>%
    mutate(record_id = c(train_event_index, test_event_index)) %>%
    select(record_id, everything())
}

shap_interventional <- function(option, outer_i, id){
  # read dtrain to get the name of the features
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
  
  shap_matrix_train <- read_csv(
    paste0(
      "./data/WS/model_fits/xgb_opt_",
      option,
      "_iter_",
      outer_i,
      "/shap_train_",
      id,
      ".csv"
    ),
    col_types = cols(.default = col_double()),
    col_names = names(dtrain) 
  )
  
  shap_matrix_test <- read_csv(
    paste0(
      "./data/WS/model_fits/xgb_opt_",
      option,
      "_iter_",
      outer_i,
      "/shap_test_",
      id,
      ".csv"
    ),
    col_types = cols(.default = col_double()),
    col_names = names(dtrain) 
  )
  
  shap_matrix <- shap_matrix_train %>%
    bind_rows(shap_matrix_test)

  
  
  # adding record id index
  train_event_index <- analysis(cv_folds$splits[[outer_i]])$record_id %>% unlist()
  test_event_index <- assessment(cv_folds$splits[[outer_i]])$record_id %>% unlist()
  
  
  shap_matrix %>%
    mutate(record_id = c(train_event_index, test_event_index)) %>%
    select(record_id, everything())
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


# SHC save SHAP -----------------------------------------------------------

optimal_candidate_id <- read_csv("./data/WS/model_fits/optimal_candidate_id.csv",
                                 col_types = cols("i", "i", "i"))

shap_matrixs <- optimal_candidate_id %>%
  mutate(shap_ob = vector("list", 1),
         shap_int = vector("list", 1))

for (i in 1:nrow(optimal_candidate_id)){
  
  c(option, outer_i, id) %<-% optimal_candidate_id[i, ]
  
  shap_matrixs$shap_ob[[i]] <- shap_observational(option, outer_i, id)
  shap_matrixs$shap_int[[i]] <- shap_interventional(option, outer_i, id)
  
}

save(shap_matrixs, file = "./data/WS/shap_matrixs.Rda")


# Prepare plot ------------------------------------------------------------


load("./data/WS/shap_matrixs.Rda")


shap_distributed <- shap_matrixs %>%
  dplyr::select(option, outer_i, id) %>%
  mutate(
    ob_contr = vector("list",1),
    ob_impor = vector("list",1),
    int_contr = vector("list",1),
    int_impor = vector("list",1)
  )

for (i in 1:nrow(shap_distributed)){
  
  shap_ob <- shap_matrixs$shap_ob[[i]]
  shap_int <- shap_matrixs$shap_int[[i]]
  
  shap_ob_dist <- distribute_shap(1:nrow(shap_ob), shap_ob)
  shap_int_dist <- distribute_shap(1:nrow(shap_int), shap_int)
  
  shap_distributed$ob_contr[[i]] <- shap_ob_dist %>%
    colMeans()
  shap_distributed$ob_impor[[i]] <- abs(shap_ob_dist) %>%
    colMeans()
  
  shap_distributed$int_contr[[i]] <- shap_int_dist %>%
    colMeans()
  shap_distributed$int_impor[[i]] <- abs(shap_int_dist) %>%
    colMeans()
}


add_time_step_dist_shap <- function(col){
  col <- dplyr::enquo(col)
  shap_distributed %>%
    dplyr::select(option, outer_i, !!col) %>%
    unnest(cols = !!col) %>%
    mutate(time_step = rep(0:1440, nrow(shap_distributed)))
}

ob_contr <- add_time_step_dist_shap(ob_contr)
ob_impor <- add_time_step_dist_shap(ob_impor)
int_contr <- add_time_step_dist_shap(int_contr)
int_impor <- add_time_step_dist_shap(int_impor)


# p1 ----------------------------------------------------------------------

data_plot_interventional <- int_impor %>%
  bind_rows() %>%
  dplyr::mutate(method = "Interventional") %>%
  rename(shap = int_impor)

data_plot_observational <- ob_impor %>%
  bind_rows() %>%
  dplyr::mutate(method = "Observational") %>%
  rename(shap = ob_impor)

data_plot <- data_plot_interventional %>%
  bind_rows(data_plot_observational)

data_plot_mean <- data_plot %>%
  group_by(time_step, option, method) %>%
  dplyr::summarise(shap = mean(shap)) %>%
  mutate(outer_i = 0) # for binding rows

data_plot <- data_plot_mean %>%
  bind_rows(data_plot) %>%
  mutate(case = map_chr(outer_i, function(x)
    ifelse(x == 0, "Mean of all outer CV iterations", "Each outer CV iteration"))) %>%
  mutate(
    option = factor(
      option,
      levels = c(1:4),
      labels = str_c("Aggregation option ", 1:4)
    ),
    method = factor(
      method,
      levels = c("Observational", "Interventional"),
      labels = c("Observational SHAP", "Interventional SHAP")
    )
  )

p1 <- ggplot(data_plot,
       aes(
         time_step,
         shap,
         group = outer_i,
         color = case,
         linetype = case
       )) +
  geom_line(size = 0.25) +
  scale_linetype_manual(values = c("dashed", "solid")) +
  scale_color_manual(values = c("grey40", "red")) +
  facet_grid(option~method, scales = "free_y") +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 1, 5, 10, 50, 100, 500, 1000),
    labels = c(0, 1, " ", 10, " ", 100, " ", 1000)
  ) +
  labs(x = "Time step in the past",
       y = "Average importance of rainfall for discharge prediction [L/s]") +
  theme_bw(base_size = 8) +
  theme(legend.position = "top",
        legend.title = element_blank(),
        legend.key.height = unit(0.2, "line"))

p1


# p2 ----------------------------------------------------------------------

data_plot_interventional <- int_contr %>%
  bind_rows() %>%
  dplyr::mutate(method = "Interventional") %>%
  rename(shap = int_contr)

data_plot_observational <- ob_contr %>%
  bind_rows() %>%
  dplyr::mutate(method = "Observational") %>%
  rename(shap = ob_contr)

data_plot <- data_plot_interventional %>%
  bind_rows(data_plot_observational)

data_plot_mean <- data_plot %>%
  group_by(time_step, option, method) %>%
  dplyr::summarise(shap = mean(shap)) %>%
  mutate(outer_i = 0) # for binding rows

data_plot <- data_plot_mean %>%
  bind_rows(data_plot) %>%
  mutate(case = map_chr(outer_i, function(x)
    ifelse(x == 0, "Mean of all outer CV iterations", "Each outer CV iteration"))) %>%
  mutate(
    option = factor(
      option,
      levels = c(1:4),
      labels = str_c("Aggregation option ", 1:4)
    ),
    method = factor(
      method,
      levels = c("Observational", "Interventional"),
      labels = c("Observational SHAP", "Interventional SHAP")
    )
  )

p2 <- ggplot(data_plot,
             aes(
               time_step,
               shap,
               group = outer_i,
               color = case,
               linetype = case
             )) +
  geom_line(size = 0.25) +
  scale_linetype_manual(values = c("dashed", "solid")) +
  scale_color_manual(values = c("grey40", "red")) +
  facet_grid(option~method, scales = "free_y") +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 1, 5, 10, 50, 100, 500, 1000),
    labels = c(0, 1, " ", 10, " ", 100, " ", 1000)
  ) +
  labs(x = "Time step in the past",
       y = "Average contribution of rainfall to discharge prediction [L/s]") +
  theme_bw(base_size = 8) +
  theme(legend.position = "top",
        legend.title = element_blank(),
        legend.key.height = unit(0.2, "line"))

p2


# Output ------------------------------------------------------------------

cowplot::plot_grid(p1,p2,align = "h", ncol = 2,labels = c('(a)', '(b)'), label_fontface = "plain", label_size = 10)

ggsave(filename = "./data/WS/plot/mean_shap_observational.png", width = 7, height = 4.6, units = "in", dpi = 600)
ggsave(filename = "./data/WS/plot/figure5.pdf", width = 7, height = 4.6, units = "in")

