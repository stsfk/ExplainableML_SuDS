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

load("./data/WS/shap_matrixs.Rda")

trainable_record_id <- trainable_df$record_id %>%
  unlist()

optimal_candidate_id <- read_csv("./data/WS/model_fits/optimal_candidate_id.csv",
                                 col_types = cols("i", "i", "i"))


load("./data/WS/model_fits/xgb_opt_1_iter_1.Rda")

optimal_para <- optObj$scoreSummary[optObj$scoreSummary$Score %>% which.max(),]

BIAS <- shap_matrixs$shap_ob[[1]]$BIAS[1]

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


decomposition_feautre <- function(event_id_chosen, model_i = 1){
  
  # BIAS from global
  
  # shap_distributed from global
  distributed_impor_m <- shap_distributed$shap_ob_dist[[model_i]] %>%
    dplyr::select(V1:V1441) %>%
    data.matrix()
  
  record_id_lookup <- shap_distributed$shap_ob_dist[[model_i]]$record_id
  
  row_index <- which(record_id_lookup == event_id_chosen)
  contribute_series <- distributed_impor_m[row_index, ]
  
  s_e <- tibble(
    s = (0:6)*6 + 1,
    e = c(1:6*6, 1441)
  )
  
  out <- rep(0, nrow(s_e) + 1)
  out[1] <- BIAS
  
  for (j in 1:nrow(s_e)){
    out[j + 1] <- contribute_series[s_e$s[j]:s_e$e[j]] %>% sum()
  }
  
  out
}

create_decomposition_feautre_df <- function(event_ids){
  
  col_names <- c("Bias", "1h", "2h", "3h", "4h", "5h", "6h", ">6h")
  
  decomposition_df <- sapply(event_ids, decomposition_feautre) %>%
    t() %>%
    as_tibble() %>%
    set_names(col_names)
  
  out <- tibble(
    time_step = seq_along(event_ids),
    rain = data_process$X[event_ids],
  ) %>%
    bind_cols(decomposition_df)
  
  out
}

# process
event_id_large <- trainable_df[trainable_df$peak_flow %>% which.max(),] %>%
  dplyr::pull(record_id) %>%
  unlist()

event_id_mid <- trainable_df[17,] %>%
  dplyr::pull(record_id) %>%
  unlist()

event_id_small <- trainable_df[29,] %>%
  dplyr::pull(record_id) %>%
  unlist()


rain_l <- tibble(
  time_step = 1:100,
  rain = data_process$X[event_id_large[1:100]],
  case = "Large runoff event"
)

rain_m <- tibble(
  time_step = 1:300,
  rain = data_process$X[event_id_mid[1:300]],
  case = "Medium runoff event"
)

rain_s <- tibble(
  time_step = 1:250,
  rain = data_process$X[event_id_small[1:250]],
  case = "Small runoff event"
)

rain_df <- rain_l %>%
  bind_rows(rain_m) %>%
  bind_rows(rain_s)

p1 <- ggplot(rain_df)+
  geom_line(aes(time_step/6, rain), size = 0.4) +
  facet_wrap(~case, scales = "free") +
  labs(y = "Rainfall intensity\n[mm/h]") +
  theme_ipsum(
    axis_title_just = "m",
    axis_title_size = 9,
    base_size = 8,
    strip_text_size = 8,
    plot_margin = margin(10, 10, 10, 10),
    base_family  = "sans"
  ) +
  theme(legend.position = "right", 
        legend.key.size = unit(0.3, "cm"),
        legend.key.width = unit(0.4, "cm"),
        legend.text=element_text(size=rel(1.1), face = "italic"),
        panel.spacing = unit(1, "lines"),
        axis.title.x = element_blank())



# Plot b

data_plot_l <- create_decomposition_feautre_df(event_id_large)%>%
  dplyr::filter(time_step <=100) %>%
  mutate(case = "Large runoff event")

data_plot_m <- create_decomposition_feautre_df(event_id_mid)%>%
  dplyr::filter(time_step <=300) %>%
  mutate(case = "Medium runoff event")

data_plot_s <- create_decomposition_feautre_df(event_id_small)%>%
  dplyr::filter(time_step <=250) %>%
  mutate(case = "Small runoff event")


data_plot <- data_plot_l %>%
  bind_rows(data_plot_m) %>%
  bind_rows(data_plot_s) %>%
  gather(item, value, Bias:`>6h`) %>%
  mutate(item = factor(item,
                       levels = c(
                         ">6h", "6h", "5h", "4h", "3h", "2h", "1h" , "Bias"
                       ),
                       labels = c(
                         ">6 h", "6 h", "5 h", "4 h", "3 h", "2 h", "1 h" , "Bias"
                       )))

data_plot$positive <- ifelse(data_plot$value >= 0, data_plot$value, 0)
data_plot$negative <- ifelse(data_plot$value < 0, data_plot$value, -1e-36)


p2 <- ggplot(data_plot) + 
  geom_area(aes(time_step/6, positive, fill = item), alpha=0.6 , size=0.1, colour="white")+
  geom_area(aes(time_step/6, negative, fill = item), alpha=0.6 , size=0.1, colour="white")+
  scale_fill_viridis(discrete = T,   
                     guide = guide_legend(
                       direction = "horizontal",
                       title.position = "top",
                       label.position = "right",
                       label.hjust = 0,
                       label.vjust = 1,
                       nrow = 1
                     ),
                     option = "D") +
  labs(x = "Time since the beginning of runoff event [h]",
       y = "Rainfall's contribution to\ndischarge prediction [L/s]",
       fill = "Rainfall recorded in specific period in the past and other factors") +
  facet_wrap(~case, scales = "free") +
  theme_ipsum(
    axis_title_just = "m",
    axis_title_size = 9,
    base_size = 8,
    strip_text_size = 8,
    plot_margin = margin(10, 10, 10, 10),
    base_family  = "sans"
  ) +
  theme(legend.position = "bottom", 
        legend.key.size = unit(0.3, "cm"),
        legend.key.width = unit(0.4, "cm"),
        legend.text=element_text(size=rel(1.1), face = "italic"),
        panel.spacing = unit(1, "lines"),
        strip.background = element_blank(),
        strip.text.x = element_blank())



# Output ------------------------------------------------------------------


cowplot::plot_grid(p1,p2, ncol = 1,labels = c('(a)', '(b)'), label_fontface = "plain", label_size = 10,
                   rel_heights = c(0.45,1))


ggsave(filename = "./data/WS/plot/figure9.png", width = 7, height = 4.5, units = "in", dpi = 600)
ggsave(filename = "./data/WS/plot/figure9.pdf", width = 7, height = 4.5, units = "in")




