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

# from https://rpubs.com/Koundy/71792
theme_Publication <- function(base_size=10, base_family="") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold",
                                      size = rel(1.2), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(1)),
            axis.title.y = element_text(angle=90,vjust =2),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(), 
            axis.line = element_line(colour="black"),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour="#f0f0f0"),
            panel.grid.minor = element_blank(),
            legend.key = element_rect(colour = NA),
            legend.position = "bottom",
            legend.direction = "horizontal",
            legend.key.size= unit(0.2, "cm"),
            legend.margin = unit(0, "cm"),
            legend.title = element_text(face="italic"),
            plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
            strip.text = element_text(face="bold")
    ))
  
}

# Data --------------------------------------------------------------------


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

shap_observational <- function(option, id){
  
  dtrain <- read_csv(
    paste0(
      "./data/SHC/model_fits/xgb_opt_",
      option,
      "/train_",
      id,
      ".csv"
    ),
    col_types = cols(.default = col_double())
  )
  
  dtest <- read_csv(
    paste0(
      "./data/SHC/model_fits/xgb_opt_",
      option,
      "/test_",
      id,
      ".csv"
    ),
    col_types = cols(.default = col_double())
  )
  
  model <- xgboost::xgb.load(
    paste0(
      "./data/SHC/model_fits/xgb_opt_",
      option,
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
  
  out <- predict(model, new_data, predcontrib = T) %>%
    as_tibble()
  
  # adding record id index
  out %>%
    mutate(record_id = c(train_index, val_index, test_index)) %>%
    select(record_id, everything())
}

shap_interventional <- function(option, id){
  # read dtrain to get the name of the features
  dtrain <- read_csv(
    paste0(
      "./data/SHC/model_fits/xgb_opt_",
      option,
      "/train_",
      id,
      ".csv"
    ),
    col_types = cols(.default = col_double())
  )
  
  shap_matrix_train <- read_csv(
    paste0(
      "./data/SHC/model_fits/xgb_opt_",
      option,
      "/shap_train_",
      id,
      ".csv"
    ),
    col_types = cols(.default = col_double()),
    col_names = names(dtrain) 
  )
  
  shap_matrix_test <- read_csv(
    paste0(
      "./data/SHC/model_fits/xgb_opt_",
      option,
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
  shap_matrix %>%
    mutate(record_id = c(train_index, val_index, test_index)) %>%
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

optimal_candidate_id <- read_csv("./data/SHC/model_fits/optimal_candidate_id.csv",
                                 col_types = cols("i", "i"))

shap_matrixs <- optimal_candidate_id %>%
  mutate(shap_ob = vector("list", 1),
         shap_int = vector("list", 1))

for (i in 1:nrow(optimal_candidate_id)){
  
  c(option, id) %<-% optimal_candidate_id[i, ]
  
  shap_matrixs$shap_ob[[i]] <- shap_observational(option, id)
  shap_matrixs$shap_int[[i]] <- shap_interventional(option, id)
  
}

save(shap_matrixs, file = "./data/SHC/shap_matrixs.Rda")



# Prepare plot ------------------------------------------------------------

load("./data/SHC/shap_matrixs.Rda")

shap_distributed <- shap_matrixs %>%
  dplyr::select(option, id) %>%
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
    dplyr::select(option, !!col) %>%
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
  bind_rows(data_plot_observational) %>%
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
               shap
             )) +
  geom_line(size = 0.25) +
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
  bind_rows(data_plot_observational) %>%
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
               shap
             )) +
  geom_line(size = 0.25) +
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

ggsave(filename = "./data/SHC/plot/mean_shap_observational.png", width = 7, height = 4.6, units = "in", dpi = 600)
ggsave(filename = "./data/SHC/plot/figure6.pdf", width = 7, height = 4.6, units = "in")

