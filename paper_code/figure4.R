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
  lemon
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

scale_fill_Publication <- function(...){
  library(scales)
  discrete_scale("fill","Publication",manual_pal(values = c("#386cb0","#fdb462","#7fc97f","#ef3b2c","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
  
}

scale_colour_Publication <- function(...){
  library(scales)
  discrete_scale("colour","Publication",manual_pal(values = c("#386cb0","#fdb462","#7fc97f","#ef3b2c","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
  
}

# WS XGBoost --------------------------------------------------------------

derive_gof <- function(option, outer_i){
  load(paste0("./data/WS/model_fits/xgb_opt_", option, "_iter_", outer_i, ".Rda"))
  
  pred_df <- optObj$scoreSummary %>% 
    as_tibble() %>%
    dplyr::slice(which.max(Score)) %>% # the optimal hyperpara
    dplyr::select(val_rmse = Score, test_pred, Epoch) %>%
    pull(test_pred) %>%
    .[[1]]
  
  out <- hydroGOF::gof(sim = pred_df$pred, obs = pred_df$ob, digits=10) %>%
    t() %>%
    as_tibble() %>%
    dplyr::select(RMSE, NSE, R2) %>%
    mutate(
      option = as.character(option),
      outer_i  = outer_i,
      model = "XGBoost",
      site = "WS") 
  
  out
}

eval_grid <- expand.grid(
  option = 1:4,
  outer_i = 1:5
) %>%
  as_tibble()

outs <- vector("list", nrow(eval_grid))
for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]
  outer_i <- eval_grid$outer_i[i]
  
  outs[[i]] <- derive_gof(option, outer_i)
}


WS_XGB <- outs %>%
  bind_rows()



# WS LM -------------------------------------------------------------------

derive_gof <- function(option, outer_i){
  load(paste0("./data/WS/lm_fits/lm_opt_", option, "_iter_", outer_i, ".Rda"))
  
  pred_df <- optObj$scoreSummary %>% 
    as_tibble() %>%
    dplyr::slice(which.max(Score)) %>% # the optimal hyperpara
    dplyr::select(val_rmse = Score, test_pred, Epoch) %>%
    pull(test_pred) %>%
    .[[1]]
  
  out <- hydroGOF::gof(sim = pred_df$pred, obs = pred_df$ob, digits=10) %>%
    t() %>%
    as_tibble() %>%
    dplyr::select(RMSE, NSE, R2) %>%
    mutate(
      option = as.character(option),
      outer_i  = outer_i,
      model = "LM",
      site = "WS") 
  
  out
}

eval_grid <- expand.grid(
  option = 1:4,
  outer_i = 1:5
) %>%
  as_tibble()

outs <- vector("list", nrow(eval_grid))
for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]
  outer_i <- eval_grid$outer_i[i]
  
  outs[[i]] <- derive_gof(option, outer_i)
}


WS_LM <- outs %>%
  bind_rows()



# SHC XGBoost -------------------------------------------------------------

derive_gof <- function(option){
  load(paste0("./data/SHC/model_fits/xgb_opt_", option, ".Rda"))
  
  pred_df <- optObj$scoreSummary %>% 
    as_tibble() %>%
    dplyr::slice(which.max(Score)) %>% # the optimal hyperpara
    dplyr::select(val_rmse = Score, test_pred, Epoch) %>%
    pull(test_pred) %>%
    .[[1]]
  
  out <- hydroGOF::gof(sim = pred_df$pred, obs = pred_df$ob, digits=10) %>%
    t() %>%
    as_tibble() %>%
    dplyr::select(RMSE, NSE, R2) %>%
    mutate(
      option = as.character(option),
      outer_i  = 1,
      model = "XGBoost",
      site = "SHC") 
  
  out
}

eval_grid <- expand.grid(
  option = 1:4
) %>%
  as_tibble()

outs <- vector("list", nrow(eval_grid))
for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]

  outs[[i]] <- derive_gof(option)
}

SHC_XGB <- outs %>%
  bind_rows()


# SHC LM ------------------------------------------------------------------

derive_gof <- function(option){
  load(paste0("./data/SHC/lm_fits/lm_", option, ".Rda"))
  
  pred_df <- optObj$scoreSummary %>% 
    as_tibble() %>%
    dplyr::slice(which.max(Score)) %>% # the optimal hyperpara
    dplyr::select(val_rmse = Score, test_pred, Epoch) %>%
    pull(test_pred) %>%
    .[[1]]
  
  out <- hydroGOF::gof(sim = pred_df$pred, obs = pred_df$ob, digits=10) %>%
    t() %>%
    as_tibble() %>%
    dplyr::select(RMSE, NSE, R2) %>%
    mutate(
      option = as.character(option),
      outer_i  = 1,
      model = "LM",
      site = "SHC") 
  
  out
}

eval_grid <- expand.grid(
  option = 1:4
) %>%
  as_tibble()

outs <- vector("list", nrow(eval_grid))
for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]
  
  outs[[i]] <- derive_gof(option)
}

SHC_LM <- outs %>%
  bind_rows()

# SHC SWMM ----------------------------------------------------------------

data_swmm <- read.table("./data/SHC/SWMM/outflow.txt", skip = 8) %>%
  as_tibble() %>%
  transmute(datetime = ymd_hms(paste(V2, V3, V4, V5, V6, V7)),
            pred = V8/35.314666212661) %>%
  arrange(datetime)

# load observed flow
load(paste0("./data/SHC/lm_fits/lm_", option, ".Rda"))
pred_df <- optObj$scoreSummary %>% 
  as_tibble() %>%
  dplyr::slice(which.max(Score)) %>% # the optimal hyperpara
  dplyr::select(val_rmse = Score, test_pred, Epoch) %>%
  pull(test_pred) %>%
  .[[1]] %>%
  dplyr::select(-pred) %>%
  left_join(data_swmm, by = "datetime")

SHC_SWMM <- hydroGOF::gof(sim = pred_df$pred, obs = pred_df$ob, digits=10) %>%
  t() %>%
  as_tibble() %>%
  dplyr::select(RMSE, NSE, R2) %>%
  mutate(
    option = "None",
    outer_i  = 1,
    model = "SWMM",
    site = "SHC") 




# Plot --------------------------------------------------------------------

data_plot <- list(
  WS_XGB,
  WS_LM,
  SHC_XGB,
  SHC_LM,
  SHC_SWMM
) %>%
  bind_rows() %>%
  gather(metrics, value, RMSE:R2) %>%
  mutate(option = factor(option),
         model = factor(model, levels = c("XGBoost", "LM", "SWMM")),
         site = factor(site, levels = c("WS", "SHC")),
         metrics = factor(metrics, levels = c("RMSE", "NSE", "R2"), labels = c("RMSE", "NSE", "R²")))


ggplot(data_plot, aes(model, value, color = option, shape = option))+
  geom_point(position = position_dodge(0.6))+
  scale_y_continuous(limits = c(-0.12,1), breaks = c(0,0.2,0.4,0.6,0.8,1)) +
  facet_grid(site~metrics, scales = "free_y")+
  labs(color = "Aggregation option", shape = "Aggregation option") +
  coord_capped_cart(bottom = brackets_horisontal(direction = "up", length = unit(0.12, "npc")))+
  scale_colour_Publication()+
  theme_Publication(base_size = 10)+
  theme(legend.position = "top", 
        panel.grid.major.x = element_blank(),
        panel.background = element_rect(fill = "grey97"),
        panel.spacing = unit(1, "lines"))

ggsave(filename = "./data/SHC/plot/metrics.png", width = 7, height = 5, units = "in", dpi = 600)
ggsave(filename = "./data/SHC/plot/figure4.pdf", width = 7, height = 5, units = "in")
 
