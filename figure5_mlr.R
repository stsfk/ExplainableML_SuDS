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


# WS XGB ------------------------------------------------------------------

eval_grid <- tibble(
  iter = 1:5,
  fpath = paste0("./data/WS/model_fits/gof_iter=", iter, "opt=1.Rda")
)

outs <- vector("list", nrow(eval_grid))

for (i in 1:nrow(eval_grid)){
  load(eval_grid$fpath[i])
  df <- out$test_pred[[1]]
  
  ob <- df$ob
  pred <- df$pred
  
  outs[[i]] <- hydroGOF::gof(sim = pred, obs = ob, digits=10) %>%
    t() %>%
    as_tibble() %>%
    dplyr::select(RMSE, NSE, R2)
}

WS_XGB <- outs %>%
  bind_rows() %>%
  mutate(iter = 1:5,
         model = "XGBoost",
         site = "WS")


# WS LM -------------------------------------------------------------------


eval_grid <- tibble(
  iter = 1:5,
  fpath = paste0("./data/WS/lm_fits/gof_iter=", iter, "opt=1.Rda")
)

outs <- vector("list", nrow(eval_grid))

for (i in 1:nrow(eval_grid)){
  load(eval_grid$fpath[i])
  df <- out$test_pred[[1]]
  
  ob <- df$ob
  pred <- df$pred
  
  outs[[i]] <- hydroGOF::gof(sim = pred, obs = ob, digits=10) %>%
    t() %>%
    as_tibble() %>%
    dplyr::select(RMSE, NSE, R2)
}

WS_LM <- outs %>%
  bind_rows() %>%
  mutate(iter = 1:5,
         model = "LM",
         site = "WS")


# SHC XGB ------------------------------------------------------------------

load("./data/SHC/model_fits/gof_opt=1.Rda")
df <- out$test_pred[[1]]

ob <- df$ob
pred <- df$pred

SHC_XGB <- hydroGOF::gof(sim = pred, obs = ob, digits=10) %>%
  t() %>%
  as_tibble() %>%
  dplyr::select(RMSE, NSE, R2) %>%
  mutate(iter  = 1,
         model = "XGBoost",
         site = "SHC")

# SHC LM ------------------------------------------------------------------

load("./data/SHC/lm_fits/gof_opt=1.Rda")
df <- out$test_pred[[1]]

ob <- df$ob
pred <- df$pred

SHC_LM <- hydroGOF::gof(sim = pred, obs = ob, digits=10) %>%
  t() %>%
  as_tibble() %>%
  dplyr::select(RMSE, NSE, R2) %>%
  mutate(iter  = 1,
         model = "LM",
         site = "SHC")

# SHC SWMM ----------------------------------------------------------------

data_swmm <- read.table("./data/SHC/SWMM/outflow.txt", skip = 8) %>%
  as_tibble() %>%
  transmute(datetime = ymd_hms(paste(V2, V3, V4, V5, V6, V7)),
            pred = V8/35.314666212661) %>%
  arrange(datetime)

# load observed flow
load("./data/SHC/lm_fits/gof_opt=1.Rda")

pred_df <- out$test_pred[[1]]

pred_df <- pred_df %>%
  dplyr::select(-pred) %>%
  left_join(data_swmm, by = "datetime") %>%
  arrange(datetime)
  

SHC_SWMM <- hydroGOF::gof(sim = pred_df$pred, obs = pred_df$ob, digits=10) %>%
  t() %>%
  as_tibble() %>%
  dplyr::select(RMSE, NSE, R2) %>%
  mutate(
    iter  = 1,
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
  mutate(model = factor(model, levels = c("XGBoost", "LM", "SWMM")),
         site = factor(site, levels = c("WS", "SHC")),
         metrics = factor(metrics, levels = c("RMSE", "NSE", "R2"), labels = c("RMSE", "NSE", "R²")))


ggplot(data_plot, aes(model, value))+
  geom_point(aes(color = model, shape = model), position = position_dodge(0.6))+
  geom_line(aes(group = iter), color = "black", size = 0.3)+
  scale_y_continuous(limits = c(-0.12,1), breaks = c(0,0.2,0.4,0.6,0.8,1)) +
  facet_grid(site~metrics, scales = "free_y")+
  labs(color = "Model", shape = "Model") +
  coord_capped_cart(bottom = brackets_horisontal(direction = "up", length = unit(0.12, "npc")))+
  scale_colour_Publication()+
  theme_Publication(base_size = 10)+
  theme(legend.position = "top", 
        panel.grid.major.x = element_blank(),
        panel.background = element_rect(fill = "grey97"),
        panel.spacing = unit(1, "lines"))

ggsave(filename = "./data/figures/figure5.png", width = 7, height = 5, units = "in", dpi = 600)
ggsave(filename = "./data/figures/figure5.pdf", width = 7, height = 5, units = "in")
 
