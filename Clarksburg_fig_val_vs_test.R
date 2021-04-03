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
  cowplot
)


# Function ----------------------------------------------------------------

derive_gof <- function(option, outer_i){
  load(paste0("./data/Clarksburg/model_fits/xgb_opt_", option, "_iter_", outer_i, ".Rda"))
  
  out <- optObj$scoreSummary %>% 
    dplyr::select(val_rmse = Score, test_pred, Epoch) %>%
    as_tibble() %>%
    mutate(
      val_rmse = -val_rmse,
      test_rmse = map_dbl(test_pred, function(x) hydroGOF::rmse(sim = x$pred, obs = x$ob)),
      option = option,
      outer_i  = outer_i) %>%
    dplyr::select(option, outer_i, val_rmse, test_rmse, test_pred, Epoch)
  
  out
}


# Data process ------------------------------------------------------------


eval_grid <- expand.grid(
  option = 1:1,
  outer_i = 1:5
) %>%
  as_tibble()

outs <- vector("list", nrow(eval_grid))
for (i in 1:nrow(eval_grid)){
  option <- eval_grid$option[i]
  outer_i <- eval_grid$outer_i[i]
  
  outs[[i]] <- derive_gof(option, outer_i)
}


# Plot --------------------------------------------------------------------


data_plot <- outs %>%
  bind_rows() %>%
  mutate(
    outer_i = factor(
      outer_i,
      levels = c(1:5),
      labels = str_c("Outer CV iteration ", 1:5)
    ),
    option = factor(
      option,
      levels = c(1:4),
      labels = str_c("Aggregation option ", 1:4)
    )
  )


ggplot(data_plot, aes(val_rmse, test_rmse)) +
  geom_point(shape = 1, color = "steelblue", size = 1.2, stroke = 0.4) +
  geom_smooth(method = "lm", se = 0, color = "black", size = 0.5)+
  labs(x = "Inner CV error (RMSE) [L/s]",
       y = "Outer CV error (RMSE) [L/s]") +
  facet_grid(option~outer_i, scales = "free") +
  theme_bw(base_size = 8)

ggsave(filename = "./data/WS/plot/inner_vs_outer.png", width = 7, height = 5, units = "in", dpi = 600)

