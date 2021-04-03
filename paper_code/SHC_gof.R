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


# XGB modeling ------------------------------------------------------------

load("./data/SHC/xgb_fits.Rda")

extract_prediction <- function(optObj){
  # function to return the optimal prediction
  
  optObj$scoreSummary %>%
    arrange(desc(Score)) %>%
    dplyr::slice(1) %>%
    pull(test_pred) %>%
    .[[1]]
}

preds <- optObjs %>%
  lapply(extract_prediction)

for (i in seq_along(preds)){
  pred <- preds[[i]]
  
  hydroGOF::gof(pred$pred, pred$ob)
}

















