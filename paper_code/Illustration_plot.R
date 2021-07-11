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
  ParBayesianOptimization,
  hydroGOF,
  xgboost,
  svglite
)




