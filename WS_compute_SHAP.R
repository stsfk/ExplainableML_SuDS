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

sapply(1:5, function(iter) analysis(cv_folds$splits[[iter]])$record_id %>% unlist() %>% length())


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



