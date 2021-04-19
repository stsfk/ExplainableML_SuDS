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
  dataRetrieval
)


# data --------------------------------------------------------------------

# The documentation of the dataset can be found in https://doi.org/10.23719/1503369

dir_path <- c("./data/StFrancis/")

if (!dir.exists(dir_path)){
  dir.create(dir_path)
}

destfile <- c("./data/StFrancis/St.Francis_All_Data.xlsx")
if (!file.exists(destfile)){
  url <- "https://pasteur.epa.gov/uploads/10.23719/1503369/St.Francis_All_Data.xlsx"
  download.file(url, destfile, mode="wb")
}

