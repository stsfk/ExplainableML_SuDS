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
  dataRetrieval,
  readxl
)


# data --------------------------------------------------------------------

# The documentation of the dataset can be found in https://doi.org/10.23719/1503369
dir_path <- c("./data/")

if (!dir.exists(dir_path)){
  dir.create(dir_path)
}

dir_path <- c("./data/StFrancis/")

if (!dir.exists(dir_path)){
  dir.create(dir_path)
}

destfile <- c("./data/StFrancis/St.Francis_All_Data.xlsx")
if (!file.exists(destfile)){
  url <- "https://pasteur.epa.gov/uploads/10.23719/1503369/St.Francis_All_Data.xlsx"
  download.file(url, destfile, mode="wb")
}


# Preprocess --------------------------------------------------------------

# read flow data
data_raw <- readxl::read_xlsx("./data/StFrancis/St.Francis_All_Data.xlsx", sheet = "Flows_2011-2014", skip = 1) %>%
  dplyr::select(-starts_with("..."), -starts_with("time"), -starts_with("volume"))

upper_inflow <- data_raw %>%
  dplyr::select(datetime = `DateTime EST...1`,
                flow = UpperQ) %>%
  .[complete.cases(.),]

lower_inflow <- data_raw %>%
  dplyr::select(datetime = `DateTime EST...7`,
                flow = `Lower Q`) %>%
  .[complete.cases(.),]

lower_outflow <- data_raw %>%
  dplyr::select(datetime = `DateTime EST...13`,
                flow = outcfs) %>%
  .[complete.cases(.),]

# read rainfall data
data_rain <- readxl::read_xlsx("./data/StFrancis/St.Francis_All_Data.xlsx", sheet = "Weather2011-2014", skip = 0) %>%
  dplyr::select(datetime = "DateTime EST...16",
                rain = "Precip (inches)")








