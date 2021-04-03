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

dir_path <- paste0("./data/Clarksburg/")

if (!dir.exists(dir_path)){
  dir.create(dir_path)
}

if (!file.exists("./data/Clarksburg/rainfall_runoff.Rda")){
  forest <- readNWISuv(
    "01643395",parameterCd="00060",
    startDate = "",
    endDate = "",
    tz = "UTC"
  )
  
  control <- readNWISuv(
    "01644375",parameterCd="00060",
    startDate = "",
    endDate = "",
    tz = "UTC"
  )
  
  treatment1 <- readNWISuv(
    "01644371",parameterCd="00060",
    startDate = "",
    endDate = "",
    tz = "UTC"
  )
  
  treatment2 <- readNWISuv(
    "01644372",parameterCd="00060",
    startDate = "",
    endDate = "",
    tz = "UTC"
  )
  
  rain <- readNWISuv(
    "391328077185901",parameterCd="00045",
    startDate = "",
    endDate = "",
    tz = "UTC"
  )
  
  rain_alternative <- readNWISuv(
    "391407077174001",parameterCd="00045",
    startDate = "",
    endDate = "",
    tz = "UTC"
  )
  
  save(forest, control, treatment1, treatment2, rain, rain_alternative,
       file = "./data/Clarksburg/rainfall_runoff.Rda")
}

load("./data/Clarksburg/rainfall_runoff.Rda")

# Pre-process -------------------------------------------------------------

# runoff data, rename and filter
process_runoff <-
  function(flow_raw,
           name = "forest",
           year_of_choice = c(2018,2019)) {
    flow_raw %>%
      as_tibble() %>%
      transmute(datetime = ymd_hms(dateTime),
                flow = X_00060_00000) %>%
      filter(year(datetime) %in% year_of_choice) %>%
      mutate(case = name) %>%
      arrange(datetime)
  }

input1 <- list(forest, control, treatment1, treatment2)
input2 <- list("forest", "control", "treatment1", "treatment2")

runoffs <- map2(input1, input2, process_runoff)

# rainfall data, rename and filter

process_rain <-
  function(rain_raw,
           name = "rain",
           year_of_choice = c(2018,2019)) {
    rain_raw %>%
      as_tibble() %>%
      transmute(datetime = ymd_hms(dateTime),
                rain = X_00045_00000) %>%
      filter(year(datetime) %in% year_of_choice) %>%
      mutate(case = name) %>%
      arrange(datetime)
  }

rain <- process_rain(rain, "rain")
rain_alternative <- process_rain(rain_alternative, "rain")

# check intervals in measurement
check_interval <- function(data){
  data$datetime %>% 
    diff() %>% 
    table()
}

rain %>% check_interval() # a few missing data
lapply(runoffs, check_interval) # a few missing data

# make regular sequence, fill gaps with NA
regular_seq <- function(df,interval_of_choice){
  
  s_ind <- df$datetime[1]
  e_ind <- tail(df$datetime, 1)
  
  tibble(datetime = seq(from = s_ind, to = e_ind, by = interval_of_choice)) %>%
    left_join(df, by = "datetime") %>%
    arrange(datetime)
}

rain <- regular_seq(rain, 900)
rain_alternative <- regular_seq(rain_alternative, 900)
runoffs <- lapply(runoffs, regular_seq, 300)

# join rainfall and runoffs

forest <- rain %>%
  left_join(runoffs[[1]], by = "datetime")

control <- rain %>%
  left_join(runoffs[[2]], by = "datetime")

treatment1 <- rain %>%
  left_join(runoffs[[3]], by = "datetime")

treatment2 <- rain %>%
  left_join(runoffs[[4]], by = "datetime")


# Functions ---------------------------------------------------------------

wet_dry <- function(datetime, x, IETD = 24, before_n = 0){
  # This function returns wet and dry index, using default 24 hour thd
  # Turn the irregular time series to regurlar one, this can aviod search for the duration
  # Vars: 
  #   datetime: time index; x: observation series; IETD: inter-event time definition; 
  #   before_n: before_n
  
  df <- tibble(datetime = datetime,
               x = x)
  
  full_df <- tibble(datetime = seq(from = datetime[[1]], to = tail(datetime, 1), by = "1 min")) %>%
    full_join(df, by = "datetime") %>%
    mutate(wet = F) %>%
    arrange(datetime)
  
  rain_ind <- which(full_df$x != 0) # index where rain occurs
  max_le <- length(full_df$x)
  for (i in seq_along(rain_ind)){
    ind_sta <- max(1, rain_ind[i] - before_n) # events before_n step ahead rain is wet
    ind_end <- min(max_le, (rain_ind[i] + IETD * 60)) # IETD convert to minutes
    full_df$wet[ind_sta:ind_end] <- T 
  }
  
  df <- df %>%
    select(datetime) %>%
    left_join(full_df, by = "datetime") %>%
    arrange(datetime)
  
  # return wet/dry vector
  df$wet
}

mark_event <- function(x){
  # This is a function mark the number of rainfall event based on wet_dry output
  # Vars: x is the wet vector
  
  out <- rep(0, length(x))
  
  temp <- rle(x)
  values <- temp$values
  lengths <- temp$lengths
  
  loc <- 1
  event <- 1
  for (i in seq_along(values)){
    step <- lengths[[i]]
    if (values[[i]]){
      out[loc:(loc + step - 1)] <- event
      event <- event + 1
    }
    loc <- loc + step
  }
  
  # dry period is marked as 0
  out
}

# Process data for training -----------------------------------------------

data_raw <- treatment2

data_process <- data_raw %>% 
  mutate(year = year(datetime)) %>%
  group_by(year) %>%
  mutate(wet = wet_dry(datetime, rain, IETD = 24)) %>%
  mutate(event_id = mark_event(wet)) %>%
  ungroup() %>%
  mutate(event_id = paste(year, event_id, sep = "_")) %>%
  mutate(record_id = 1:n()) %>%
  rename(Y = flow,
         X = rain) %>%
  mutate(X = replace(X, is.na(X), 0)) # fill missing rainfalls with 0

# Mark_trainable event
# Available events for ML:
#   when (1) April <= time <= Nov, 
#        (2) wet
#        (3) rainfall not missing

# get index for data_process available for training 
trainable_df <- data_process %>%
  mutate(month = month(datetime),
         trainable = (month > 3) & (month < 11) & wet & !is.na(X) & !is.na(Y)) %>%
  dplyr::filter(trainable) %>% 
  group_by(event_id) %>%
  summarise(datetime = list(datetime),
            record_id = list(record_id),
            peak_flow = max(Y))

# save
save(data_process, trainable_df, file = "./data/Clarksburg/trainable_index.Rda")
