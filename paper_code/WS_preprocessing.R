if (!require("pacman")) {
  install.packages("pacman")
}

pacman::p_load(tidyverse,
               lubridate,
               RcppRoll,
               caret)

org_wd <- getwd()


# Read and join data ------------------------------------------------------

# Raw
rain_raw <- read_csv("./data/WS/raw/Geauga_rain.csv",
                     comment = "#",
                     guess_max = 50000) # rainfall
overflow_raw <- read_csv("./data/WS/raw/Geauga_overflow.csv",
                         comment = "#",
                         guess_max = 50000) # overflow from rain garden
underdrain_raw <- read_csv("./data/WS/raw/Geauga_underdrain.csv",
                           comment = "#",
                           guess_max = 50000) # underdrain flow from pp

# Keep key columns
rain <- rain_raw %>%
  select(`Timestamp (UTC-05:00)`, Value)
names(rain) <- c("datetime", "rain")
rain <- rain %>%
  mutate(rain = rain * 25.4 * 6,
         datetime = force_tz(datetime, "UTC")) # conversion, inch/10 min to mm/hour

overflow <- overflow_raw %>%
  select(`Timestamp (UTC-05:00)`, Value)
names(overflow) <- c("datetime", "overflow")
overflow <- overflow %>%
  mutate(overflow = overflow * 28.3168466,
         datetime = force_tz(datetime, "UTC")) # conversion, CFS to LPS

underdrain <- underdrain_raw %>%
  select(`Timestamp (UTC-05:00)`, Value)
names(underdrain) <- c("datetime", "underdrain")
underdrain <- underdrain %>%
  mutate(underdrain = underdrain * 28.3168466,
         datetime = force_tz(datetime, "UTC")) # conversion, CFS to LPS

# Reference time series
ref <- vector("list", 5) # create a list to store data from 2010 to 2013 between Mar. and Oct.
for (i in 1:5) {
  start_date <- paste0(i + 2008, "-03-01 00:00:00") %>%
    parse_datetime
  end_date <- paste0(i + 2008, "-10-31 23:50:00") %>%
    parse_datetime
  ref[[i]] <- seq(start_date, end_date, by = "10 min")
}
ref <- tibble(datetime = do.call(c, ref))

# function to write formated data
write_data <- function(data, file_name, time_format = "%m/%d/%Y %H:%M") {
  data <- data %>%
    mutate(datetime = format(datetime, time_format))
  write.table(
    data,
    file = file_name,
    col.names = T,
    row.names = FALSE,
    quote = FALSE ,
    sep = ","
  )
}

# join
ref <- ref %>%
  left_join(rain, by = "datetime") %>%
  left_join(underdrain, by = "datetime") %>%
  left_join(overflow, by = "datetime") %>% # join rain, underdrain and overflow series
  mutate(year = year(datetime),
         month = month(datetime))

# fill missing rainfall data with 0
ind <- (ref$year < 2013) &
  (is.na(ref$rain)) # fill missing rainfall data in 2009 to 2012 with zero
# because 2013 has many missing rainfall
sum(ind) # number of missing rainfall fixed, 780 gap filled
ref$rain[ind] <- 0
ref <- ref %>%
  mutate(rain_fill_flag = F)
ref$rain_fill_flag[ind] <- T

# adding underdrain flow and overflow
ref <- ref %>%
  mutate(
    flow_fill_flag = (!is.na(underdrain)) &
      (is.na(overflow)),
    # underdrain not missing, overflow missing
    overflow_filled = replace(overflow, is.na(overflow), 0),
    # NA in overflow to 0
    flow = underdrain + overflow_filled
  ) %>%
  dplyr::select(-overflow_filled) %>%
  dplyr::select(
    datetime,
    rain,
    flow,
    year,
    month,
    underdrain,
    overflow,
    rain_fill_flag,
    flow_fill_flag
  ) %>% # joining underdrain and overflow
  arrange(datetime)

file_name <- paste0("./data/WS/processed_Geauga.csv")
write_data(ref, file_name) # write output


# Function ----------------------------------------------------------------

wet_dry <- function(datetime,
                    x,
                    IETD = 24,
                    before_n = 0) {
  # This function returns wet and dry index, using default 24 hour thd
  # Turn the irregular time series to regurlar one, this can aviod search for the duration
  # Vars:
  #   datetime: time index; x: observation series; IETD: inter-event time definition;
  #   before_n: before_n
  
  df <- tibble(datetime = datetime,
               x = x)
  
  full_df <-
    tibble(datetime = seq(
      from = datetime[[1]],
      to = tail(datetime, 1),
      by = "1 min"
    )) %>%
    full_join(df, by = "datetime") %>%
    mutate(wet = F) %>%
    arrange(datetime)
  
  rain_ind <- which(full_df$x != 0) # index where rain occurs
  max_le <- length(full_df$x)
  for (i in seq_along(rain_ind)) {
    ind_sta <-
      max(1, rain_ind[i] - before_n) # events before_n step ahead rain is wet
    ind_end <-
      min(max_le, (rain_ind[i] + IETD * 60)) # IETD convert to minutes
    full_df$wet[ind_sta:ind_end] <- T
  }
  
  df <- df %>%
    select(datetime) %>%
    left_join(full_df, by = "datetime") %>%
    arrange(datetime)
  
  # return wet/dry vector
  df$wet
}

mark_event <- function(x) {
  # This is a function mark the number of rainfall event based on wet_dry output
  # Vars: x is the wet vector
  
  out <- rep(0, length(x))
  
  temp <- rle(x)
  values <- temp$values
  lengths <- temp$lengths
  
  loc <- 1
  event <- 1
  for (i in seq_along(values)) {
    step <- lengths[[i]]
    if (values[[i]]) {
      out[loc:(loc + step - 1)] <- event
      event <- event + 1
    }
    loc <- loc + step
  }
  
  # dry period is marked as 0
  out
}

feature_vector <- function(x, s, e) {
  # This is a function to return vector of input feature,
  # s is the start location
  # e is the end location
  
  e <- e + 1
  
  if (s == 0) {
    out <- roll_sum(x, e, align = "right", fill = NA)
  } else {
    out <-
      roll_sum(x, e, align = "right", fill = NA) - roll_sum(x, s, align = "right", fill = NA)
  }
  
  out
}


# Process data for training -----------------------------------------------

# Qaulity check 1: the start of rainfall observation each year
ref %>%
  dplyr::filter(!rain_fill_flag) %>%
  group_by(year) %>%
  summarise(rain_ob_s = datetime[1])

# Qaulity check 2: number of rainfall records being filled
ref %>%
  filter(month %in% c(3:10)) %>%
  group_by(year) %>%
  summarise(rain_num_filled = sum(rain_fill_flag))

# Raw data + wet dry marked + event id marked by year + record id marked
# exclude 2009 events
data_process <- ref %>%
  filter(year != 2009) %>%
  group_by(year) %>%
  mutate(wet = wet_dry(datetime, rain, IETD = 24)) %>%
  mutate(event_id = mark_event(wet)) %>%
  ungroup() %>%
  mutate(event_id = paste(year, event_id, sep = "_")) %>%
  mutate(record_id = 1:n())

# Mark_trainable event
# Available events for ML:
#   when (1) April <= time <= Nov,
#        (2) wet
#        (3) flow not missing
#        (4) rainfall not missing


data_process <- data_process %>%
  rename(Y = flow,
         X = rain) %>%
  mutate(
    month = month(datetime),
    X_cum = feature_vector(X, s = 0, e = 1440),
    trainable = (month > 3) &
      (month < 11) &
      wet &
      !is.na(Y)  &
      !is.na(X_cum) & (datetime < ymd_hms("2013-07-20 00:40:00"))
  ) %>%
  select(-X_cum)

# Join CumRain feature

cum_rain <-
  tibble(datetime = seq(
    from = rain$datetime[1],
    to = rev(rain$datetime)[1],
    by = "10 min"
  )) %>%
  left_join(rain, by = "datetime") %>%
  arrange(datetime) %>%
  mutate(rain = replace(rain, is.na(rain), 0),
         cum_rain = cumsum(rain)) %>%
  select(datetime, cum_rain)

data_process <- data_process %>%
  left_join(cum_rain, by = "datetime") %>%
  arrange(datetime)

# Create dummy season vars and join
dmy <- dummyVars(" ~ month", data = data_process %>%
                   mutate(month = as.character(month)))

dmy_df <- data.frame(predict(dmy, newdata = data_process %>%
                               mutate(month = as.character(month)))) %>%
  tibble() %>%
  select(month4:month9, month10)

data_process <- data_process %>%
  bind_cols(dmy_df)

# Process and save data
trainable_event_id <- data_process %>%
  filter(trainable) %>%
  group_by(event_id) %>%
  count() %>%
  pull(event_id)
trainable_record_id <- data_process$record_id[data_process$trainable]  # 38,835 data points

trainable_df <- data_process %>%
  filter(trainable) %>%
  group_by(event_id) %>%
  summarise(
    record_id = list(record_id),
    datetime = datetime[1],
    peak_flow = max(Y),
    peak_rain = max(X),
    sum_flow = sum(Y),
    sum_rain = sum(X)
  ) %>%
  arrange(datetime) 

# Save data ---------------------------------------------------------------

save(data_process, trainable_df, file = "./data/WS/ready_for_training.Rda")
