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
  viridis,
  hrbrthemes
)



# Distribute SHAP ---------------------------------------------------------

set.seed(123)

n_steps <- 40
turn_off_ratio <- 0.6

data_plot <- tibble(
  time_step = 1:n_steps,
  contribution = runif(n_steps)*time_step^1.9
)
data_plot$contribution[sample(n_steps, n_steps*turn_off_ratio)] <- 0
data_plot$contribution <- data_plot$contribution/sum(data_plot$contribution)*10.5 # consistent with Figure 1

ggplot(data_plot, aes(time_step, contribution)) +
  geom_bar(stat="identity", fill = "steelblue") +
  scale_x_continuous(breaks = c(0:4)*10, labels = c("t-40","t-30","t-20","t-10","t")) +
  labs(y = "Impact on prediction",
       x = "Time step") +
  theme_minimal() +
  theme(axis.text.x = element_text(face = "italic"))

ggsave(filename = "./data/WS/SHAP_distribution_illustration.svg", width=5, height=2)




p <- tibble(
  variables = c(
    "Rainfall depth of past 1h",
    "Rainfall depth of past 24h",
    "Season",
    "SuDS age"
  ),
  SHAP = c(7, 2, 1, 0.5)
) %>%
  ggplot(aes(SHAP, reorder(variables, SHAP))) +
  geom_bar(stat = "identity", fill = "grey50") +
  labs(x = "SHAP value (impact on prediction)") +
  theme_minimal() +
  theme(axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 9))

ggsave(file="SHAP_illu.svg", plot=p, width=4, height=1.4)



load(file = "./data/WS/average_contri_ill.Rda")

data_plot %>%
  ggplot(aes(x-1,y))+
  geom_line(color = "coral")+
  labs(x = "Time step in the past", 
       y = "Impact on prediction",
       title = "Catchment response time estimation") +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 1, 5, 10, 50, 100, 500, 1000),
    labels = c("0", "1", " ", "10", " ", "100", " ", 1000)
  ) +
  coord_cartesian(xlim = c(0, 500))+
  theme_minimal()+
  theme(panel.grid.minor = element_blank(),
        axis.text.x = element_text(face = "italic"))

ggsave(filename = "./data/WS/response_time_illu.svg", width=4, height=2.5)




load("./data/WS/hydrograph_decomposition_illu.Rda")

ggplot(data_plot %>%
         filter(case == "Medium runoff event")) + 
  geom_area(aes(time_step-140, positive, fill = item), alpha=0.6 , size=0.1, colour="white")+
  geom_area(aes(time_step-140, negative, fill = item), alpha=0.6 , size=0.1, colour="white")+
  scale_fill_viridis(discrete = T,   
                     guide = guide_legend(
                       direction = "horizontal",
                       title.position = "top",
                       label.position = "right",
                       label.hjust = 0,
                       label.vjust = 1,
                       nrow = 1
                     ),
                     option = "B") +
  labs(x = "Time step",
       y = "Impact on prediction",
       title = "Hydrograph separation") +
  coord_cartesian(xlim = c(0, 60))+
  theme_minimal() +
  theme(legend.position = "none", 
        legend.key.size = unit(0.3, "cm"),
        legend.key.width = unit(0.4, "cm"),
        legend.text=element_text(size=rel(1.1), face = "italic"),
        panel.spacing = unit(1, "lines"),
        strip.background = element_blank(),
        strip.text.x = element_blank(),
        plot.title = element_text(hjust = 0.5))

ggsave(filename = "./data/WS/hydrograph_separation_illu.svg", width=4, height=2.5)


load("./data/WS/rainfall_age_illu.Rda")

data_plot %>%
  filter(item == "Average age of rainfall\naffecting prediction [h]",
         case == "Medium runoff event") %>%
  ggplot(aes(time_step - 140, value))+
  geom_line(color = "orange3")+
  labs(x = "Time step",
       y = "Age",
       title = "Determining the average age of \nthe rainfall that affects discharge prediction") +
  coord_cartesian(xlim = c(0, 60), ylim = c(0, NA))+
  theme_minimal() +
  theme(legend.position = "none", 
        legend.key.size = unit(0.3, "cm"),
        legend.key.width = unit(0.4, "cm"),
        legend.text=element_text(size=rel(1.1), face = "italic"),
        panel.spacing = unit(1, "lines"),
        strip.background = element_blank(),
        strip.text.x = element_blank(),
        plot.title = element_text(hjust = 0.5))
ggsave(filename = "./data/WS/rainfall_age_illu.svg", width=4, height=2.5)
