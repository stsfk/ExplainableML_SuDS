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

# Data --------------------------------------------------------------------

load("./data/WS/inconsist_exp/data_plot.Rda")

data_plot2 <- data_plot %>%
  mutate(prop = paste0(prop, "% training samples"),
         prop = factor(prop, levels = str_c(c(5,10,20,30), "% training samples"))) %>%
  group_by(prop, consistency) %>%
  summarise(n = n())%>%
  mutate(nse=-0.4,
         n = paste0("n = ", n),
         consistency = factor(consistency, labels = c("inconsistent", "consistent")))


data_plot %>%
  mutate(prop = paste0(prop, "% training samples"),
         prop = factor(prop, levels = str_c(c(5,10,20,30), "% training samples")),
         consistency = factor(consistency, labels = c("inconsistent", "consistent"))) %>%
  ggplot(aes(consistency, nse, color = consistency)) +
  geom_boxplot(outlier.size = 1) +
  geom_jitter(size = 0.75, shape = 1) +
  geom_text(data = data_plot2, aes(label = n), color = "black", fontface = "italic", size = 3) +
  facet_grid(~prop)+
  coord_capped_cart(bottom = brackets_horisontal(direction = "up", length = unit(0.21, "npc")))+
  scale_colour_Publication()+
  scale_color_manual(values = c("#E69F00", "#00AFBB")) +
  labs(y="NSE")+
  theme_Publication(base_size = 10)+
  theme(legend.position = "none", 
        panel.grid.major.x = element_blank(),
        panel.background = element_rect(fill = "grey97"),
        panel.spacing = unit(1, "lines"),
        axis.title.x = element_blank())

ggsave(filename = "./data/figures/figure9.png", width = 7, height = 3.25, units = "in", dpi = 600)
ggsave(filename = "./data/figures/figure9.pdf", width = 7, height = 3.25, units = "in")


