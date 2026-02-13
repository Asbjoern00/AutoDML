library(tidyverse)
data0 <- read_csv("dope_neural_nets/different_riesz_weights/tmle0_full_x.csv") %>%
  mutate(type = 'TMLE weight 0, Outcome MSE weight 1', index=1:50)
data05 <- read_csv("dope_neural_nets/different_riesz_weights/tmle05_full_x.csv") %>%
  mutate(type = 'TMLE weight 0.5, Outcome MSE weight 0.5', index=1:50)
data1 <- read_csv("dope_neural_nets/different_riesz_weights/tmle1_full_x.csv") %>%
  mutate(type = 'TMLE weight 1, Outcome MSE weight 0', index=1:50)

data = rbind(data0[1:30,],data05[1:30,], data1[1:30,])
ggplot(data) +
  geom_line(aes(x=index, y=estimate-truth))+
  facet_wrap(~type)+
  xlab('index')+
  ylab('estimation error')+
  theme_classic()+
  ggtitle('RieszNet estimation error in first 30 iterations')


